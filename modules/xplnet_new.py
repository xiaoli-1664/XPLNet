# Includes code adapted from XFeat; modified for XPLNet.
# See THIRD_PARTY_NOTICES.md and licenses/XFeat-Apache-2.0.txt.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, groups=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, dilation, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class RepConvBranch(nn.Module):
    """Re-parameterizable local branch with automatic mode selection.

    Training uses the 3x3, 1x1, and identity paths. Evaluation automatically
    uses their equivalent fused 3x3 convolution without deleting the training
    branches, so ``train()`` can switch back safely. ``switch_to_deploy()``
    performs a permanent conversion for serialization and deployment.
    """

    def __init__(self, channels, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        self.register_buffer('_eval_kernel', None, persistent=False)
        self.register_buffer('_eval_bias', None, persistent=False)

        if deploy:
            self.reparam_conv = nn.Conv2d(
                channels, channels, 3, padding=1, bias=True)
        else:
            self.conv3 = nn.Conv2d(
                channels, channels, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(channels)
            self.conv1 = nn.Conv2d(
                channels, channels, 1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        if bn.running_mean is None or bn.running_var is None:
            raise RuntimeError(
                'RepConvBranch requires BatchNorm running statistics for fusion')

        kernel = conv.weight
        if conv.bias is None:
            conv_bias = torch.zeros_like(bn.running_mean)
        else:
            conv_bias = conv.bias

        gamma = bn.weight if bn.weight is not None else torch.ones_like(
            bn.running_mean)
        beta = bn.bias if bn.bias is not None else torch.zeros_like(
            bn.running_mean)
        scale = gamma / torch.sqrt(bn.running_var + bn.eps)
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = beta + (conv_bias - bn.running_mean) * scale
        return fused_kernel, fused_bias

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.reparam_conv.weight, self.reparam_conv.bias

        kernel3, bias3 = self._fuse_conv_bn(self.conv3, self.bn3)
        kernel1, bias1 = self._fuse_conv_bn(self.conv1, self.bn1)
        kernel1 = F.pad(kernel1, (1, 1, 1, 1))

        identity_kernel = torch.zeros_like(kernel3)
        indices = torch.arange(self.channels, device=kernel3.device)
        identity_kernel[indices, indices, 1, 1] = 1.0
        identity_bias = torch.zeros_like(bias3)

        kernel = kernel3 + kernel1 + identity_kernel
        bias = bias3 + bias1 + identity_bias
        return kernel, bias

    def _invalidate_eval_cache(self):
        self._eval_kernel = None
        self._eval_bias = None

    @torch.no_grad()
    def _refresh_eval_cache(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self._eval_kernel = kernel.detach()
        self._eval_bias = bias.detach()

    def _forward_branches(self, x):
        return self.bn3(self.conv3(x)) + self.bn1(self.conv1(x)) + x

    @staticmethod
    def _forward_fused(x, kernel, bias):
        return F.conv2d(x, kernel, bias, stride=1, padding=1)

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam_conv(x))

        if self.training:
            return self.act(self._forward_branches(x))

        if torch.is_grad_enabled():
            kernel, bias = self.get_equivalent_kernel_bias()
        else:
            if self._eval_kernel is None or self._eval_bias is None:
                self._refresh_eval_cache()
            kernel, bias = self._eval_kernel, self._eval_bias
        return self.act(self._forward_fused(x, kernel, bias))

    def train(self, mode=True):
        super().train(mode)
        if not self.deploy:
            if mode:
                self._invalidate_eval_cache()
            else:
                self._refresh_eval_cache()
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        self._invalidate_eval_cache()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return self

        self.eval()
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = nn.Conv2d(
            self.channels, self.channels, 3, padding=1, bias=True,
            device=kernel.device, dtype=kernel.dtype)
        reparam_conv.weight.copy_(kernel)
        reparam_conv.bias.copy_(bias)
        reparam_conv.train(self.training)
        self.reparam_conv = reparam_conv

        del self.conv3
        del self.bn3
        del self.conv1
        del self.bn1
        self.deploy = True
        self._invalidate_eval_cache()
        return self


class RepPConvBlock(nn.Module):
    """Partial-channel local mixing for latency-friendly neck refinement."""

    def __init__(self, channels, n_div=4, deploy=False):
        super().__init__()
        self.partial_channels = max(1, channels // n_div)
        self.remaining_channels = channels - self.partial_channels
        self.partial_mixer = RepConvBranch(
            self.partial_channels, deploy=deploy)
        self.channel_mixer = BasicLayer(channels, channels, 1, padding=0)

    def forward(self, x):
        x_active, x_pass = torch.split(
            x, [self.partial_channels, self.remaining_channels], dim=1)
        x_active = self.partial_mixer(x_active)
        return self.channel_mixer(torch.cat([x_active, x_pass], dim=1))


class DynamicScaleRouter(nn.Module):
    """Task-conditioned routing over aligned multi-scale features."""

    def __init__(self, channels, num_tasks=3, num_scales=3, hidden_dim=64):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_scales = num_scales
        self.router = nn.Sequential(
            nn.Linear(channels * num_scales, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_tasks * num_scales),
        )

    def forward(self, features):
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in features]
        route_input = torch.cat(pooled, dim=1)
        B = route_input.shape[0]
        weights = self.router(route_input).view(
            B, self.num_tasks, self.num_scales)
        weights = F.softmax(weights, dim=-1)

        stacked = torch.stack(features, dim=1)  # [B, S, C, H, W]
        routed = []
        for task_idx in range(self.num_tasks):
            w = weights[:, task_idx, :, None, None, None]
            routed.append((w * stacked).sum(dim=1))
        return routed


class LowRankTaskAdapter(nn.Module):
    """Small task-private residual adapter inspired by low-rank adaptation."""

    def __init__(self, channels, rank=8):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, padding=0, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(rank, channels, 1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.up(self.act(self.down(x)))


class CrossTaskGatedFusion(nn.Module):
    """Low-cost task-dimension interaction for point, junction, and line."""

    def __init__(self, channels, num_tasks=3, hidden_dim=16):
        super().__init__()
        self.num_tasks = num_tasks
        self.source_proj = nn.ModuleList([
            BasicLayer(channels, channels, 1, padding=0)
            for _ in range(num_tasks)
        ])
        self.pair_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, padding=0, bias=True),
        )
        self.alpha = nn.Parameter(torch.full((num_tasks,), 0.1))
        self.out_proj = nn.ModuleList([
            BasicLayer(channels, channels, 1, padding=0)
            for _ in range(num_tasks)
        ])

    def forward(self, features):
        projected = [proj(feat)
                     for proj, feat in zip(self.source_proj, features)]
        outputs = []
        denom = max(self.num_tasks - 1, 1)

        for target_idx, target in enumerate(features):
            update = 0.0
            for source_idx, source in enumerate(features):
                if source_idx == target_idx:
                    continue
                gate = torch.sigmoid(self.pair_gate(
                    torch.cat([target, source], dim=1)))
                update = update + gate * projected[source_idx]

            scale = self.alpha[target_idx].view(1, 1, 1, 1)
            fused = target + scale * update / denom
            outputs.append(self.out_proj[target_idx](fused))

        return outputs


class DGTANeck(nn.Module):
    """Dynamic Geometry-aware Task Adapter neck.

    The neck keeps the old model's stride-8 output contract while replacing
    fixed concatenation with task-conditioned scale routing and lightweight
    cross-task feature exchange.
    """

    def __init__(self, in_channels=64, bottleneck_channels=48,
                 out_channels=64, deploy=False):
        super().__init__()
        c = bottleneck_channels
        self.align = nn.ModuleList([
            BasicLayer(in_channels, c, 1, padding=0),
            BasicLayer(in_channels, c, 1, padding=0),
            BasicLayer(in_channels, c, 1, padding=0),
        ])
        self.local_mix = nn.ModuleList([
            RepPConvBlock(c, deploy=deploy),
            RepPConvBlock(c, deploy=deploy),
            RepPConvBlock(c, deploy=deploy),
        ])
        self.router = DynamicScaleRouter(c, num_tasks=3, num_scales=3)
        self.point_adapter = LowRankTaskAdapter(c, rank=8)
        self.junction_adapter = LowRankTaskAdapter(c, rank=8)
        self.line_adapter = LowRankTaskAdapter(c, rank=12)
        self.cross_task = CrossTaskGatedFusion(c, num_tasks=3)
        self.out_proj = nn.ModuleList([
            BasicLayer(c, out_channels, 1, padding=0),
            BasicLayer(c, out_channels, 1, padding=0),
            BasicLayer(c, out_channels, 1, padding=0),
        ])

    def forward(self, x3, x4, x5):
        out_size = x3.shape[-2:]

        s3 = self.align[0](x3)
        s4 = F.interpolate(self.align[1](x4), out_size,
                           mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.align[2](x5), out_size,
                           mode='bilinear', align_corners=False)

        s3 = self.local_mix[0](s3)
        s4 = self.local_mix[1](s4)
        s5 = self.local_mix[2](s5)

        point_base, junction_base, line_base = self.router([s3, s4, s5])

        point_feat = self.point_adapter(point_base)
        junction_feat = self.junction_adapter(junction_base)
        line_feat = self.line_adapter(line_base)

        point_feat, junction_feat, line_feat = self.cross_task(
            [point_feat, junction_feat, line_feat])

        point_feat = self.out_proj[0](point_feat)
        junction_feat = self.out_proj[1](junction_feat)
        line_feat = self.out_proj[2](line_feat)

        return point_feat, junction_feat, line_feat


class XPLNetNew(nn.Module):
    def __init__(self, resolution=(608, 800), deploy=False):
        super().__init__()
        self.resolution = resolution
        self.deploy = deploy

        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4),
                                   nn.Conv2d(1, 24, 1, stride=1, padding=0))

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )

        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.dgta_neck = DGTANeck(
            in_channels=64, bottleneck_channels=48, out_channels=64,
            deploy=deploy)

        self.desc_head = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0),
        )

        self.junc_desc_head = nn.Sequential(
            BasicLayer(64, 512, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(128, 128, 1, padding=0),
        )

        self.edge_desc_head = nn.Sequential(
            BasicLayer(64, 256, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 4, 1, padding=0),
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        self.junctions_head = nn.Sequential(
            BasicLayer(128, 64, stride=1),
            nn.PixelShuffle(2),
            BasicLayer(16, 16, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 1, 1, padding=0),
        )

    def _unfold2d(self, x, ws=2):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(
            3, ws, ws).reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Permanently fuse all RepConv branches for model export."""
        self.eval()
        for module in list(self.modules()):
            if isinstance(module, RepConvBranch):
                module.switch_to_deploy()
        self.deploy = True
        return self

    def forward(self, x):
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        x1 = self.block1(x)  # [B, 24, H/4, W/4]
        x2 = self.block2(x1 + self.skip1(x))  # [B, 24, H/4, W/4]
        x3 = self.block3(x2)  # [B, 64, H/8, W/8]
        x4 = self.block4(x3)  # [B, 64, H/16, W/16]
        x5 = self.block5(x4)  # [B, 64, H/32, W/32]

        feats_point, feats_junction, feats_line = self.dgta_neck(x3, x4, x5)

        descs = self.desc_head(feats_point)
        heatmap = self.heatmap_head(descs.detach())

        edge_desc = self.edge_desc_head(feats_line)

        junc_desc = self.junc_desc_head(feats_junction)
        junctions = self.junctions_head(junc_desc)

        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))

        return descs, keypoints, heatmap, junctions, edge_desc, junc_desc


# Drop-in alias for experiments that import XPLNet from this module.
XPLNet = XPLNetNew
