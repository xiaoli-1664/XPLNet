# Includes code adapted from XFeat; modified for XPLNet.
# See THIRD_PARTY_NOTICES.md and licenses/XFeat-Apache-2.0.txt.

from modules.line_description import LineDescription
from modules.interpolator import InterpolateSparse2d
from modules.verify import Verification
from modules.xplnet_new import XPLNetNew
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights/xplnet.pth"
DEFAULT_LINE_WEIGHTS = ROOT / "weights/ldesc.pth"


class XPLDetector(nn.Module):
    def __init__(self, weights=DEFAULT_WEIGHTS, top_k=4096, kp_th=0.05, junc_th=0.8,
                 line_th=0.8, resize_max=1600, dev=None,
                 line_desc_weights=None, junc_cov_radius=3,
                 junc_cov_temperature=1.0, junc_cov_floor=1e-4):
        super().__init__()
        if dev is None:
            self.dev = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.dev = dev

        self.net = XPLNetNew()
        self.sample = 32
        self.dim = 128
        self.verify = Verification(2*self.dim+(self.sample-2)*4)
        self.line_descriptor = LineDescription(
            samples=32, p_dim=64, l_dim=64)
        self.line_descriptor_loaded = False
        self.top_k = top_k
        self.kp_th = kp_th
        self.junc_th = junc_th
        self.line_th = line_th
        self.resize_max = resize_max
        if junc_cov_radius < 0:
            raise ValueError('junc_cov_radius must be non-negative')
        if junc_cov_temperature <= 0:
            raise ValueError('junc_cov_temperature must be positive')
        if junc_cov_floor < 0:
            raise ValueError('junc_cov_floor must be non-negative')
        self.junc_cov_radius = int(junc_cov_radius)
        self.junc_cov_temperature = float(junc_cov_temperature)
        self.junc_cov_floor = float(junc_cov_floor)

        if weights is not None:
            ckpt = torch.load(weights, map_location=self.dev, weights_only=True)
            self.net.load_state_dict(ckpt['xplnet'])
            self.verify.load_state_dict(ckpt['verify_coarse'])
            print(f'Loaded weights from {weights}')

        default_line_desc_weights = DEFAULT_LINE_WEIGHTS
        explicit_line_desc_weights = line_desc_weights is not None
        if line_desc_weights is None:
            line_desc_weights = default_line_desc_weights

        if line_desc_weights is not False:
            if not os.path.isfile(line_desc_weights):
                if explicit_line_desc_weights:
                    raise FileNotFoundError(
                        f'Line descriptor weights not found: '
                        f'{line_desc_weights}')
            else:
                line_desc_ckpt = torch.load(
                    line_desc_weights, map_location=self.dev, weights_only=True)
                line_desc_state = line_desc_ckpt.get(
                    'line_desc', line_desc_ckpt)
                self.line_descriptor.load_state_dict(line_desc_state)
                self.line_descriptor_loaded = True
                print(
                    f'Loaded line descriptor weights from '
                    f'{line_desc_weights}')

        self.interpolator = InterpolateSparse2d('bicubic')

    @torch.inference_mode()
    def forward(self, x, top_k=None, kp_th=None, junc_th=None,
                valid_mask=None, resize_mode='max_limit', line_desc=False):
        if top_k is None:
            top_k = self.top_k
        if kp_th is None:
            kp_th = self.kp_th
        if junc_th is None:
            junc_th = self.junc_th

        x, scale_h, scale_w, valid_mask = self.preprocess_tensor(
            x, resize_mode=resize_mode, resize_max=self.resize_max, valid_mask=valid_mask)

        B, _, H_, W_ = x.shape

        desc_map, keypoint, heatmap, junction_logits, edge_desc, junc_desc = self.net(
            x)
        junctions = torch.sigmoid(junction_logits)

        keypoint_h = self.get_kpts_heatmap(keypoint)

        if valid_mask is not None:
            keypoint_h = keypoint_h * valid_mask
            junctions = junctions * valid_mask

        mkpts = self.NMS(
            keypoint_h, threshold=kp_th, kernel_size=5)
        juncs_pts = self.NMS(
            junctions, threshold=junc_th, kernel_size=5, is_int=False)

        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')

        scores = (_nearest(keypoint_h, mkpts, H_, W_) *
                  _bilinear(heatmap, mkpts, H_, W_)).squeeze(-1)
        juncs_scores = _bilinear(
            junctions, juncs_pts, H_, W_).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1
        juncs_scores[torch.all(juncs_pts == 0, dim=-1)] = -1

        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.stack([mkpts_x, mkpts_y], -1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        # Keep every NMS junction. Invalid batch-padding entries are filtered
        # by juncs_valid below; slicing with -1 would discard one real peak for
        # single-image inference.
        juncs_topk = juncs_pts.shape[1]
        juncs_idxs = torch.argsort(-juncs_scores)
        juncs_x = torch.gather(
            juncs_pts[..., 0], -1, juncs_idxs)[:, :juncs_topk]
        juncs_y = torch.gather(
            juncs_pts[..., 1], -1, juncs_idxs)[:, :juncs_topk]
        juncs_pts = torch.stack([juncs_x, juncs_y], -1)  # [B, N, 2]
        juncs_scores = torch.gather(
            juncs_scores, -1, juncs_idxs)[:, :juncs_topk]

        valid = scores > 0

        juncs_valid = juncs_scores > 0

        juncs_pts, juncs_covariances = \
            self.refine_junctions_and_covariances(
                junction_logits, juncs_pts, juncs_valid,
                spatial_valid_mask=valid_mask)

        desc_map = F.normalize(desc_map, dim=1)
        (lines_batch, scores_batch, line_descs_batch,
         line_endpoint_covariances_batch) = self.process_lines(
            desc_map, edge_desc, junc_desc, juncs_pts,
            juncs_covariances, juncs_valid, line_desc=line_desc)

        descs = self.interpolator(desc_map, mkpts, H=H_, W=W_)
        descs = F.normalize(descs, dim=-1)

        image_scale = torch.tensor(
            [scale_w, scale_h], device=juncs_pts.device,
            dtype=juncs_pts.dtype)
        mkpts = mkpts * image_scale.view(1, 1, 2)
        juncs_pts = juncs_pts * image_scale.view(1, 1, 2)
        juncs_covariances = self.scale_covariances(
            juncs_covariances, image_scale)

        desc_map = F.interpolate(desc_map, size=(int(
            desc_map.shape[2]*scale_h), int(desc_map.shape[3]*scale_w)), mode='bicubic', align_corners=False)

        line_scale = image_scale.repeat(2)
        for b, lines in enumerate(lines_batch):
            lines[:, :, 0] = lines[:, :, 0] * scale_w
            lines[:, :, 1] = lines[:, :, 1] * scale_h
            line_endpoint_covariances_batch[b] = self.scale_covariances(
                line_endpoint_covariances_batch[b], line_scale)

        outputs = [{'keypoints': mkpts[b, valid[b]],
                    'scores': scores[b, valid[b]],
                    'descriptors': descs[b, valid[b]],
                    'junctions': juncs_pts[b, juncs_valid[b]],
                    'junction_scores': juncs_scores[b, juncs_valid[b]],
                    'junction_covariances':
                        juncs_covariances[b, juncs_valid[b]],
                    'lines': lines_batch[b],
                    'lines_scores': scores_batch[b],
                    'line_endpoint_covariances':
                        line_endpoint_covariances_batch[b],
                    'desc_map': desc_map[b]}
                   for b in range(B)]

        if line_desc:
            for b in range(B):
                outputs[b]['line_descriptors'] = line_descs_batch[b]
                outputs[b]['lines_descriptors'] = line_descs_batch[b]

        return outputs

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def NMS2d(self, x, threshold=0.05, kernel_size=5):
        local_max = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size//2)(x)
        keep = (x == local_max) & (x > threshold)
        return x * keep

    def NMS(self, x, threshold=0.05, kernel_size=5, is_int=True, gaussian_blur=False):
        if gaussian_blur:
            x = kornia.filters.gaussian_blur2d(
                x, kernel_size=(5, 5), sigma=(0.5, 0.5))
        B, _, H, W = x.shape
        pad = kernel_size//2
        local_max = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        if is_int:
            pos = torch.zeros(
                (B, pad_val, 2), dtype=torch.long, device=x.device)
        else:
            pos = torch.zeros(
                (B, pad_val, 2), dtype=torch.float32, device=x.device)

        # Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos

    def refine_junctions_and_covariances(
            self, junction_logits, junctions, junction_valid,
            spatial_valid_mask=None):
        """Refine junction peaks and estimate their local 2D covariance.

        A local softmax over the pre-sigmoid junction logits defines a
        categorical distribution in a square window around every integer NMS
        peak. Its first and second moments give a sub-pixel location and a
        covariance in the resized-image pixel coordinate system.

        Args:
            junction_logits: Tensor of shape [B, 1, H, W].
            junctions: NMS peak coordinates of shape [B, N, 2] in (x, y)
                order.
            junction_valid: Boolean tensor of shape [B, N].
            spatial_valid_mask: Optional tensor of shape [B, 1, H, W].

        Returns:
            Refined coordinates [B, N, 2] and covariances [B, N, 2, 2].
            Invalid padded junctions are returned as zeros.
        """
        if junction_logits.ndim != 4 or junction_logits.shape[1] != 1:
            raise ValueError('junction_logits must have shape [B, 1, H, W]')
        if junctions.ndim != 3 or junctions.shape[-1] != 2:
            raise ValueError('junctions must have shape [B, N, 2]')
        if junction_valid.shape != junctions.shape[:2]:
            raise ValueError('junction_valid must have shape [B, N]')

        B, _, H, W = junction_logits.shape
        if junctions.shape[0] != B:
            raise ValueError('junction batch size does not match logits')

        radius = self.junc_cov_radius
        offset_axis = torch.arange(
            -radius, radius + 1, device=junctions.device,
            dtype=junctions.dtype)
        offset_y, offset_x = torch.meshgrid(
            offset_axis, offset_axis, indexing='ij')
        offsets = torch.stack([offset_x, offset_y], dim=-1).reshape(1, 1, -1, 2)

        sample_coords = junctions.unsqueeze(2) + offsets
        sample_x = sample_coords[..., 0].round().long()
        sample_y = sample_coords[..., 1].round().long()
        sample_valid = ((sample_x >= 0) & (sample_x < W)
                        & (sample_y >= 0) & (sample_y < H))

        sample_x_clamped = sample_x.clamp(0, W - 1)
        sample_y_clamped = sample_y.clamp(0, H - 1)
        batch_idx = torch.arange(
            B, device=junctions.device).view(B, 1, 1)
        local_logits = junction_logits[:, 0][
            batch_idx, sample_y_clamped, sample_x_clamped]

        if spatial_valid_mask is not None:
            if spatial_valid_mask.shape != junction_logits.shape:
                raise ValueError(
                    'spatial_valid_mask must match junction_logits shape')
            local_spatial_valid = spatial_valid_mask[:, 0][
                batch_idx, sample_y_clamped, sample_x_clamped] > 0.5
            sample_valid = sample_valid & local_spatial_valid

        local_logits = local_logits.masked_fill(~sample_valid, -torch.inf)
        weights = F.softmax(
            local_logits / self.junc_cov_temperature, dim=-1)

        # A valid NMS peak is inside the image, so every valid junction has at
        # least its center sample. Avoid NaNs for malformed/padded entries.
        finite_weights = torch.isfinite(weights).all(dim=-1, keepdim=True)
        weights = torch.where(finite_weights, weights, torch.zeros_like(weights))

        refined = (weights.unsqueeze(-1) * sample_coords).sum(dim=2)
        centered = sample_coords - refined.unsqueeze(2)
        covariances = (
            weights.unsqueeze(-1).unsqueeze(-1)
            * centered.unsqueeze(-1)
            * centered.unsqueeze(-2)).sum(dim=2)

        if self.junc_cov_floor > 0:
            covariances = covariances + self.junc_cov_floor * torch.eye(
                2, device=covariances.device,
                dtype=covariances.dtype).view(1, 1, 2, 2)

        valid_xy = junction_valid.bool().unsqueeze(-1)
        refined = torch.where(valid_xy, refined, torch.zeros_like(refined))
        covariances = torch.where(
            valid_xy.unsqueeze(-1), covariances,
            torch.zeros_like(covariances))
        return refined, covariances

    @staticmethod
    def scale_covariances(covariances, coordinate_scale):
        """Apply a diagonal coordinate scaling to covariance matrices."""
        if covariances.ndim < 2:
            raise ValueError('covariances must have at least two dimensions')
        dimension = covariances.shape[-1]
        if covariances.shape[-2:] != (dimension, dimension):
            raise ValueError('covariances must contain square matrices')
        if coordinate_scale.ndim != 1 or coordinate_scale.numel() != dimension:
            raise ValueError(
                'coordinate_scale must match the covariance dimension')

        scale = coordinate_scale.to(
            device=covariances.device, dtype=covariances.dtype)
        leading_dims = [1] * (covariances.ndim - 2)
        row_scale = scale.view(*leading_dims, dimension, 1)
        col_scale = scale.view(*leading_dims, 1, dimension)
        return covariances * row_scale * col_scale

    def preprocess_tensor(self, x, resize_mode='original', resize_max=1024, valid_mask=None):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = torch.tensor(x).permute(2, 0, 1)[None]
            elif len(x.shape) == 2:
                x = torch.tensor(x[..., None]).permute(2, 0, 1)[None]
            else:
                raise RuntimeError('Input numpy array must be 2D or 3D')

        if len(x.shape) != 4:
            raise RuntimeError('Input tensor must be (B, C, H, W)')

        x = x.to(self.dev).float()

        if x.max() > 1.5:
            x = x / 255.

        h, w = x.shape[-2:]

        if resize_mode == 'force_max':
            scale = resize_max / max(h, w)
            new_h, new_w = h * scale, w * scale
        elif resize_mode == 'max_limit':
            if max(h, w) > resize_max:
                scale = resize_max / max(h, w)
                new_h, new_w = h * scale, w * scale
            else:
                new_h, new_w = h, w
        else:
            new_h, new_w = h, w

        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        self.width = int(new_w)
        self.height = int(new_h)

        scale_h, scale_w = h / self.height, w / self.width

        x = F.interpolate(
            x, size=(self.height, self.width), mode='bilinear', align_corners=False)
        valid_mask_resized = None
        if valid_mask is not None:
            valid_mask = valid_mask.to(self.dev).float()
            valid_mask_resized = F.interpolate(
                valid_mask, size=(self.height, self.width), mode='nearest')

        return x, scale_h, scale_w, valid_mask_resized

    def parse_input(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0, 3, 1, 2)/255

        x = x.to(self.dev)

        return x

    @torch.inference_mode()
    def match(self, feats1, feats2, min_cossim=0.82):
        if feats1.shape[0] == 0 or feats2.shape[0] == 0:
            empty = torch.empty(
                0, dtype=torch.long, device=feats1.device)
            return empty, empty.clone()

        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1


    @torch.inference_mode()
    def match_xpl(self, img1, img2, top_k=None, min_cossim=-1,
                  valid_mask1=None, valid_mask2=None,
                  resize_mode='original', line_match=False,
                  line_min_cossim=None):
        """
                Simple extractor and MNN matcher.
                For simplicity it does not support batched mode due to possibly different number of kpts.
                input:
                        img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                        img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                        top_k -> int: keep best k features
                returns:
                        mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        if top_k is None:
            top_k = self.top_k
        img1 = self.parse_input(img1)
        img2 = self.parse_input(img2)

        out1 = self.forward(
            img1, top_k=top_k, valid_mask=valid_mask1,
            resize_mode=resize_mode, line_desc=line_match)[0]

        out2 = self.forward(
            img2, top_k=top_k, valid_mask=valid_mask2,
            resize_mode=resize_mode, line_desc=line_match)[0]

        idxs0_pts, idxs1_pts = self.match(
            out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)

        if line_match:
            if line_min_cossim is None:
                line_min_cossim = min_cossim
            idxs0_lines, idxs1_lines = self.match(
                out1['line_descriptors'], out2['line_descriptors'],
                min_cossim=line_min_cossim)

        results = {
            "keypoints1": out1['keypoints'][idxs0_pts].cpu().numpy(),
            "keypoints2": out2['keypoints'][idxs1_pts].cpu().numpy(),
        }

        if line_match:
            results.update({
                'lines1': out1['lines'][idxs0_lines].cpu().numpy(),
                'lines2': out2['lines'][idxs1_lines].cpu().numpy(),
                'line_endpoint_covariances1':
                    out1['line_endpoint_covariances'][idxs0_lines]
                    .cpu().numpy(),
                'line_endpoint_covariances2':
                    out2['line_endpoint_covariances'][idxs1_lines]
                    .cpu().numpy(),
            })

        return results

    @torch.inference_mode()
    def match_point(self, img1, img2, top_k=None, min_cossim=-1, valid_mask1=None, valid_mask2=None):
        """
                Simple extractor and MNN matcher.
                For simplicity it does not support batched mode due to possibly different number of kpts.
                input:
                        img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                        img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                        top_k -> int: keep best k features
                returns:
                        mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        if top_k is None:
            top_k = self.top_k
        img1 = self.parse_input(img1)
        img2 = self.parse_input(img2)

        out1 = self.forward(img1, top_k=top_k,
                            valid_mask=valid_mask1, resize_mode='original')[0]

        out2 = self.forward(img2, top_k=top_k,
                            valid_mask=valid_mask2, resize_mode='original')[0]

        idxs0_pts, idxs1_pts = self.match(
            out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)

        return out1['keypoints'][idxs0_pts].cpu().numpy(), out2['keypoints'][idxs1_pts].cpu().numpy()

    def sample_segments(self, junctions, junc_valid, num_sample):
        '''
        Sample points along line segments defined by junctions.
        '''

        B, N, _ = junctions.shape
        device = junctions.device

        # All undirected pairs (i < j)
        idx = torch.triu_indices(N, N, offset=1, device=device)
        i_idx, j_idx = idx[0], idx[1]
        M = i_idx.numel()

        if M == 0:
            sampled_points = junctions.new_zeros((B, num_sample, 0, 2))
            seg_valid = junc_valid.new_zeros((B, 0)).bool()
            return sampled_points, seg_valid

        # Gather segment endpoints
        p0 = junctions[:, i_idx, :]  # [B, M, 2]
        p1 = junctions[:, j_idx, :]  # [B, M, 2]

        # Valid segments require both endpoints valid
        jv = junc_valid.bool()
        seg_valid = jv[:, i_idx] & jv[:, j_idx]  # [B, M]

        # Sample uniformly along each segment
        t = torch.linspace(0.0, 1.0, steps=num_sample,
                           device=device, dtype=torch.float)  # [S]
        t = t.view(1, num_sample, 1, 1)
        p0e = p0.unsqueeze(1)  # [B, 1, M, 2]
        p1e = p1.unsqueeze(1)  # [B, 1, M, 2]
        sampled_points = p0e * (1.0 - t) + p1e * t  # [B, S, M, 2]

        return sampled_points, seg_valid

    def sample_segments_map(self, junctions, line_map, num_sample):
        '''
        Sample points along line segments defined by junctions and line_map.
        '''

        B, N, _ = junctions.shape
        device = junctions.device

        # M = N * (N - 1) / 2
        idx = torch.triu_indices(N, N, offset=1, device=device)
        i_idx, j_idx = idx[0], idx[1]
        M = i_idx.numel()

        if M == 0:
            sampled_points = junctions.new_zeros((B, num_sample, 0, 2))
            seg_valid = line_map.new_zeros((B, 0)).bool()
            return sampled_points, seg_valid

        # [B, N, 2] -> [B, M, 2]
        p0 = junctions[:, i_idx, :]
        p1 = junctions[:, j_idx, :]

        seg_valid_values = line_map[:, i_idx, j_idx]

        seg_valid = seg_valid_values.bool()  # [B, M]

        # t: [1, S, 1, 1]
        t = torch.linspace(0.0, 1.0, steps=num_sample,
                           device=device, dtype=torch.float)
        t = t.view(1, num_sample, 1, 1)

        p0e = p0.unsqueeze(1)
        p1e = p1.unsqueeze(1)

        sampled_points = p0e * (1.0 - t) + p1e * t

        return sampled_points, seg_valid

    def process_lines(self, desc_map, edge_desc, junc_desc,
                      junctions, junction_covariances, junc_valid,
                      line_desc=False):

        B = desc_map.shape[0]
        device = edge_desc.device
        line_desc_dim = self.line_descriptor.l_dim
        expected_covariance_shape = (*junctions.shape[:2], 2, 2)
        if junction_covariances.shape != expected_covariance_shape:
            raise ValueError(
                'junction_covariances must have shape [B, N, 2, 2]')

        if line_desc and not self.line_descriptor_loaded:
            raise RuntimeError(
                'Line descriptors were requested, but no trained line '
                'descriptor weights are loaded')

        def empty_outputs():
            lines = [torch.empty(
                0, 2, 2, device=device, dtype=junctions.dtype)
                for _ in range(B)]
            scores = [torch.empty(
                0, device=device, dtype=desc_map.dtype)
                for _ in range(B)]
            descriptors = [torch.empty(
                0, line_desc_dim, device=device, dtype=desc_map.dtype)
                for _ in range(B)]
            covariances = [torch.empty(
                0, 4, 4, device=device, dtype=junctions.dtype)
                for _ in range(B)]
            return lines, scores, descriptors, covariances

        line_map = self.generate_line_map(junc_valid)  # [B, N, N]
        sampled_points, seg_valid = self.sample_segments_map(
            junctions, line_map, self.sample)  # [B, S, M, 2], [B, M]
        B, S, M, _ = sampled_points.shape

        sampled_descs = self.verification_preprocess(
            sampled_points, edge_desc, junc_desc)
        if sampled_descs.shape[2] == 0:
            return empty_outputs()
        verification = self.verify(sampled_descs)  # [B, 2, M]
        verification = F.softmax(verification, dim=1)
        verification = verification[:, 1, :]  # [B, M]

        # Build line map from verified segments
        idx = torch.triu_indices(junctions.shape[1], junctions.shape[1], offset=1,
                                 device=device)
        i_idx, j_idx = idx[0], idx[1]  # [M]

        b_idx, n_idx = torch.where(seg_valid)

        if b_idx.shape[0] == 0:
            return empty_outputs()

        i_valid = i_idx[n_idx]
        j_valid = j_idx[n_idx]

        p0 = junctions[b_idx, i_valid, :]  # [TotalValidLines, 2]
        p1 = junctions[b_idx, j_valid, :]  # [TotalValidLines, 2]
        lines = torch.stack([p0, p1], dim=1)  # [TotalValidLines, 2, 2]
        line_endpoint_covariances = junction_covariances.new_zeros(
            (b_idx.shape[0], 4, 4))
        line_endpoint_covariances[:, :2, :2] = \
            junction_covariances[b_idx, i_valid]
        line_endpoint_covariances[:, 2:, 2:] = \
            junction_covariances[b_idx, j_valid]
        lines_sample = sampled_points[b_idx, :, n_idx, :]
        scores = verification[b_idx, n_idx]

        scores_mask = scores > self.line_th
        final_lines = lines[scores_mask]
        final_scores = scores[scores_mask]
        final_lines_sample = lines_sample[scores_mask]
        final_line_endpoint_covariances = \
            line_endpoint_covariances[scores_mask]
        final_b_idx = b_idx[scores_mask]

        lines_batch = []
        scores_batch = []
        line_descs_batch = []
        line_endpoint_covariances_batch = []
        for b in range(B):
            b_mask = final_b_idx == b
            lines_batch.append(final_lines[b_mask])
            scores_batch.append(final_scores[b_mask])
            line_endpoint_covariances_batch.append(
                final_line_endpoint_covariances[b_mask])

            if line_desc and b_mask.any():
                samples_b = final_lines_sample[b_mask]
                samples_b = samples_b.permute(1, 0, 2).unsqueeze(0)
                line_features = self.line_description_preprocess(
                    samples_b, desc_map[b:b + 1])
                line_features = line_features.permute(0, 1, 3, 2)
                descriptors_b = self.line_descriptor(line_features)[0]
                line_descs_batch.append(descriptors_b)
            else:
                line_descs_batch.append(torch.empty(
                    0, line_desc_dim, device=device,
                    dtype=desc_map.dtype))

        return (lines_batch, scores_batch, line_descs_batch,
                line_endpoint_covariances_batch)

    def generate_line_map(self, junc_valid):
        mask_col = junc_valid.unsqueeze(2)  # [B, N, 1]
        mask_row = junc_valid.unsqueeze(1)  # [B, 1, N]
        line_map = mask_col & mask_row  # [B, N, N]
        line_map = torch.triu(line_map, diagonal=1)
        return line_map

    def line_description_preprocess(self, lines_sample, desc):
        _, _, H, W = desc.shape

        x = lines_sample[..., 0]
        y = lines_sample[..., 1]
        x = 2.0 * (x / (8.0 * W - 1)) - 1.0
        y = 2.0 * (y / (8.0 * H - 1)) - 1.0
        grid = torch.stack((
            x.clamp(-1.0 + 1e-6, 1.0 - 1e-6),
            y.clamp(-1.0 + 1e-6, 1.0 - 1e-6),
        ), dim=-1)
        return F.grid_sample(
            desc, grid, mode='bilinear', align_corners=False)

    def verification_preprocess(self, lines_sample, edge_desc, junc_desc):
        B, M, N, _ = lines_sample.shape
        _, C_junc, H, W = junc_desc.shape
        _, C_edge, _, _ = edge_desc.shape

        x = lines_sample[..., 0]
        y = lines_sample[..., 1]

        w_factor = 4.0 * W - 1
        h_factor = 4.0 * H - 1

        grid = torch.stack([
            x.div(w_factor).mul_(2.0).sub_(
                1.0).clamp_(-1.0 + 1e-6, 1.0 - 1e-6),
            y.div(h_factor).mul_(2.0).sub_(1.0).clamp_(-1.0 + 1e-6, 1.0 - 1e-6)
        ], dim=-1)  # [B, M, N, 2]

        grid_junc = torch.stack(
            [grid[:, 0], grid[:, -1]], dim=1)  # [B, 2, N, 2]
        grid_edge = grid[:, 1:-1]  # [B, M-2, N, 2]

        feat_junc = F.grid_sample(junc_desc, grid_junc,
                                  mode='bilinear', align_corners=False)
        feat_edge = F.grid_sample(edge_desc, grid_edge,
                                  mode='bilinear', align_corners=False)

        feat_start = feat_junc[:, :, 0]
        feat_end = feat_junc[:, :, 1]

        feat_edge_flat = feat_edge.flatten(1, 2)

        line_desc = torch.cat([
            feat_start,
            feat_edge_flat,
            feat_end
        ], dim=1)

        return line_desc
