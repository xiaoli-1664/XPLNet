import torch.nn as nn
import torch.nn.functional as F


class LineDescription(nn.Module):
    def __init__(self, samples=32, p_dim=64, l_dim=64, stride=2,
                 hidden_dim=192, bottleneck_dim=96):
        super().__init__()
        if samples % stride != 0:
            raise ValueError(
                f'samples ({samples}) must be divisible by stride ({stride})')

        self.samples = samples
        self.p_dim = p_dim
        self.l_dim = l_dim
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        pooled_samples = samples // stride
        self.max_pool = nn.MaxPool1d(kernel_size=stride, stride=stride)
        self.mlp = nn.Sequential(
            nn.Linear(p_dim * pooled_samples, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, l_dim),
        )

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'Expected input shape [B, C, N, L], got {tuple(x.shape)}')

        _, C, _, L = x.shape
        if L != self.samples:
            raise ValueError(
                f'Input length L ({L}) does not match expected samples '
                f'({self.samples})')
        if C != self.p_dim:
            raise ValueError(
                f'Input channel C ({C}) does not match expected p_dim '
                f'({self.p_dim})')

    def forward(self, x):
        self._check_input(x)
        B, C, N, L = x.shape
        if N == 0:
            return x.new_empty(B, 0, self.l_dim)

        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, C, L)
        x = self.max_pool(x)
        x = x.view(B * N, C * (L // self.stride))
        x = self.mlp(x)
        x = x.view(B, N, self.l_dim)
        return F.normalize(x, p=2, dim=-1)
