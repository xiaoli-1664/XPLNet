import torch.nn as nn


class Verification(nn.Module):
    def __init__(self, in_dim=32*32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 128,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 64,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 2, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.net(x)
        return x
