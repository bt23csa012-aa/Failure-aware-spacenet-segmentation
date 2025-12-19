import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.dec(self.enc(x)))
