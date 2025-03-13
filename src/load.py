import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = spnn.Conv3d(32, 32, 2, stride=2)

    def forward(self, x):
        return self.conv(x)

c = Conv()
x = torch.load("x_before_down.pt")

y = c(x)

mask = y.C[:, 0]

print(mask.min(), mask.max())