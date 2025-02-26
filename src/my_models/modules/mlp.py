import torch.nn as nn

class SharedMLP(nn.Module):
    def __init__(self, features_in: int, features_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.ReLU(),
            nn.Linear(features_in, features_out),
        )
    
    def forward(self, z):
        z.F = self.layers(z.F)
        return z