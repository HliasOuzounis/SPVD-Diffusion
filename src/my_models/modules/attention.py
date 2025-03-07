from .sparse_utils import sparse_to_dense

import torch.nn as nn

class Attention(nn.Module):
    """
    Attention Module that works with masked input.
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        import torch
        assert not torch.isnan(mask).any(), "NaN values in the mask tensor Attention"
        assert not torch.isnan(x).any(), "NaN values in the input tensor Attention"
        
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, C).
                - B: Batch size.
                - N: Number of points.
                - C: Number of input features.
            mask (Tensor): Mask tensor of shape (B, N).
                - B: Batch size.
                - N: Number of points.
        Returns:
            Tensor: Output tensor of shape (B, N, C).
        
        q: (B, N, C)
        k: (B, N, C)
        v: (B, N, C)
        """
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        assert not torch.isnan(q).any(), "NaN values in the q tensor Attention"
        assert not torch.isnan(k).any(), "NaN values in the k tensor Attention"
        assert not torch.isnan(v).any(), "NaN values in the v tensor Attention"

        x = self.attn(q, k, v)[0]#, key_padding_mask=mask)
        
        assert not torch.isnan(x).any(), "NaN values in the output tensor Attention"

        x = self.proj(x)
        x = self.proj_drop(x)
        
        assert not torch.isnan(x).any(), "NaN values in the final output tensor Attention"
        
        return x

class SparseAttention(Attention):
    def __init__(self, dim, num_heads, dropout=0):
        super().__init__(dim, num_heads, dropout)

        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
        x_dense = self.norm(x_dense)
        x_dense = x_dense + super().forward(x_dense, mask)

        x.F = x_dense[mask]
        
        import torch
        assert not torch.isnan(x.F).any(), "NaN values in the output tensor SparseAttention"

        return x

    
class CrossAttention:
    ...
    
class SparseCrossAttention:
    ...