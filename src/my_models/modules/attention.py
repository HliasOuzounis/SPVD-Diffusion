from .sparse_utils import sparse_to_dense

import torch.nn as nn

def masked_softmax(x, mask):
    # calculate softmax only for the masked elements
    # x.shape : B x H x N x N 
    # mask.shape : B x N
    
    mask = mask.unsqueeze(1) # broadcast across H dim
    mask = mask.unsqueeze(2) # broadcast across N dim
    # mask.shape B x 1 x 1 x N

    x_exp = x.exp() * mask
    x = x_exp / x_exp.sum(-1, keepdims=True)

    return x

# class Attention(nn.Module):
#     """
#     Attention Module that works with masked input.
#     """
#     def __init__(self, dim, num_heads, dropout=0.0):
#         super().__init__()
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
    
#     def forward(self, x, mask):
#         import torch
#         assert not torch.isnan(mask).any(), "NaN values in the mask tensor Attention"
#         assert not torch.isnan(x).any(), "NaN values in the input tensor Attention"
        
#         """
#         Args:
#             x (Tensor): Input tensor of shape (B, N, C).
#                 - B: Batch size.
#                 - N: Number of points.
#                 - C: Number of input features.
#             mask (Tensor): Mask tensor of shape (B, N).
#                 - B: Batch size.
#                 - N: Number of points.
#         Returns:
#             Tensor: Output tensor of shape (B, N, C).
        
#         q: (B, N, C)
#         k: (B, N, C)
#         v: (B, N, C)
#         """
#         B, N, C = x.shape
#         q, k, v = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         assert not torch.isnan(q).any(), "NaN values in the q tensor Attention"
#         assert not torch.isnan(k).any(), "NaN values in the k tensor Attention"
#         assert not torch.isnan(v).any(), "NaN values in the v tensor Attention"

#         x = self.attn(q, k, v)[0]#, key_padding_mask=mask)
        
#         assert not torch.isnan(x).any(), "NaN values in the output tensor Attention"

#         x = self.proj(x)
#         x = self.proj_drop(x)
        
#         assert not torch.isnan(x).any(), "NaN values in the final output tensor Attention"
        
#         return x

class Attention(nn.Module):
    """
        Attention module that can handle a masked input.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., qkv_bias=False, qk_scale=None, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        #attn = attn.softmax(dim=-1)
        attn = masked_softmax(attn, mask)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SparseAttention(nn.Module):
    """
        An attention module that works with sparse tensors.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        #self.emb_proj = nn.Linear(dim, 2*dim)
    def forward(self, x):
        x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
        x_dense = x_dense + self.attn(self.norm1(x_dense), mask)
        x.F = x_dense[mask]

        return x

# class SparseAttention(Attention):
#     def __init__(self, dim, num_heads, dropout=0):
#         super().__init__(dim, num_heads, dropout)

#         self.norm = nn.LayerNorm(dim)
    
#     def forward(self, x):
#         x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
#         x_dense = self.norm(x_dense)
#         x_dense = x_dense + super().forward(x_dense, mask)

#         x.F = x_dense[mask]
        
#         import torch
#         assert not torch.isnan(x.F).any(), "NaN values in the output tensor SparseAttention"

#         return x

    
class CrossAttention:
    ...
    
class SparseCrossAttention:
    ...