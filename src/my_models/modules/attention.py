import torch
import torch.nn as nn

from .sparse_utils import sparse_to_dense


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.):
        super().__init__()        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            kdim=dim,
            vdim=dim,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, query, key_value, mask=None):
        # query.shape : B x N x C
        # key_value.shape : B x N x C
        # mask.shape : B x N

        # Sparse Tensor mask has True on the real elements
        # MultiHeadAttention mask expects True on elements to ignore
        if mask is not None:
            mask = ~mask

        attn_output, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=mask,
            need_weights=False,
        )

        return self.norm(query + attn_output)
    

class SparseAttention(Attention):
    def __init__(self, dim, num_heads=8, dropout=0):
        super().__init__(dim, num_heads, dropout)

        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
        x_dense = self.norm(x_dense)
        
        x_dense = super().forward(x_dense, x_dense, mask)

        x.F = x_dense[mask]
        return x
    
    
class SparseCrossAttention(Attention):
    def __init__(self, dim, image_dim, num_heads=8, dropout=0):
        super().__init__(dim, num_heads, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(image_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, y):
        x_dense, x_mask = sparse_to_dense(x.F, x.C[:, 0].long())
        x_dense = self.norm(x_dense)

        y = self.mlp(y)
        
        x_dense = super().forward(x_dense, y)

        x.F = x_dense[x_mask]

        return x