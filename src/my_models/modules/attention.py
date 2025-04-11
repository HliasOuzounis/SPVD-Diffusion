import torch
import torch.nn as nn

from .sparse_utils import sparse_to_dense


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads=8, attn_drop=0.):
        super().__init__()        
        self.attn = nn.MultiheadAttention(
            embed_dim=q_dim,
            num_heads=num_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(q_dim)
    
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
        super().__init__(dim, dim, num_heads, dropout)

        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
        x_dense = self.norm(x_dense)
        
        x_dense = super().forward(x_dense, x_dense, mask)

        x.F = x_dense[mask]
        return x
    
    
class SparseCrossAttention(Attention):
    def __init__(self, dim, image_dim, num_heads=8, dropout=0):
        super().__init__(dim, image_dim, num_heads, dropout)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, y):
        x_dense, x_mask = sparse_to_dense(x.F, x.C[:, 0].long())
        x_dense = self.norm(x_dense)

        x_dense = super().forward(x_dense, y)

        x.F = x_dense[x_mask]

        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import DropPath
# from torch_geometric.utils import to_dense_batch

# def masked_softmax(x, mask):
#     # calculate softmax only for the masked elements
#     # x.shape : B x H x N x N 
#     # mask.shape : B x N
    
#     mask = mask.unsqueeze(1) # broadcast across H dim
#     mask = mask.unsqueeze(2) # broadcast across N dim
#     # mask.shape B x 1 x 1 x N

#     x_exp = x.exp() * mask
#     x = x_exp / x_exp.sum(-1, keepdims=True)

#     return x

# __all__ = ['SparseAttention', 'SparseTransformer', 'SparseCrossAttention']
           
# # ----- Backbone Modules ----- #

# def sparse_to_dense(x, b):
#     "Receives a sparse representation and the batch idx for each element and returns a dense sequence representation and a mask indicating the actual values"
#     feats_dense, mask = to_dense_batch(x, batch=b)
#     return feats_dense, mask

# class Mlp(nn.Module):
#     """
#         MLP module for the transformer feedforward block.
#     """

#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features

#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
    
# class Attention(nn.Module):
#     """
#         Attention module that can handle a masked input.
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
#         self.attn_drop=nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, mask):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k , v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         #attn = attn.softmax(dim=-1)
#         attn = masked_softmax(attn, mask)
#         attn = self.attn_drop(attn)
        
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class CrossAttention(nn.Module):
#     """
#         Cross Attention module that can handle a masked input.
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
#         self.attn_drop=nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x, y):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, 1, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)[0]

#         _, Ny, Cy = y.shape
#         kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, Cy//self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         # NOTE: Y is a dense representation, so we don't need to mask it
#         # Also, if we needed to mask it, we would nedd a y_mask
#         attn = F.softmax(attn, dim=-1) # masked_softmax(attn, mask)
#         attn = self.attn_drop(attn)
        
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# # ----- Usable Blocks ----- #

# class SparseAttention(nn.Module):
#     """
#         An attention module that works with sparse tensors.
#         This modules handles the preprocessing required for the sparse tensor 
#         and the `Attention` module is where the actual computation is done.
#     """
#     def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm):
#         super().__init__()
        
#         # ATTENTION BLOCK
#         self.norm1 = norm_layer(dim) # TODO: This normalization should move inside the `Attention` block
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)


#     def forward(self, x):
            
#         x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
            
#         x_dense = x_dense + self.attn(self.norm1(x_dense), mask)
#         x.F = x_dense[mask]

#         return x
    
# class SparseCrossAttention(nn.Module):

#     def __init__(self, dim, cond_dim, num_heads, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm):
#         super().__init__()
        
#         # ATTENTION BLOCK
#         self.norm1 = norm_layer(dim) # TODO: This normalization should move inside the `CrossAttention` block
#         self.cross_attn = CrossAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        
#         self.cond_proj = nn.Linear(cond_dim, dim)
#         self.cond_norm = norm_layer(dim)

#     def forward(self, x, cond_emb):
#         """
#             x: Sparse tensor
#             y: Dense tensor
#         """
#         x_dense, mask = sparse_to_dense(x.F, x.C[:, 0].long())
        
#         # Project and normalize the conditional embedding
#         cond_emb = self.cond_proj(cond_emb)  # (B, dim)
#         cond_emb = self.cond_norm(cond_emb)  # Normalize to stabilize values
#         if len(cond_emb.shape) == 2:         # if the conditional embedding is a single vector
#             cond_emb = cond_emb.unsqueeze(1) # (B, 1, dim)

#         x_dense = x_dense + self.cross_attn(self.norm1(x_dense), cond_emb)
#         x.F = x_dense[mask]

#         return x
