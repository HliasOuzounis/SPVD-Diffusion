import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from .embeddings import TimeEmbeddingBlock
from .attention import SparseAttention, SparseCrossAttention

from typing import Iterable


class SparseConv3DBlock(nn.Module):
    def __init__(self, features_in: int, features_out: int, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            spnn.BatchNorm(features_in),
            spnn.SiLU(),
            spnn.Conv3d(features_in, features_out, kernel_size, bias=True),
        )

    def forward(self, x):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_in`).
        Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_out`).
        """
        return self.layers(x)


class SparseResidualBlock(nn.Module):
    """
    A nn block that applies a 3D convolution to the input features and adds a residual connection.
    Also takes into account the time embedding information.
    Optionally, has an attention layer. (Fig. 4)
    """

    def __init__(
        self,
        features_in: int,
        features_out: int,
        t_emb_features: int,
        kernel_size: int = 3,
        attn_heads: int | None = None,
        cross_attn_heads: int | None = None,
        cross_attn_cond_dim: int | None = None,
    ):
        super().__init__()
        self.f_in = features_in
        self.conv1 = SparseConv3DBlock(features_in, features_out, kernel_size)
        self.conv2 = SparseConv3DBlock(features_out, features_out, kernel_size)

        self.t_embedding = TimeEmbeddingBlock(t_emb_features, features_out)

        self.res_connection = (
            nn.Identity()
            if features_in == features_out
            else nn.Linear(features_in, features_out)
        )

        self.has_attn = attn_heads is not None
        if self.has_attn:
            self.attn = SparseAttention(features_out, attn_heads)

        self.has_cross_attn = cross_attn_heads is not None
        if self.has_cross_attn:
            self.cross_attn = SparseCrossAttention(
                features_out, cross_attn_cond_dim, cross_attn_heads
            )

    def forward(self, x_in, t, reference=None):
        """
        Args:
            x_in (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_in`).
            t (Tensor): Time embedding tensor of shape (B, T).
                - B: Batch size.
                - T: Time embedding features (`t_emb_features`).
            reference (Tensor, optional): Image features tensor of shape (B, T, F).
                - B: Batch size.
                - T: Number of tokens (e.g., 179).
                - F: Number of features (e.g., 784).
                Defaults to None.

        Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_out`).
        """
        x = self.conv1(x_in)
        x.F = self.t_embedding(x.F, t, x.C[:, 0])
        x = self.conv2(x)
        x.F = x.F + self.res_connection(x_in.F)

        if self.has_attn:
            x = self.attn(x)

        if reference is not None and self.has_cross_attn:
            x = self.cross_attn(x, reference)

        return x


class DownBlock(nn.Module):
    """
    A component of the downsample path of the U-Net architecture.
    It applies a series of SparseResidualBlocks.
    """

    def __init__(
        self,
        features_in: int,
        features_out: int,
        t_emb_features: int,
        add_down: bool = True,
        num_layers: int = 1,
        attn_heads: int | None = None,
        cross_attn_heads: int | None = None,
        cross_attn_cond_dim: int | None = None,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            SparseResidualBlock(
                features_in=features_in,
                features_out=features_out,
                t_emb_features=t_emb_features,
                attn_heads=attn_heads,
                cross_attn_heads=cross_attn_heads,
                cross_attn_cond_dim=cross_attn_cond_dim,
            )
            for i in range(num_layers)
        )
        self.down = (
            spnn.Conv3d(features_out, features_out, 2, stride=2)
            if add_down
            else nn.Identity()
        )

    def forward(self, x, t, reference=None):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_in`).
            t (Tensor): Time embedding tensor of shape (B, T).
                - B: Batch size.
                - T: Time embedding features (`t_emb_features`).
            reference (Tensor, optional): Image features tensor of shape (B, T, F).
                - B: Batch size.
                - T: Number of tokens (e.g., 179).
                - F: Number of features (e.g., 784).
                Defaults to None.
        Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_out`).
        """
        skip_connections = [x]
        for res_block in self.res_blocks:
            x = res_block(x, t, reference)
            skip_connections.append(x)
        
        x = self.down(x)
        
        return x, skip_connections


class UpBlock(nn.Module):
    def __init__(
        self,
        features_in: int,
        features_out: int,
        skip_connection_features: list[int],
        t_emb_features: int,
        add_up: bool = True,
        num_layers: int = 1,
        attn_heads: int | None = None,
        cross_attn_heads: int | None = None,
        cross_attn_cond_dim: int | None = None,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            SparseResidualBlock(
                features_in=(features_in if i == 0 else features_out) + skip_connection_features.pop(),
                features_out=features_out,
                t_emb_features=t_emb_features,
                attn_heads=attn_heads,
                cross_attn_heads=cross_attn_heads,
                cross_attn_cond_dim=cross_attn_cond_dim,
            )
            for i in range(num_layers)
        )
        self.up_sample = (
            spnn.Conv3d(features_out, features_out, 2, stride=2, transposed=True)
            if add_up
            else nn.Identity()
        )
    
    def forward(self, x, t, skip_connections, reference=None):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F). Concatenated with the skip connection from the downsample path.
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_in + prev_downsample_features`).
            t (Tensor): Time embedding tensor of shape (B, T).
                - B: Batch size.
                - T: Time embedding features (`t_emb_features`).
            reference (Tensor, optional): Image features tensor of shape (B, T, F).
                - B: Batch size.
                - T: Number of tokens (e.g., 179).
                - F: Number of features (e.g., 784).
                Defaults to None.
        Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_out`).
        """
        for res_block in self.res_blocks:
            x = torchsparse.cat([x, skip_connections.pop()])
            x = res_block(x, t, reference)
        
        x = self.up_sample(x)

        return x
