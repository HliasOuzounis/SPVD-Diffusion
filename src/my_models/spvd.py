import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from .modules.convolution import DownBlock, UpBlock, SparseResidualBlock
from .modules.mlp import SharedMLP
from .modules.sparse_utils import PointTensor, mixed_mix, initial_voxelize, voxel_to_point
from .modules.embeddings import timestep_embedding

from typing import Iterable


class SPVUnet(nn.Module):
    """
    The Sparse Point Voxel Diffuion (SPVD) model.
    """

    def __init__(
        self,
        features: Iterable[int],
        attn_heads_list: Iterable[int],
        cross_attn_heads_list: Iterable[int],
        cross_attn_cond_dim: int,
        t_emb_features: int = None,
        num_layers: int = 1,
        point_channels: int = 3,
        point_res: float = 1e-5,
        voxel_size: float = 0.1,
    ):
        assert len(features) - 1 == len(attn_heads_list) == len(cross_attn_heads_list), "Mismatch in number of features and attention heads lists."
        super().__init__()

        self.pres = point_res
        self.voxel_size = voxel_size
        
        self.in_conv = spnn.Conv3d(point_channels, features[0], kernel_size=3, padding=1)

        self.t_emb_features = features[0] if t_emb_features is None else t_emb_features
        emb_features = features[0] * 4

        self.t_emb_mlp = nn.Sequential(
            nn.BatchNorm1d(self.t_emb_features),
            nn.SiLU(),
            nn.Linear(self.t_emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )
        
        self.down_blocks = nn.ModuleList()
        
        prev_features = features[0]
        skip_connection_features = []
        for i, (features_out, attn_heads, cross_attn_heads) in enumerate(
            zip(features[1:], attn_heads_list, cross_attn_heads_list), start=1
        ):
            self.down_blocks.append(
                DownBlock(
                    features_in=prev_features,
                    features_out=features_out,
                    t_emb_features=emb_features,
                    attn_heads=attn_heads,
                    cross_attn_heads=cross_attn_heads,
                    cross_attn_cond_dim=cross_attn_cond_dim,
                    add_down = i != len(attn_heads_list),
                    num_layers=num_layers,
                )
            )
            skip_connection_features += [prev_features] + [features_out] * num_layers

            prev_features = features_out

        self.mid_block = nn.ModuleList([
            SparseResidualBlock(
                features_in=features_out,
                features_out=features_out,
                t_emb_features=emb_features,
            ),
        ])
        
        self.up_blocks = nn.ModuleList()
        for i, (features_out, attn_heads, cross_attn_heads) in enumerate(
            zip(reversed(features[1:]), reversed(attn_heads_list), reversed(cross_attn_heads_list)), start=1
        ):
            self.up_blocks.append(
                UpBlock(
                    features_in=prev_features,
                    features_out=features_out,
                    skip_connection_features=skip_connection_features,
                    t_emb_features=emb_features,
                    attn_heads=attn_heads,
                    cross_attn_heads=cross_attn_heads,
                    cross_attn_cond_dim=cross_attn_cond_dim,
                    add_up = i != len(attn_heads_list),
                    num_layers=num_layers + 1,
                )
            )
            prev_features = features_out
        
        self.point_branch = SharedMLP(point_channels, features[0])
        
        self.out_conv = nn.Sequential(
            nn.BatchNorm1d(features[0]),
            nn.SiLU(),
            nn.Linear(features[0], point_channels, bias=False),
        )
        

    def forward(self, inp, reference=None):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`point_channels`).
            t (int): Time stamp.
            reference (Tensor, optional): Image features tensor of shape (B, T, F).
                - B: Batch size.
                - T: Number of tokens (e.g., 179).
                - F: Number of features (e.g., 784).
                Defaults to None.

         Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`point_channels`).
        """

        x, t = inp

        if (isinstance(reference, list) and any(image is None for image in reference)):
            reference = None
        
        z = PointTensor(x.F, x.C.float())

        t = timestep_embedding(t, self.t_emb_features)
        t_emb = self.t_emb_mlp(t)

        x = initial_voxelize(z, self.pres, self.voxel_size)

        x = self.in_conv(x)

        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb, reference)
            skip_connections.extend(skip)
            
        for mid_block in self.mid_block:
            x = mid_block(x, t_emb)
        
        for up_block in self.up_blocks:
            x = up_block(x, t_emb, skip_connections, reference)

        assert len(skip_connections) == 0, "Skip connections are not empty, something is wrong"

        z1 = voxel_to_point(x, z)
        
        z1.F += self.point_branch(z).F

        return self.out_conv(z1.F)

