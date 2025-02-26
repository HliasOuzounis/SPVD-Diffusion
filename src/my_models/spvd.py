import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn

from modules.convolution import DownBlock, UpBlock
from modules.mlp import SharedMLP
from modules.sparse_utils import mixed_mix, PointTensor, initial_voxelize

from typing import Iterable


class StemStage(nn.Module):
    """
    The initial stage of the U-Net architecture. (Fig. 3)
    """
    def __init__(self, features_in: int, features_out: int, kernel_size: int = 3):
        super().__init__()
        self.voxel_conv = nn.Sequential(
            spnn.Conv3d(features_in, features_out, kernel_size),
            spnn.BatchNorm(features_out),
            spnn.SiLU(),
            spnn.Conv3d(features_out, features_out, kernel_size),
        )
        self.point_conv = SharedMLP(features_in, features_out)
    
    def forward(self, x, z):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_in`).
            z (SparseTensor): Point Representation tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features. (`features_in`)
        Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_out`).
            SparseTensor: Point Representation tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_out`).
        """

        x = self.voxel_conv(x)
        z = self.point_conv(z)

        x, z = mixed_mix(x, z)
        
        return x, z


class SPVDownStage(nn.Module):
    """
    The downsampling path of the U-Net architecture. (Fig. 3)
    Contains a series of DownBlocks and a residual connetion.
    Also saves the intermediate features for skip connections to the upsample path.
    """

    def __init__(
        self,
        features_list: Iterable[int],
        t_emb_features: int,
        num_layers_list: Iterable[int] | int = 1,
        attn_heads: Iterable[int] | int | None = None,
    ):
        if isinstance(num_layers_list, int):
            num_layers_list = [num_layers_list] * (len(features_list) - 1)
        if isinstance(attn_heads, int):
            attn_heads = [attn_heads] * (len(features_list) - 1)
        if isinstance(attn_heads, None):
            attn_heads = [None] * (len(features_list) - 1)

        assert (
            len(features_list) == len(num_layers_list) + 1 == len(attn_heads) + 1
        ), "Features, num_layers and attn_heads must have the same length"

        super().__init__()
        self.down_blocks = nn.ModuleList(
            DownBlock(
                features_in=features_list[i],
                features_out=features_list[i + 1],
                t_emb_features=t_emb_features,
                add_down=(i != len(features_list) - 2),
                num_layers=num_layers_list[i],
                attn_heads=attn_heads[i],
            )
            for i in range(len(features_list) - 1)
        )

        self.residual = SharedMLP(features_list[0], features_list[-1])

    def forward(self, x, z, t, image_features=None):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_list[0]`).
            z (SparseTensor): Point Representation tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features. (`features_list[0]`)
            t (Tensor): Time embedding tensor of shape (B, T).
                - B: Batch size.
                - T: Time embedding features (`t_emb_features`).
            image_features (Tensor, optional): Image features tensor of shape (B, T, F).
                - B: Batch size.
                - T: Number of tokens (e.g., 179).
                - F: Number of features (e.g., 784).
                Defaults to None.

         Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_list[-1]`).
            SparseTensor: Point Representation tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_list[-1]`).
            List[SparseTensor]: List of intermediate features for skip connections
        """
        residual = self.residual(z)

        saved = []
        for down_block in self.down_blocks:
            x = down_block(x, t, image_features)
            saved.append(x)

        x, z = mixed_mix(x, residual)

        return x, z, saved


class SPVUpStage(nn.Module):
    """
    The upsampling path of the U-Net architecture. (Fig. 3)
    Contains a series of UpBlocks and a residual connetion.
    """

    def __init__(
        self,
        features_list: Iterable[int],
        t_emb_features: int,
        num_layers_list: Iterable[int] | int = 1,
        attn_heads: Iterable[int] | int | None = None,
    ):
        if isinstance(num_layers_list, int):
            num_layers_list = [num_layers_list] * (len(features_list) - 1)
        if isinstance(attn_heads, int):
            attn_heads = [attn_heads] * (len(features_list) - 1)
        if isinstance(attn_heads, None):
            attn_heads = [None] * (len(features_list) - 1)

        assert (
            len(features_list) == len(num_layers_list) + 1 == len(attn_heads) + 1
        ), "Features, num_layers and attn_heads must have the same length"

        super().__init__()
        self.up_blocks = nn.ModuleList(
            UpBlock(
                features_in=features_list[i],
                features_out=features_list[i + 1],
                t_emb_features=t_emb_features,
                add_up=(i != len(features_list) - 2),
                num_layers=num_layers_list[i],
                attn_heads=attn_heads[i],
            )
            for i in range(len(features_list) - 1)
        )

        self.residual = SharedMLP(features_list[0], features_list[-1])

    def forward(self, x, z, t, skip_connections, image_features=None):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`features_list[0]`).
            z (SparseTensor): Point Representation tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features. (`features_list[0]`)
            t (Tensor): Time embedding tensor of shape (B, T).
                - B: Batch size.
                - T: Time embedding features (`t_emb_features`).
            skip_connections List[SparseTensor]: List of intermediate features for skip connections
            image_features (Tensor, optional): Image features tensor of shape (B, T, F).
                - B: Batch size.
                - T: Number of tokens (e.g., 179).
                - F: Number of features (e.g., 784).
                Defaults to None.

         Returns:
            SparseTensor: Output sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_list[-1]`).
            SparseTensor: Point Representation tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of output features (`features_list[-1]`).
        """

        residual = self.residual(z)

        for up_block in self.up_blocks:
            x = torchsparse.cat([x, skip_connections.pop()])
            x = up_block(x, t, image_features)

        x, z = mixed_mix(x, residual)

        return x, z


class SPVD(nn.Module):
    """
    The Sparse Point Voxel Diffuion (SPVD) model.
    """

    def __init__(
        self,
        down_blocks: Iterable[dict],
        up_blocks: Iterable[dict],
        t_emb_features: int,
        point_channels: int = 3,
        point_res: float = 1e-5,
        voxel_size: float = 0.05,
    ):
        super().__init__()

        assert sum(
            len(down_block["features_list"]) for down_block in down_blocks
        ) == sum(
            len(up_block["features_list"]) for up_block in up_blocks
        ), "Down and Up must have the same number of stages"

        self.point_res = point_res
        self.voxel_size = voxel_size

        self.stem_stage = StemStage(
            features_in=point_channels, features_out=down_blocks[0]["features_list"][0]
        )

        self.down_stages = nn.ModuleList(
            SPVDownStage(
                features_list=down_block["features_list"],
                t_emb_features=t_emb_features,
                num_layers_list=down_block["num_layers_list"],
                attn_heads=down_block["attn_heads"],
            )
            for down_block in down_blocks
        )

        self.up_stages = nn.ModuleList(
            SPVUpStage(
                features_list=up_block["features_list"],
                t_emb_features=t_emb_features,
                num_layers_list=up_block["num_layers_list"],
                attn_heads=up_block["attn_heads"],
            )
            for up_block in up_blocks
        )

        self.conv_out = SharedMLP(up_blocks[-1]["features_list"][0], point_channels)

    def forward(self, inp, image_features=None):
        """
        Args:
            x (SparseTensor): Input sparse tensor of shape (B, N, F).
                - B: Batch size.
                - N: Number of points.
                - F: Number of input features (`point_channels`).
            t (int): Time stamp.
            image_features (Tensor, optional): Image features tensor of shape (B, T, F).
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
        z = PointTensor(x.F, x.C.float())

        t = ...  # TimeStamp embedding
        time_embeddings = ...  # TimeStamp embedding

        # Initial Voxelization
        x0 = initial_voxelize(z, self.point_res, self.voxel_size)

        # Initial Convolution
        x, z = self.stem_stage(x0, z)

        skip_connections = []
        for down_stage in self.down_stages:
            x, z, saved = down_stage(x, z, t, image_features)
            skip_connections += saved

        return x, z
        for up_stage in self.up_stages:
            x, z = up_stage(x, z, t, skip_connections, image_features)

        return self.conv_out(z.F)
