import torch
from abc import ABC, abstractmethod
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn
from models.sparse_utils import batch_sparse_quantize_torch

__all__ = ['Torch2Sparse', 'Torch2TorchsparseCPU', 'Torch2TorchsparseGPU']

class Torch2Sparse(ABC):
    """
    Abstract base class for converting a torch.Tensor representation to a torchsparse sparse tensor.
    """
    def __call__(self, pts: torch.Tensor, shape, pres):
        return self.torch2sparse(pts, shape, pres)

    @abstractmethod
    def torch2sparse(self, pts: torch.Tensor, shape, pres):
        pass

class Torch2TorchsparseCPU(Torch2Sparse):
    """
    CPU implementation for converting tensors to sparse tensors.
    The CPU implementation allows for high grid resolution, ie smaller voxel size, 
    but on the other hand, data have to be send back to CPU for computations.
    """

    def torch2sparse(self, pts: torch.Tensor, shape, pres):
        pts = pts.cpu().reshape(shape)
        coords = pts[:, :, :3]
        coords = coords - coords.min(dim=1, keepdim=True)[0]
        coords_np = coords.numpy()

        batch = []
        for b in range(shape[0]):
            c, indices = sparse_quantize(coords_np[b], pres, return_index=True)
            f = pts[b][indices]
            batch.append({"pc": SparseTensor(coords=torch.tensor(c), feats=f)})

        batch = sparse_collate_fn(batch)["pc"]
        return batch
    

class Torch2TorchsparseGPU(Torch2Sparse):
    """
    GPU implementation for converting tensors to sparse tensors.
    """

    def torch2sparse(self, pts: torch.Tensor, shape, pres):
        pts = pts.reshape(shape)
        coords = pts[:, :, :3]
        coords = coords - coords.min(dim=1, keepdim=True).values
        coords_quantized, indices = batch_sparse_quantize_torch(coords, voxel_size=pres, return_index=True, return_batch_index=False)

        feats = pts.view(-1, pts.shape[-1])[indices]
        return SparseTensor(coords=coords_quantized, feats=feats)