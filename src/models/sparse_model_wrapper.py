import torch
import torch.nn as nn

from torchsparse import SparseTensor
from .sparse_utils import batch_sparse_quantize_torch



class SparseModelWrapper(nn.Module):
    """
    A wrapper class for sparse models that handles the quantization and inverse quantization
    of point clouds. This class takes a sparse model as input and provides a forward method
    to process the input data passed a BxNxF tensor through the model.

    Attributes:
        sparse_model (nn.Module): The sparse model to be wrapped.

    Methods:
        forward(x):
            Performs the forward pass through the sparse model with quantization and inverse quantization.
        
        _quantize(x):
            Quantizes the input point cloud.
        
        _inverse_quantize(x, inverse_indices, shape):
            Inverse quantizes the output point cloud.
    """
    def __init__(self, sparse_model):
        super(SparseModelWrapper, self).__init__()
        self.sparse_model = sparse_model   

    def forward(self, x):
        
        # Get the data out of the input
        x_t, t, cond_emb = x if len(x) == 3 else (x, t, None)

        # Get the point cloud shape
        shape = x_t.shape

        # Quantize the point cloud
        x_t, inverse_indices = self._quantize(x_t)

        # Forward pass on the diffusion model
        out = self.diff_unet((x_t, t)) if cond_emb is None else self.diff_unet((x_t, t, cond_emb))

        # Inverse quantize the point cloud
        out = self._inverse_quantize(out, inverse_indices, shape)

        return out
    
    def _quantize(self, x):
        B, N, F = x.shape

        coords = x - x.min(dim=1, keepdim=True).values
        coords, indices = batch_sparse_quantize_torch(coords, 
                                                      voxel_size=self.sparse_model.pres, 
                                                      return_index=True, 
                                                      return_batch_index=False)
        inverse_indices = torch.empty_like(indices)
        inverse_indices[indices] = torch.arange(indices.shape[0], device=indices.device)

        feats = x.view(-1, F)[indices]

        return SparseTensor(coords=coords.int(), feats=feats), inverse_indices
    
    def _inverse_quantize(self, x, inverse_indices, shape):
        return x[inverse_indices].reshape(shape)