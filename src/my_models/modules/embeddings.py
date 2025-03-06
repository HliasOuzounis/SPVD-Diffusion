import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class TimeEbeddingBlock(MessagePassing):
    """ 
    A block to incorporate the time embedding information to a sparse representation of node features. 
    Use to intergrade the time embedding information to sparse voxels.
    """
    
    def __init__(self, n_emb, embed_dim):
        super().__init__('add', flow='target_to_source')

        self.proj_mlp = nn.Linear(n_emb, embed_dim * 2)
        self.silu = nn.SiLU()

    def forward(self, x, t, b):
        # x: features of each node
        # t: time embedding for each point cloud
        # b: a tensor indicate the batch index of each node

        node_idx = torch.arange(0, len(x), device=x.device)
        edge_index = torch.stack([node_idx, b]).long()

        t = self.silu(t)
        t = self.proj_mlp(t)

        return  self.propagate(edge_index=edge_index, x=x, t=t)
    
    
    def message(self, x_i, t_j):
        scale, shift = torch.chunk(t_j, 2, dim=-1)
        out = (1 + scale) * x_i + shift

        return out
    
def timestep_embedding(tsteps, emb_dim, max_period= 10000):
    exponent = -torch.log(max_period) * torch.linspace(0, 1, emb_dim//2, device=tsteps.device)
    emb = tsteps[:,None].float() * exponent.exp()[None,:]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return F.pad(emb, (0,1,0,0)) if emb_dim%2==1 else emb