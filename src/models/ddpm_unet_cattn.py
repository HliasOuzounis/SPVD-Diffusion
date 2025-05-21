import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from functools import wraps
import fastcore.all as fc
from .sparse_utils import PointTensor, initial_voxelize, voxel_to_point


from .modules import TimeEmbeddingBlock, SparseAttention, SparseCrossAttention
from .utils import lin, timestep_embedding

def saved(m, blk):
    m_ = m.forward

    @wraps(m.forward)
    def _f(*args, **kwargs):
        res = m_(*args, **kwargs)
        blk.saved.append(res)
        return res

    m.forward = _f
    return m

def unet_conv(ni, nf, ks=2, bias=True):
    layers = nn.Sequential(
        spnn.BatchNorm(ni),
        spnn.SiLU(), 
        spnn.Conv3d(ni, nf, ks, bias=bias)
    )
    return layers

class PointBlock(nn.Module):

    def __init__(self, ni, nf):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(ni), 
            nn.SiLU(), 
            nn.Linear(ni, nf)
        )

    def forward(self, z):
        z.F = self.conv(z.F)
        return z

class EmbResBlock(nn.Module):
    def __init__(self, n_emb, ni, nf=None, ks=3, attn_chans=None, cross_attn_chans=None, cross_attn_cond_dim=None):
        super().__init__()
        nf = nf or ni

        self.conv1 = unet_conv(ni, nf, ks)
        self.conv2 = unet_conv(nf, nf, ks)

        self.t_emb = TimeEmbeddingBlock(n_emb, nf)
        
        self.idconv = fc.noop if ni==nf else nn.Linear(ni, nf) 

        self.attn = False
        if attn_chans: self.attn = SparseAttention(nf, attn_chans)

        self.cross_attn = False
        if cross_attn_chans: self.cross_attn = SparseCrossAttention(nf, cross_attn_cond_dim, cross_attn_chans)

    
    def forward(self, x_in, t, cond=None):

        # first conv
        x = self.conv1(x_in)

        # time embedding
        x.F = self.t_emb(x.F, t, x.C[:, 0])

        # second conv
        x = self.conv2(x)

        # residual connection
        x.F = x.F + self.idconv(x_in.F)

        if self.attn: 
            x = self.attn(x) # Residual connection inside the attention module
        
        if self.cross_attn:
            x = self.cross_attn(x, cond) # Residual connection inside the cross attention module

        return x
    
class DownBlock(nn.Module):
    def __init__(self, n_emb, ni, nf, add_down=True, num_layers=1, attn_chans=None, cross_attn_chans=None, cross_attn_cond_dim=None):
        super().__init__()
        self.resnets = nn.ModuleList([saved(EmbResBlock(n_emb, ni if i==0 else nf, nf, attn_chans=attn_chans, 
                                                        cross_attn_chans=cross_attn_chans, cross_attn_cond_dim=cross_attn_cond_dim), self)
                                      for i in range(num_layers)])
        
        self.down = saved(spnn.Conv3d(nf, nf, 2, stride=2),self) if add_down else nn.Identity()
            
    def forward(self, x, t, cond=None):
        self.saved = []
        for resnet in self.resnets: x = resnet(x, t, cond)
        x = self.down(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, n_emb, ni, prev_nf, nf, add_up=True, num_layers=2, attn_chans=None, cross_attn_chans=None, cross_attn_cond_dim=None):
        super().__init__()
        self.resnets = nn.ModuleList([EmbResBlock(n_emb, (prev_nf if i==0 else nf)+(ni if (i==num_layers-1) else nf), nf, attn_chans=attn_chans, 
                                                  cross_attn_chans=cross_attn_chans, cross_attn_cond_dim=cross_attn_cond_dim)
                                     for i in range(num_layers)])
        
        self.up = spnn.Conv3d(nf, nf, 2, stride=2, transposed=True) if add_up else nn.Identity()

    def forward(self, x, t, ups, cond=None):
        for resnet in self.resnets: x = resnet(torchsparse.cat([x, ups.pop()]), t, cond)
        return self.up(x)


class SPVUnet(nn.Module):

    def __init__(self, voxel_size, in_channels=3, out_channels=3, nfs=(224,448,672,896), pres=1e-5, num_layers=1, attn_chans=None, attn_start=1, 
                cross_attn_chans=None, cross_attn_start=1, cross_attn_seq_length=50, cross_attn_cond_dim=None, use_point_branch=True):
        super().__init__()

        self.pres = pres
        self.voxel_size = voxel_size
        
        # This is crucial in the sparse implementation to correct dublicate points etc 
        # before the introduction of skip connections
        self.conv_in = spnn.Conv3d(in_channels, nfs[0], kernel_size=3, padding=1)
        
        
        self.n_temb = nf = nfs[0]
        n_emb = nf*4
        self.emb_mlp = nn.Sequential(lin(self.n_temb, n_emb, norm=nn.BatchNorm1d),
                                     lin(n_emb, n_emb))
        
        n = len(nfs)
        self.downs = nn.ModuleList()
        for i in range(len(nfs)):
            ni = nf
            nf = nfs[i]
            self.downs.append(DownBlock(n_emb, ni, nf, add_down=i!=len(nfs)-1, num_layers=num_layers, 
                                        attn_chans=None if i<attn_start else attn_chans,
                                        cross_attn_chans=None if i<cross_attn_start else cross_attn_chans, 
                                        cross_attn_cond_dim=cross_attn_cond_dim))
            
        self.mid_block = nn.ModuleList([
            EmbResBlock(n_emb, nfs[-1]), 
            #Transformer(4, nfs[-1], 8)
        ])
            

        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(len(nfs)):
            prev_nf = nf
            nf = rev_nfs[i]
            ni = rev_nfs[min(i+1, len(nfs)-1)]
            self.ups.append(UpBlock(n_emb, ni, prev_nf, nf, add_up=i!=len(nfs)-1, num_layers=num_layers+1,
                                       attn_chans=None if i+1>len(nfs)-attn_start else attn_chans,
                                       cross_attn_chans=None if i+1>len(nfs)-cross_attn_start else cross_attn_chans,
                                       cross_attn_cond_dim=cross_attn_cond_dim))
             
        self.conv_out = nn.Sequential(
            nn.BatchNorm1d(nfs[0]),
            nn.SiLU(), 
            nn.Linear(nfs[0], out_channels, bias=False),
        )

        self.cross_attn_chans = cross_attn_chans
        if cross_attn_chans:
            self.uncond_embedding = nn.Parameter(torch.randn(cross_attn_seq_length, cross_attn_cond_dim))
        
        self.use_point_branch = use_point_branch
        if use_point_branch:
            self.point_branch = PointBlock(in_channels, nfs[0])

    def forward(self, inp):

        # Input Processing
        x, t, cond = inp if len(inp)==3 else (*inp, None)
        if cond is None and self.cross_attn_chans:
            bs = t.shape[0]
            cond = self.uncond_embedding[None].expand(bs, -1, -1)
        z = PointTensor(x.F, x.C.float())
        t = timestep_embedding(t, self.n_temb)
        emb = self.emb_mlp(t)
        
        # map the point tensor to the sparse tensor
        x0 = initial_voxelize(z, self.pres, self.voxel_size)
        # pass through an initial convolution to extract local features
        x = self.conv_in(x0)
        
        saved = [x]

        for block in self.downs:
            x = block(x, emb, cond)
        saved += [p for o in self.downs for p in o.saved]

        for mb in self.mid_block:
            x = mb(x, emb)
        
        for block in self.ups:
            x = block(x, emb, saved, cond)
        
        z1 = voxel_to_point(x, z)

        if self.use_point_branch:
            z1.F = z1.F + self.point_branch(z).F
            
        return self.conv_out(z1.F)