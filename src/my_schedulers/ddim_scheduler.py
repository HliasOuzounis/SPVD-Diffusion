import torch
import random
import math
from torchsparse import SparseTensor

from .scheduler import Scheduler
from my_models.modules.sparse_utils import batch_sparse_quantize_torch

class DDIMScheduler(Scheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steps=1024, mode='linear', sigma='bt'):
        super().__init__()

        self.steps = steps
        self.beta = torch.linspace(beta_min, beta_max, steps, requires_grad=False)
        self._alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self._alpha, dim=0)
        self.sigma = self.beta.sqrt() # Linear schedule
        
    def update(self, x, t, noise, shape, stochastic=False):
        bs = shape[0]

        x = x.reshape(shape)
        noise = noise.reshape(shape)
        device = x.device
        
        t = t.cpu()
        
        a_t, sigma_t, a_t1, sigma_t1 = self.get_params(t, bs, device)
        
        new_x = a_t1 / a_t * (x - sigma_t * noise) + sigma_t1 * noise
        
        # new_x = new_x - new_x.mean(dim=1, keepdim=True)
        return new_x
        
    def denoise(self, x, noise, a_t):
        sigma_t = (1 - ahat_t ** 2).sqrt()
        x0 = (x - sigma_t * noise) / a_t
        return x0
    
    def add_noise(self, x0, t=None):
        if t is None:
            t = random.randint(0, self.steps - 1)

        noise = torch.randn(x0.shape, requires_grad=False)

        a_t_bar = self.alpha_cumprod[t]
        x_t = a_t_bar.sqrt() * x0 + (1 - a_t_bar).sqrt() * noise

        return x_t
    
    def create_noise(self, shape, device):
        return torch.randn(shape).clamp(-5, 5).to(device)
    
    def snr_weight(self, t):
        return torch.ones_like(t)

    def __call__(self, x, t=None):
        return self.add_noise(x, t)

    def get_params(self, t, bs, device):
        t = t.clamp(0, self.steps - 1)
        t = t.cpu()
        
        a_t = self.alpha_cumprod[t].sqrt().reshape(bs, 1, 1).to(device)
        sigma_t = (1 - a_t ** 2).sqrt()

        a_t1 = torch.ones_like(a_t).to(device)
        a_t1[t != 0] = self.alpha_cumprod[t[t != 0] - 1].sqrt().reshape(-1, 1, 1).to(device)
        sigma_t1 = (1 - a_t1 ** 2).sqrt()

        return a_t, sigma_t, a_t1, sigma_t1
    

class DDIMSparseScheduler(DDIMScheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steps=1024, mode='linear', sigma='bt', pres=1e-5):
        super().__init__(beta_min, beta_max, steps, mode, sigma)
        self.pres = pres

    def torch2sparse(self, coords, feats=None):
        # coords BxNx3
        # feats BxNxC
        if feats is None:
            feats = coords.clone()
            
        B, N, C = feats.shape
        device = coords.device
        
        coords -= coords.min(dim=1, keepdim=True).values
        batch_idx = torch.arange(0, B).repeat_interleave(N).unsqueeze(-1).to(coords.device)
        
        coords = (coords / self.pres).reshape(B * N, 3).int()
        coords = torch.cat([batch_idx, coords], dim=-1)
        _, idxs = torch.unique(coords, return_inverse=True, dim=0)

        idxs = idxs.sort().values

        coords = coords[idxs]
        feats = feats.reshape(-1, C)[idxs]

        return SparseTensor(coords=coords, feats=feats).to(device)

    def create_noise(self, shape, device):
        x = super().create_noise(shape, device)
        return self.torch2sparse(x)

    def update(self, x, t, noise, shape, stochastic=True):
        # Takse as input a SparseTensor for x and returns a SparseTensor
        # Noise and t are regular tensors
        x = x.F
        x = super().update(x, t, noise, shape, stochastic=stochastic)
        return self.torch2sparse(x)

    def post_process(self, x):
        # Convert the sparse tensor to a dense tensor
        return x.F