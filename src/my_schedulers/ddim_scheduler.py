import torch
import random
import math
from torchsparse import SparseTensor

from .scheduler import Scheduler
from my_models.modules.sparse_utils import batch_sparse_quantize_torch

class DDIMScheduler(Scheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, init_steps=None, steps=1024, mode='linear', step_size=1):
        super().__init__(init_steps=init_steps, steps=steps, beta_min=beta_min, beta_max=beta_max, mode=mode)

        self.alpha = torch.cumprod(1 - self.beta, dim=0).sqrt()

        while steps != len(self.alpha):
            if steps > len(self.alpha):
                raise ValueError("Can't reach the desired number of steps by halving the starting steps")

            self.alpha = self.alpha[::2]

        self.sigma = (1 - self.alpha ** 2).sqrt()

        self.step_size = step_size
        self.t_steps = self.t_steps[::step_size]

    def update(self, x, t, noise, shape, stochastic=False):
        bs = shape[0]

        x = x.reshape(shape)
        noise = noise.reshape(shape)
        device = x.device
        
        a_t, sigma_t = self.get_params(t, bs, device)

        if self.step_size is not None:
            a_ti, sigma_ti = self.get_params(t - self.step_size, bs, device)
        
        new_x = a_ti / a_t * (x - sigma_t * noise) + sigma_ti * noise
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

        x_t = self.alpha[t] * x0 + self.sigma[t] * noise

        return x_t, t, noise
    
    def create_noise(self, shape, device):
        return torch.randn(shape).clamp(-5, 5).to(device)
    
    def snr_weight(self, t):
        return torch.ones_like(t)

    def __call__(self, x, t=None):
        return self.add_noise(x, t)

    def get_params(self, t, bs, device):
        t = t.clamp(0, self.steps - 1)
        t = t.cpu()
        
        a_t = self.alpha[t].reshape(bs, 1, 1).to(device)
        sigma_t = self.sigma[t].reshape(bs, 1, 1).to(device)

        return a_t, sigma_t
    

class DDIMSparseScheduler(DDIMScheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steps=1024, mode='linear', init_steps=1024, pres=1e-5, step_size=1):
        super().__init__(init_steps=init_steps, steps=steps, beta_min=beta_min, beta_max=beta_max, mode=mode, step_size=step_size)
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