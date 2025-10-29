import torch
import random
import math
from torchsparse import SparseTensor

from .scheduler import Scheduler

class DDPMScheduler(Scheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, init_steps=None, steps=1024, mode='linear'):
        super().__init__(init_steps=init_steps, steps=steps, beta_min=beta_min, beta_max=beta_max, mode=mode)

        self.alpha = 1 - self.beta
        self.ahat = torch.cumprod(self.alpha, dim=0)
        
        prev_t = self.t_steps[-1]
        for t in reversed(self.t_steps[:-1]):
            self.alpha[prev_t:t] = self.ahat[t] / self.ahat[prev_t]
            prev_t = t
        
        self.beta = 1 - self.alpha
        self.sigma = self.beta.sqrt()
        # prev_alpha = torch.cat((torch.tensor([1]), self.alpha[:-1]))
        # self.sigma = ((1 - prev_alpha) / (1 - self.alpha) * self.beta).sqrt()

    def update(self, x, t, noise, shape, stochastic=True, save=False):
        bs = shape[0]

        x = x.reshape(shape)
        noise = noise.reshape(shape)
        device = x.device
        
        epsilon = torch.randn(x.shape).to(device) if stochastic else torch.zeros_like(x)
        for i, t_i in enumerate(t):
            if t_i == 0:
                epsilon[i] = torch.zeros_like(epsilon[i])
        epsilon = epsilon.to(device)

        t = t.cpu()
        sigma_t = self.sigma[t].reshape(bs, 1, 1).to(device)
        a_t = self.alpha[t].reshape(bs, 1, 1).to(device)

        ahat_t = self.ahat[t].reshape(bs, 1, 1).to(device)
        
        x_0 = self.denoise(x, noise, a_t, ahat_t)
        new_x = 1 / a_t.sqrt() * x_0 + sigma_t * epsilon

        # new_x = new_x - new_x.mean(dim=1, keepdim=True)

        return new_x if not save else (new_x, x_0)

    def denoise(self, x, noise, a_t, ahat_t):
        x_0 = x - (1 - a_t) / (1 - ahat_t).sqrt() * noise
        return x_0
    
    def add_noise(self, x0, t=None):
        if t is None:
            t = random.randint(0, self.steps - 1)

        noise = torch.randn(x0.shape, requires_grad=False)

        a_t_bar = self.ahat[t]

        return x0 * a_t_bar.sqrt() + noise * (1 - a_t_bar).sqrt(), t, noise

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
        ahat_t = self.ahat[t].reshape(bs, 1, 1).to(device)

        a_t1 = torch.ones_like(a_t).to(device)
        a_t1[t != 0] = self.alpha[t[t != 0] - 1].reshape(t[t != 0].shape[0], 1, 1).to(device)

        return a_t, ahat_t,# a_t1

class DDPMSparseScheduler(DDPMScheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steps=1024, mode='linear', init_steps=1024, pres=1e-5):
        super().__init__(init_steps=init_steps, steps=steps, beta_min=beta_min, beta_max=beta_max, mode=mode)
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
        x = torch.randn(shape).clamp(-5, 5).to(device) # Clamping to avoid outliers
        return self.torch2sparse(x)

    def update(self, x, t, noise, shape, stochastic=True, save=False):
        # Takse as input a SparseTensor for x and returns a SparseTensor
        # Noise and t are regular tensors
        x = x.F
        x = super().update(x, t, noise, shape, stochastic=stochastic, save=save)
        return self.torch2sparse(x)

    def post_process(self, x):
        # Convert the sparse tensor to a dense tensor
        return x.F