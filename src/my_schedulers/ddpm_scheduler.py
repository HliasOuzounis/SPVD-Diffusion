import torch
import random
import math
from torchsparse import SparseTensor

from .scheduler import Scheduler
from my_models.modules.sparse_utils import batch_sparse_quantize_torch

class DDPMScheduler(Scheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steps=1024, mode='linear', sigma='bt'):
        super().__init__()

        self.steps = steps
        self.beta = torch.linspace(beta_min, beta_max, steps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sigma = self.beta.sqrt() # Linear schedule

    def update(self, x, t, noise, shape, stochastic=True):
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

        ahat_t = self.alpha_cumprod[t].reshape(bs, 1, 1).to(device)
        
        a_t1_bar = self.alpha_cumprod[t - 1].reshape(bs, 1, 1).to(device)
        for i, t_i in enumerate(t):
            if t_i == 0:
                a_t1_bar[i] = torch.ones_like(a_t1_bar[i])

        # DIfferent denoising strategy
        # x_0 = self.denoise(x, noise, ahat_t)
        # x_0_coeff = a_t1_bar.sqrt() * (1 - a_t) / (1 - ahat_t)
        # x_t_coeff = a_t.sqrt() * (1 - a_t1_bar) / (1 - ahat_t) 
        # new_x = x * x_t_coeff + x_0 * x_0_coeff + sigma_t * z
        
        x_0 = self.denoise(x, noise, a_t, ahat_t)
        new_x = 1 / a_t.sqrt() * x_0 + sigma_t * epsilon

        # print(torch.max(new_x))
        new_x = new_x - new_x.mean(dim=1, keepdim=True)
        # print(new_x.mean())

        return new_x

    def denoise(self, x, noise, a_t, ahat_t):
        x_0 = x - (1 - a_t) / (1 - ahat_t).sqrt() * noise
        # x_0 = (x - noise * (1 - ahat_t).sqrt()) / ahat_t.sqrt()
        return x_0
    
    def add_noise(self, x, t=None):
        if t is None:
            t = random.randint(0, self.steps - 1)

        noise = torch.randn(x.shape, requires_grad=False)

        a_t_bar = self.alpha_cumprod[t]

        return x * a_t_bar.sqrt() + noise * (1 - a_t_bar).sqrt(), t, noise

    def create_noise(self, shape, device):
        return torch.randn(shape).clamp(-5, 5).to(device)

    def snr_weight(self, t):
        return torch.ones_like(t)
        t = t.cpu()
        snr = self.alpha_cumprod[t] / (1 - self.alpha_cumprod[t])
        # src: https://arxiv.org/pdf/2303.09556
        gamma = 5
        return torch.clamp(gamma / snr, max=1)

    def __call__(self, x, t=None):
        return self.add_noise(x, t)


class DDPMSparseScheduler(DDPMScheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steps=1024, mode='linear', sigma='bt', pres=1e-5):
        super().__init__(beta_min, beta_max, steps, mode, sigma)
        self.pres = pres

    def torch2sparse(self, x, shape):
        x = x.reshape(shape)
        
        coords = x[..., :3] # In case points have additional features
        coords = coords - coords.min(dim=1, keepdim=True).values
        coords, indices = batch_sparse_quantize_torch(coords, voxel_size=self.pres, return_index=True, return_batch_index=False)
        feats = x.view(-1, 3)[indices]
        
        return SparseTensor(coords=coords, feats=feats).to(coords.device)

    def create_noise(self, shape, device):
        x = torch.randn(shape).clamp(-5, 5) # Clamping to avoid outliers
        return self.torch2sparse(x, shape).to(device)

    def update(self, x, t, noise, shape, stochastic=True):
        # Takse as input a SparseTensor for x and returns a SparseTensor
        # Noise and t are regular tensors
        x = x.F
        x = super().update(x, t, noise, shape, stochastic)
        return self.torch2sparse(x, shape)

    def post_process(self, x):
        # Convert the sparse tensor to a dense tensor
        return x.F