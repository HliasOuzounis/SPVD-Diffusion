import torch
import random
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
        
        z = torch.rand_like(x) if stochastic else torch.zeros_like(x)
        for i, t_i in enumerate(t):
            if t_i == 0:
                z[i] = torch.zeros_like(z[i])
        z = z.to(device)

        t = t.cpu()
        sigma_t = self.sigma[t].reshape(bs, 1, 1).to(device)
        a_t = self.alpha[t].reshape(bs, 1, 1).to(device)

        a_t_bar = self.alpha_cumprod[t].reshape(bs, 1, 1).to(device)
        a_t1_bar = self.alpha_cumprod[t - 1].reshape(bs, 1, 1).to(device)
        for i, t_i in enumerate(t):
            if t_i == 0:
                a_t1_bar[i] = torch.ones_like(a_t1_bar[i]).to(device)

        # x_0 = self.denoise(x, noise, a_t_bar)
        # x_0_coeff = a_t1_bar.sqrt() * (1 - a_t) / (1 - a_t_bar)
        # x_t_coeff = a_t.sqrt() * (1 - a_t1_bar) / (1 - a_t_bar) 
        # new_x = x * x_t_coeff + x_0 * x_0_coeff + sigma_t * z

        new_x = 1 / a_t.sqrt() * (x - noise * (1 - a_t) / (1 - a_t_bar).sqrt()) + sigma_t * z

        return new_x.to(device)

    def denoise(self, x, noise, a_t_bar):
        x_0 = (x - noise * (1 - a_t_bar).sqrt()) / a_t_bar.sqrt()
        return x_0.clamp(-1, 1).to(x.device)
    
    def add_noise(self, x, t=None):
        if t is None:
            t = random.randint(0, self.steps - 1)

        noise = torch.randn(x.shape, requires_grad=False)

        a_t_bar = self.alpha_cumprod[t]

        return x * a_t_bar.sqrt() + noise * (1 - a_t_bar).sqrt(), t, noise
    
    def create_noise(self, shape, device):
        return torch.randn(shape).clamp(-3, 3).to(device)
    
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
        x = torch.randn(shape).clamp(-3, 3) # Clamping to avoid outliers
        return self.torch2sparse(x, shape).to(device)

    def update(self, x, t, noise, shape, stochastic=True):
        # Takse as input a SparseTensor for x and returns a SparseTensor
        # Noise and t are regular tensors
        x = super().update(x.F, t, noise, shape, stochastic)
        return self.torch2sparse(x, shape)