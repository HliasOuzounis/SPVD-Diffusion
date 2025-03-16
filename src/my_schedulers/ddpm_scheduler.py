import torch
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
        device = x.device
        
        z = torch.rand_like(x) if stochastic else torch.zeros_like(x)
        for i, t_i in enumerate(t):
            if t_i == 0:
                z[i] = torch.zeros_like(z[i])
        z = z.to(device)

        t = t.cpu()
        a_t = self.alpha[t].reshape(bs, 1, 1).to(device)
        a_t_bar = self.alpha_cumprod[t].reshape(bs, 1, 1).to(device)
        sigma_t = self.sigma[t].reshape(bs, 1, 1).to(device)

        x_prediction = self.denoise(x, noise, a_t_bar)
        new_x = x * (a_t.sqrt() * (1 - a_t_bar) / (1 - a_t)) + x_prediction * (a_t_bar.sqrt() * (1 - a_t) / (1 - a_t_bar)) + sigma_t * z

        return new_x

    def denoise(self, x, noise, a_t_bar):
        return (x - noise * (1 - a_t_bar).sqrt()) / a_t_bar.sqrt()
    
    def add_noise(self, x, t=None):
        if t is None:
            t = torch.randint(0, self.steps, (x.size(0),))

        noise = torch.randn(x.shape)

        a_t_bar = self.alpha_cumprod[t]

        return x * a_t_bar.sqrt() + noise * (1 - a_t_bar).sqrt()

class DDPMSparseScheduler(DDPMScheduler):
    def __init__(self, beta_min=0.0001, beta_max=0.02, steos=1024, mode='linear', sigma='bt', pres=1e-5):
        super().__init__(beta_min, beta_max, steos, mode, sigma)
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
        return self.torch2sparse(x, shape)

    def update(self, x, t, noise, shape, stochastic=True):
        # Takse as input a SparseTensor for x and returns a SparseTensor
        # Noise and t are regular tensors
        x = super().update(x.F, t, noise, shape, stochastic)
        return self.torch2sparse(x, shape)