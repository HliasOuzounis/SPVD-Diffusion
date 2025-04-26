import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm_unet_cattn import SPVUnet
# from my_models.spvd import SPVUnet
from my_schedulers.ddpm_scheduler import DDPMSparseScheduler
from my_schedulers.ddim_scheduler import DDIMSparseScheduler

import lightning as L

class Teacher(nn.Module):
    def __init__(self, model_params, model_ckpt, diffusion_steps, scheduler='ddpm'):
        super().__init__()
        self.model = SPVUnet(**model_params)
        weights = torch.load(model_ckpt, weights_only=True)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        self.load_state_dict(weights)
        
        self.type = scheduler
        self.diffusion_scheduler = (
            DDPMSparseScheduler(steps=diffusion_steps, init_steps=1000) 
            if scheduler == 'ddpm'
            else DDIMSparseScheduler(steps=diffusion_steps, init_steps=1000)
        )
    
        # Freeze the model parameters
        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, inp):
        xt, t, reference = inp

        bs = t.shape[0]
        shape = (bs, xt.F.shape[0] // bs, xt.F.shape[1])
        device = xt.F.device

        noise = self.model((xt, t, reference))
        noise = noise.reshape(shape)
        
        return noise
    
    def target(self, inp):
        x_t, t, reference = inp
        
        bs = t.shape[0]
        shape = (bs, x_t.F.shape[0] // bs, x_t.F.shape[1])
        device = x_t.F.device
        
        scaled_t = t * 2

        x_t1 = self.diffusion_scheduler.sample_step(self.model, x_t, scaled_t, shape=shape, device=device, reference=reference, stochastic=False)
        x_t2 = self.diffusion_scheduler.sample_step(self.model, x_t1, scaled_t - 1, shape=shape, device=device, reference=reference, stochastic=False)
        
        x_t2 = x_t2.F.reshape(shape)
        return x_t2


class Student(nn.Module):
    def __init__(self, model_params, model_ckpt, diffusion_steps, scheduler='ddpm'):
        super().__init__()
        self.model = SPVUnet(**model_params)
        weights = torch.load(model_ckpt, weights_only=True)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        self.load_state_dict(weights)
        
        self.diffusion_scheduler = (
            DDPMSparseScheduler(steps=diffusion_steps, init_steps=1000) 
            if scheduler == 'ddpm'
            else DDIMSparseScheduler(steps=diffusion_steps, init_steps=1000)
        )
        self.type = scheduler

        scale = 1e-3
        self.perturb_weights(scale)


    def perturb_weights(self, scale):
        for param in self.parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param) * scale
        

    def forward(self, inp):
        xt, t, reference = inp

        bs = t.shape[0]
        shape = (bs, xt.F.shape[0] // bs, xt.F.shape[1])
        device = xt.F.device

        noise = self.model((xt, t, reference))
        noise = noise.reshape(shape)
        
        return noise
    
    def target(self, inp):
        xt, t, reference = inp
        
        bs = t.shape[0]
        shape = (bs, xt.F.shape[0] // bs, xt.F.shape[1])
        device = xt.F.device

        x_t1 = self.diffusion_scheduler.sample_step(self.model, xt, t, shape=shape, device=device, reference=reference, stochastic=False)

        x_t1 = x_t1.F.reshape(shape)
        return x_t1