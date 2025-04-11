import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SPVD_S, SPVD
from my_models.spvd import SPVUnet
from my_schedulers.ddpm_scheduler import DDPMSparseScheduler

import lightning as L

class Teacher(nn.Module):
    def __init__(self, model_params, model_ckpt, diffusion_steps, init_steps=1000):
        super().__init__()
        self.model = SPVD_S()
        weights = torch.load(model_ckpt, weights_only=True)
        self.load_state_dict(weights)
        self.eval()
        
        self.diffusion_scheduler = DDPMSparseScheduler(steps=diffusion_steps)
        self.current_steps = diffusion_steps
        self.init_steps = init_steps
    
    def forward(self, inp, reference_image=None):
        # Return the noise prediction after 2 diffusion steps
        x_t, t = inp

        # Convert the step from the dataset to the equivalent step in the diffusion process
        # For example, if the dataset has 1024 steps and the current Teacher model has 16 steps,
        # step 200 from the dataset would be step 200 * 16 // 1024 = 3 for the Teacher model
        t = t * self.current_steps // self.init_steps

        bs = t.shape[0]
        shape = (bs, x_t.F.shape[0] // bs, x_t.F.shape[1])

        x_t_1 = self.diffusion_scheduler.sample_step(self.model, x_t, t, shape=shape, device=x_t.F.device)
        x_t_2 = self.diffusion_scheduler.sample_step(self.model, x_t_1, t - 1, shape=shape, device=x_t.F.device)

        # target_x = (x_t_2 - x_t * sigma_t / sigma_t'') / (a_t'' - a_t * sigma_t / sigma_t'')
        # a_t = self.diffusion_scheduler.alpha[t]
        # a_t_2 = self.diffusion_scheduler.alpha[t - 2] if t > 1 else self.diffusion_scheduler.alpha[0]
        # sigma_t = self.diffusion_scheduler.sigma[t]
        # sigma_t_2 = self.diffusion_scheduler.sigma[t - 2] if t > 1 else self.diffusion_scheduler.sigma[0]

        target = x_t_2 - x_t 

        return target_x


class Student(nn.Module):
    def __init__(self, model_params, model_ckpt):
        super().__init__()
        self.model = SPVD_S()
        weights = torch.load(model_ckpt, weights_only=True)
        self.load_state_dict(weights)
    
    def forward(self, inp, reference_image=None):
        x_t, t = x
        return self.model((x_t, t))