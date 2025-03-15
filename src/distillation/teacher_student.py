import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SPVD
from my_models.spvd import SPVUnet
from utils.schedulers import DDPMSparseSchedulerGPU

import lightning as L

class Teacher(nn.Module):
    def __init__(self, model_params, model_ckpt, diffusion_steps, init_steps=1000):
        super().__init__()
        self.model = SPVD(model_params)
        weights = torch.load(model_ckpt, weights_only=True)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        self.diffusion_scheduler = DDPMSparseSchedulerGPU(n_step=diffusion_steps)
        self.current_steps = diffusion_steps
        self.init_steps = init_steps
    
    def forward(self, inp):
        # Return the noise prediction after 2 diffusion steps
        x_t, t = inp

        # Convert the step from the dataset to the equivalent step in the diffusion process
        # For example, if the dataset has 1000 steps and the current Teacher model has 16 steps,
        # step 200 from the database would be step 200 * 16 // 1000 = 3 for the Teacher model
        t = t * self.current_steps // self.init_steps
        
        x_t = self.diffusion_scheduler.sample_step(self, x_t, t, i=t, emb=None, shape=x_t.shape, device=x_t.device)
        
        noise_pred = self.model(x_t, t)

        return noise_pred


class Student(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model = SPVUnet(**model_params)
    
    def forward(self, x):
        return self.model(x)