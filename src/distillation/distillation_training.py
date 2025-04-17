import torch
import lightning as L

from datasets.shapenet.shapenet_loader import get_dataloaders
from distillation import DistillationProcess, Teacher, Student

import os

model_args = {
    'voxel_size' : 0.1,
    'nfs' : (32, 64, 128, 256), 
    'attn_chans' : 8, 
    'attn_start' : 3, 
    'cross_attn_chans' : 8, 
    'cross_attn_start' : 2, 
    'cross_attn_cond_dim' : 768,
}

diffusion_steps = 1000
starting_checkpoint = f"../checkpoints/distillation/GSPVD/starting.ckpt"

def distillation_init():
    distillation_agent = DistillationProcess(lr=1e-4)
    
    return distillation_agent

def main():
    distillation_agent = distillation_init()

    categories = ['airplane']
    path = "../data/ShapeNet"
    tr, te, val = get_dataloaders(path, categories=categories, load_renders=True, n_steps=diffusion_steps)
    
    N = diffusion_steps
    
    while N > 0:
        previous_checkpoint = starting_checkpoint if N != diffusion_steps else f"../checkpoints/distillation/GSPVD/{N}-steps.ckpt"
        tr.set_scheduler(steps=N)
        te.set_scheduler(steps=N)
        val.set_scheduler(steps=N)
        
        distillation_agent.set_teacher(Teacher(model_params, previous_checkpoint, N, init_steps=diffusion_steps))
        distillation_agent.set_student(Student(model_params, previous_checkpoint))

        max_epochs = 200
        trainer = L.Trainer(
            max_epochs=max_epochs, 
            callbacks=[],
            gradient_clip_val=10.0,
        )

        trainer.fit(distillation_agent, tr, te)
        N = (N + 1) // 2

        torch.save(distillation_agent.student.state_dict(), f"../checkpoints/distillation/GSPVD/{N}-steps.ckpt")
        print(f"Trained Student for {N} steps.")

if __name__ == "__main__":
    main()