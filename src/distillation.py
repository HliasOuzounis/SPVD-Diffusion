import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from datasets.modelnet40.modelnet40_loader import get_dataloaders
from distillation import DistillationProcess, Teacher, Student
from train import train

import os

model_params = {
    "down_blocks": [{
        "features_list": [32, 32, 192, 256],
        "num_layers_list": 1,
        "attn_heads": None
    }],
    "up_blocks": [{
            "features_list": [256, 192, 32, 32],
            "num_layers_list": 2,
            "attn_heads": None
    }],
    "t_emb_features": 64,
}

diffusion_steps = 1000
starting_checkpoint = f"../checkpoints/ModelNet/distillation/{diffusion_steps}-steps.ckpt"

def distillation_init():
    distillation_agent = DistillationProcess(lr=1e-4)
    
    return distillation_agent

def main():
    distillation_agent = distillation_init()

    tr, te = get_dataloaders("../data/ModelNet40", categories=['bottle'])
    
    N = diffusion_steps
    
    if not os.path.exists(starting_checkpoint):
        print(f"Starting checkpoint {starting_checkpoint} not found.")
        print("Training the model from scratch.")
        model = Student(model_params)
        train(model, starting_checkpoint, tr, te)
    
    while N > 0:
        previous_checkpoint = f"../checkpoints/ModelNet/distillation/{N}-steps.ckpt"
        distillation_agent.set_teacher(Teacher(model_params, previous_checkpoint, N))
        distillation_agent.set_student(Student(model_params))

        checkpoint_callback = ModelCheckpoint(
            dirpath=f'../checkpoints/ModelNet40/distillation/',
            filename=f'{N // 2}-steps.ckpt',
            save_last=True
        )
        trainer = L.Trainer(
            max_epochs=40, 
            callbacks=[checkpoint_callback],
            gradient_clip_val=10.0,
        )

        trainer.fit(distillation_agent, tr, te)
        N //= 2

        print(f"Trained Student for {N} steps.")

if __name__ == "__main__":
    main()