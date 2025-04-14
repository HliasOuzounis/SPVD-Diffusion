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

diffusion_steps = 1024
starting_checkpoint = f"../checkpoints/ModelNet/distillation/{diffusion_steps}-steps.ckpt"
retrain = True

def distillation_init():
    distillation_agent = DistillationProcess(lr=1e-4)
    
    return distillation_agent

def main():
    distillation_agent = distillation_init()

    tr, te = get_dataloaders("../data/ModelNet40", categories=['bottle'])
    
    N = diffusion_steps
    
    if not os.path.exists(starting_checkpoint) or retrain:
        print(f"Starting checkpoint {starting_checkpoint} not found.")
        print("Training the model from scratch.")

        model = Student(model_params)
        train(model, tr, te, epochs=50)
        torch.save(model.state_dict(), starting_checkpoint)
    
    while N > 0:
        previous_checkpoint = f"../checkpoints/ModelNet/distillation/{N}-steps.ckpt"
        distillation_agent.set_teacher(Teacher(model_params, previous_checkpoint, N))
        distillation_agent.set_student(Student(model_params, previous_checkpoint))

        trainer = L.Trainer(
            max_epochs=100, 
            callbacks=[],
            gradient_clip_val=10.0,
        )

        trainer.fit(distillation_agent, tr, te)
        N //= 2

        torch.save(distillation_agent.student.state_dict(), f"../checkpoints/ModelNet/distillation/{N}-steps.ckpt")
        print(f"Trained Student for {N} steps.")

if __name__ == "__main__":
    main()