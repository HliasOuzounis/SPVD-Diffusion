import os
import torch
import lightning as L

from dataloaders.shapenet.shapenet_loader import get_dataloaders
from distillation import DistillationProcess, Teacher, Student

torch.set_float32_matmul_precision('medium')

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

    categories = ['chair']
    path = "../data/ShapeNet"
    tr, te, val = get_dataloaders(path, categories=categories, load_renders=True, n_steps=diffusion_steps)
    
    scheduler = "ddpm"
    epochs = iter((200, 200, 200, 200, 200))
    N = 250
    # N = diffusion_steps
    previous_checkpoint = f"../checkpoints/distillation/GSPVD/{'-'.join(categories)}/{N}-steps.ckpt" if N != diffusion_steps else starting_checkpoint
    
    while N > 0:
        distillation_agent.set_teacher(Teacher(model_args, previous_checkpoint, N, scheduler=scheduler))

        N = (N + 1) // 2
        distillation_agent.set_student(Student(model_args, previous_checkpoint, N, scheduler=scheduler))
        tr.dataset.set_scheduler(distillation_agent.student.diffusion_scheduler)
        te.dataset.set_scheduler(distillation_agent.student.diffusion_scheduler)
        val.dataset.set_scheduler(distillation_agent.student.diffusion_scheduler)

        try:
            max_epochs = next(epochs)
        except StopIteration:
            print("All distillation steps completed.")
            break
        
        trainer = L.Trainer(
            max_epochs=max_epochs, 
            callbacks=[],
            gradient_clip_val=10.0,
        )
        
        print(f"Training Student for {N} steps with {scheduler} scheduler.")
        trainer.fit(distillation_agent, tr, val)
        print(f"Trained Student for {N} steps.")

        folder_path = f"../checkpoints/distillation/GSPVD/{'-'.join(categories)}"
        os.makedirs(folder_path, exist_ok=True)
        new_checkpoint = f"{folder_path}/{N}-steps.ckpt"
        torch.save(distillation_agent.student.state_dict(), new_checkpoint)
        
        previous_checkpoint = new_checkpoint

if __name__ == "__main__":
    main()