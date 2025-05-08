import os
import torch
import lightning as L

from dataloaders.shapenet.shapenet_loader import get_dataloaders
from distillation import DistillationProcess, Teacher, Student

from utils.hyperparams import load_hyperparams


torch.set_float32_matmul_precision('medium')


def distillation_init():
    distillation_agent = DistillationProcess(lr=1e-4)
    
    return distillation_agent

def main():
    categories = ['chair']
    
    hparams_path = f'../checkpoints/distillation/GSPVD/{"-".join(categories)}/hparams.yaml'
    hparams = load_hyperparams(hparams_path)
    
    diffusion_steps = hparams['n_steps']
    path = "../data/ShapeNet"
    tr, te, val = get_dataloaders(path, categories=categories, load_renders=True, n_steps=diffusion_steps)

    model_args = {
        'voxel_size' : hparams['voxel_size'],
        'nfs' : hparams['nfs'], 
        'attn_chans' : hparams['attn_chans'], 
        'attn_start' : hparams['attn_start'], 
        'cross_attn_chans' : hparams['cross_attn_chans'], 
        'cross_attn_start' : hparams['cross_attn_start'], 
        'cross_attn_cond_dim' : hparams['cross_attn_cond_dim'],
    }

    scheduler_args = {
        'beta_min': hparams['beta_min'], 
        'beta_max': hparams['beta_max'],  
        'init_steps': hparams['n_steps'],
        'mode': hparams['mode'],
    }
    
    scheduler = "ddpm"
    epochs = iter((1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 2000))
    # epochs = iter((100, 100, 100, 150, 150))
    epochs = iter((500, 500))
    # epochs for   500,  250,  125,   63.   32,   16,    8,    4,   2,   1    steps
    
    N = diffusion_steps
    previous_checkpoint = f"../checkpoints/distillation/GSPVD/{'-'.join(categories)}/{N}-steps.ckpt"

    distillation_agent = distillation_init()
    
    while N > 0:
        distillation_agent.set_teacher(Teacher(model_args, previous_checkpoint, N, scheduler_args, scheduler=scheduler))

        N = (N + 1) // 2
        distillation_agent.set_student(Student(model_args, previous_checkpoint, N, scheduler_args, scheduler=scheduler))
        tr.dataset.set_scheduler(distillation_agent.student.diffusion_scheduler)
        te.dataset.set_scheduler(distillation_agent.student.diffusion_scheduler)
        val.dataset.set_scheduler(distillation_agent.student.diffusion_scheduler)

        try:
            max_epochs = next(epochs)
        except StopIteration:
            print("All distillation steps completed.")
            break

        # checkpoint_callback = L.callbacks.ModelCheckpoint(
        #     dirpath=f"../checkpoints/distillation/GSPVD/{'-'.join(categories)}/intermediate/{N}-steps",
        #     filename=f"{N}-steps-{{epoch:02d}}",
        #     save_top_k=-1,
        #     every_n_epochs=50,
        # )
        
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