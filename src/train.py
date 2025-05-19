import os
import torch
import lightning as L

from models.g_spvd import GSPVD
from models.ddpm_unet_cattn import SPVUnet
from dataloaders.shapenet.shapenet_loader import get_dataloaders
from my_schedulers.ddpm_scheduler import DDPMSparseScheduler
from my_schedulers.ddim_scheduler import DDIMSparseScheduler

from utils.hyperparams import load_hyperparams


torch.set_float32_matmul_precision('medium')

def main():
    categories = ['airplane']
    
    hparams_path = f'../checkpoints/distillation/GSPVD/{"-".join(categories)}/hparams.yaml'
    hparams = load_hyperparams(hparams_path)

    diffusion_steps = 1000
    path = "../data/ShapeNet"
    tr, te, val = get_dataloaders(path, categories=categories, load_renders=True, n_steps=diffusion_steps, total=2000)

    model_args = {
        'voxel_size' : hparams['voxel_size'],
        'nfs' : hparams['nfs'], 
        'attn_chans' : hparams['attn_chans'], 
        'attn_start' : hparams['attn_start'], 
        'cross_attn_chans' : hparams['cross_attn_chans'], 
        'cross_attn_start' : hparams['cross_attn_start'], 
        'cross_attn_cond_dim' : hparams['cross_attn_cond_dim'],
    }

    scheduler = "ddpm"
    scheduler_args = {
        'beta_min': hparams['beta_min'], 
        'beta_max': hparams['beta_max'],  
        'init_steps': hparams['n_steps'],
        'mode': hparams['mode'],
    }

    if scheduler == 'ddim':
        sched = DDIMSparseScheduler(
            beta_min=hparams['beta_min'], 
            beta_max=hparams['beta_max'], 
            steps=diffusion_steps, 
            init_steps=hparams['n_steps'],
            mode=hparams['mode'],
        )
    else:
        sched = DDPMSparseScheduler(
            beta_min=hparams['beta_min'], 
            beta_max=hparams['beta_max'], 
            steps=diffusion_steps, 
            init_steps=hparams['n_steps'],
            mode=hparams['mode'],
        )
    
    tr.scheduler = sched
    te.scheduler = sched
    val.scheduler = sched

    epochs = 1000
    lr = 1e-4

    model = SPVUnet(**model_args)
    model = GSPVD(model=model, lr=lr, training_steps=epochs * len(tr), uncond_prob=0.1)
    

    trainer = L.Trainer(
        max_epochs=epochs, 
        callbacks=[],
        gradient_clip_val=10.0,
    )
    
    trainer.fit(model=model, train_dataloaders=tr, val_dataloaders=val)
    
    folder = f"../checkpoints/ShapeNet/GSPVD/{'-'.join(categories)}/{scheduler}/"
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), f"{folder}/{diffusion_steps}-steps.ckpt")

if __name__ == "__main__":
    main()