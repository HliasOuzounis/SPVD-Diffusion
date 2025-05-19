from models.ddpm_unet_cattn import SPVUnet
import torch
import lightning as L
from models.g_spvd import GSPVD

from torch.utils.data import DataLoader
from dataloaders.shapenet.shapenet_loader import ShapeNet

from utils.hyperparams import load_hyperparams

from my_schedulers.ddpm_scheduler import DDPMSparseScheduler
from my_schedulers.ddim_scheduler import DDIMSparseScheduler
from utils.helper_functions import process_ckpt

import json
import os
from math import ceil
from tqdm.auto import tqdm

from metrics.evaluation_metrics import compute_all_metrics, jsd_between_point_cloud_sets as JSD
from pprint import pprint


def get_ckpt(categories: list[str], steps: int, scheduler: str, distilled: bool, conditional: bool):
    if distilled:
        ckpt_path = f'../checkpoints/distillation/GSPVD/{"-".join(categories)}/{"cond" if conditional else "uncond"}/{steps}-steps.ckpt'
    elif scheduler == 'ddim':
        ckpt_path = f'../checkpoints/distillation/GSPVD/{"-".join(categories)}/1000-steps.ckpt'
    else:
        ckpt_path = f'../checkpoints/ShapeNet/GSPVD/{"-".join(categories)}/{"cond" if conditional else "uncond"}/{steps}-steps.ckpt'

    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt = process_ckpt(ckpt)
    return ckpt


def get_scheduler(scheduler: str, distilled: bool, steps: int, hparams, step_size: int=1):
    if scheduler == 'ddim' and distilled:
        sched = DDIMSparseScheduler(
            beta_min=hparams['beta_min'], 
            beta_max=hparams['beta_max'], 
            steps=steps, 
            init_steps=hparams['n_steps'],
            mode=hparams['mode'],
        )
    elif scheduler == 'ddim':
        sched = DDIMSparseScheduler(
            beta_min=hparams['beta_min'], 
            beta_max=hparams['beta_max'], 
            steps=steps, 
            init_steps=hparams['n_steps'],
            mode=hparams['mode'],
            step_size=step_size,
        )
    elif distilled:
        sched = DDPMSparseScheduler(
            beta_min=hparams['beta_min'], 
            beta_max=hparams['beta_max'], 
            steps=steps, 
            init_steps=hparams['n_steps'],
            mode=hparams['mode'],
        )
    else:
        sched = DDPMSparseScheduler(
            beta_min=hparams['beta_min'], 
            beta_max=hparams['beta_max'], 
            steps=steps, 
            init_steps=steps,
            mode=hparams['mode'],
        )
    return sched


def generate_samples(model, test_loader, sched, conditional: bool, on_all: bool):
    all_ref_pc = []
    all_gen_pc = []

    mean = torch.tensor(test_loader.dataset.mean).cuda()
    std = torch.tensor(test_loader.dataset.std).cuda()

    i = 0
    for datapoint in tqdm(test_loader):
        i += 1
        if i > 5 and not on_all:
            continue

        ref_pc = datapoint['pc'].cuda()
        features = datapoint['render-features'].cuda() if conditional else None

        B, N, C = ref_pc.shape
        gen_pc = sched.sample(model, B, N, reference=features)

        all_ref_pc.append(ref_pc)
        all_gen_pc.append(gen_pc)

    all_ref_pc = torch.cat(all_ref_pc).cuda()
    all_gen_pc = torch.cat(all_gen_pc).cuda()
    
    return all_ref_pc, all_gen_pc


def compute_metrics(all_ref_pc, all_gen_pc):
    results = compute_all_metrics(all_ref_pc, all_gen_pc, batch_size=32)
    results = {k: (v.cpu().detach().item()
                if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    jsd = JSD(all_gen_pc.cpu().numpy(), all_ref_pc.cpu().numpy())
    results['JSD'] = jsd
    pprint('JSD: {}'.format(jsd))

    return results

def save_results(results, categories: list[str], steps: int, scheduler: str, distilled: bool, step_size: int, conditional: bool, norm: bool):
    if distilled:
        folder = f'../metrics/{"-".join(categories)}/{scheduler}/distilled/'
    elif step_size > 1 and scheduler == 'ddim':
        folder = f'../metrics/{"-".join(categories)}/{scheduler}/skip/'
        step_size = ceil(steps / step_size)
    else:
        folder = f'../metrics/{"-".join(categories)}/{scheduler}/retrained/'

    folder += 'uncond/' if not conditional else 'cond/'
    folder += 'norm/' if norm else 'no-norm/'
    
    os.makedirs(folder, exist_ok=True)

    file = os.path.join(folder, f'{steps}-steps.json')
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)


def normalize_to_unit_sphere(batched_points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a batched tensor of 3D points to the unit sphere.
    
    Args:
        batched_points: (B, N, 3) tensor, where B = batch size, N = num points.
        eps: Small value to avoid division by zero.
    
    Returns:
        (B, N, 3) tensor, where all points lie within or on the unit sphere.
    """
    # Center points by subtracting their mean (centroid)
    centroid = torch.mean(batched_points, dim=1, keepdim=True)  # (B, 1, 3)
    centered = batched_points - centroid  # (B, N, 3)

    # Find the maximum distance from the origin for each batch
    max_dist = torch.max(
        torch.sqrt(torch.sum(centered ** 2, dim=-1, keepdim=True)),  # (B, N, 1)
        dim=1, keepdim=True
    ).values  # (B, 1, 1)

    # Normalize by dividing by the maximum distance (+ eps for stability)
    normalized = centered / (max_dist + eps)  # (B, N, 3)

    return normalized


def main():
    ## Hyperparameters
    on_all = True
    scheduler = 'ddpm'
    distilled = True
    conditional = True
    steps_to_run = [1000, 500, 250, 125, 63, 32, 16, 8, 4, 2, 1]

    print(f"Running with distilled={distilled}, scheduler={scheduler}, conditional={conditional}, on_all={on_all} for steps {steps_to_run}")

    testing = False

    categories = ['airplane']

    path = "../data/ShapeNet"

    test_dataset = ShapeNet(path, "test", 2048, categories, load_renders=True, total=2500 if on_all else 1)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    hparams_path = f'../checkpoints/distillation/GSPVD/{"-".join(categories)}/hparams.yaml'
    hparams = load_hyperparams(hparams_path)

    model_args = {
        'voxel_size' : hparams['voxel_size'],
        'nfs' : hparams['nfs'], 
        'attn_chans' : hparams['attn_chans'], 
        'attn_start' : hparams['attn_start'], 
        'cross_attn_chans' : hparams['cross_attn_chans'], 
        'cross_attn_start' : hparams['cross_attn_start'], 
        'cross_attn_cond_dim' : hparams['cross_attn_cond_dim'],
    }
    model = SPVUnet(**model_args)
    model = GSPVD(model=model)

    model = model.cuda().eval()

    step_size = 1
    if scheduler == 'ddim' and not distilled:
        ckpt = get_ckpt(categories, 1000, scheduler, distilled, conditional)
        model.load_state_dict(ckpt)
    
    for steps in steps_to_run:
        if scheduler == 'ddim' and not distilled:
            sched = get_scheduler(scheduler, distilled, 1000, hparams, step_size=step_size)
            step_size *= 2
        else:
            ckpt = get_ckpt(categories, steps, scheduler, distilled, conditional)
            model.load_state_dict(ckpt)
            sched = get_scheduler(scheduler, distilled, steps, hparams)
            
        all_ref_pc, all_gen_pc = generate_samples(model, test_loader, sched, conditional, on_all)
        
        results = compute_metrics(all_ref_pc, all_gen_pc)
        if on_all or testing:
            save_results(results, categories, steps, scheduler, distilled, step_size, conditional, norm=False)
        
        # Normalize point clouds to unit sphere
        all_gen_pc_norm = normalize_to_unit_sphere(all_gen_pc)
        all_ref_pc_norm = normalize_to_unit_sphere(all_ref_pc)

        results = compute_metrics(all_ref_pc_norm, all_gen_pc_norm)
        if on_all or testing:
            save_results(results, categories, steps, scheduler, distilled, step_size, conditional, norm=True)
        
        print(f"Results saved for {steps} steps with scheduler {scheduler} and distilled={distilled}.")


if __name__ == "__main__":
    main()