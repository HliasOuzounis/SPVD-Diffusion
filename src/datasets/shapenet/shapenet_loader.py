import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn

import numpy as np

from typing import Any
import os
from tqdm import tqdm

from my_schedulers.ddpm_scheduler import DDPMScheduler
from my_schedulers.ddim_scheduler import DDIMScheduler

from .shapenet_utils import synsetid_to_category, category_to_synsetid


class ShapeNet(Dataset):
    def __init__(self, path: str|None = None, split: str = "train", sample_size: int = 5_000, categories: list[str]|None = None, load_renders: bool = True) -> None:
        assert split in ["train", "test", "val"], "split should be either 'train' or 'test' or 'val'"
        self.split = split
        
        self.path = path if path is not None else "./data/ShapeNet"
        self.sample_size = sample_size

        self.categories = [category_to_synsetid[cat] for cat in categories] if categories is not None else []
        self.load_renders = load_renders
        
        self.load_data(self.path)
    
    def load_data(self, path: str) -> None:
        pc_path = os.path.join(path, "pointclouds")
        renders_path = os.path.join(path, "embed_renders")

        self.pointclouds = []
        self.render_features = []
        
        self.filenames = []
        
        for category in os.listdir(pc_path):
            if self.categories and category not in self.categories:
                continue
            
            desc = f"Loading ({self.split}) {'renders' if self.load_renders else 'pointclouds'} for {synsetid_to_category[category]} ({category})"
            c = 0
            for file in tqdm(os.listdir(os.path.join(pc_path, category, self.split)), desc=desc):
                if c > 1499:
                    continue
                c += 1
                
                model = os.path.join(pc_path, category, self.split, file)
                pointcloud = np.load(model)

                self.pointclouds.append(pointcloud)

                file, _ = os.path.splitext(file)
                self.filenames.append(os.path.join(category, self.split, file))

                if self.load_renders:
                    render_features = []
                    for view in range(8):
                        render_file = os.path.join(renders_path, category, self.split, file, f"00{view}_patch_embs.pt")
                        if os.path.exists(render_file):
                            render_features.append(torch.load(render_file, weights_only=True))
                    render_features = torch.stack(render_features, dim=0)
                    self.render_features.append(render_features)
        
        self.pointclouds = np.array(self.pointclouds)
        # Normalize and standardize the pointclouds
        mean = np.mean(self.pointclouds.reshape(-1), axis=0).reshape(1, 1, 1)
        std = np.std(self.pointclouds.reshape(-1), axis=0).reshape(1, 1, 1)

        self.pointclouds = (self.pointclouds - mean) / std

    def class_weight(self, category: str) -> float:
        return self.class_sizes[category] / len(self.pointclouds)
    
    def __len__(self) -> int:
        return len(self.pointclouds)
    
    def __getitem__(self, idx) -> Any:
        pc = self.pointclouds[idx]
        
        idxs = np.random.choice(pc.shape[0], self.sample_size, replace=False)
        pc = pc[idxs, :]

        selected_file = self.filenames[idx]

        render_features = None
        selected_view = None
        
        if self.load_renders:
            render_features = self.render_features[idx]
            selected_view = np.random.randint(0, render_features.shape[0])
            render_features = render_features[selected_view]
        
        std = 0.02
        noise = np.random.normal(0, std, pc.shape)

        pc += noise
        pc = torch.tensor(pc, dtype=torch.float)
        
        return {
            "idx": idx,
            "pc": pc,
            "render-features": render_features,
            "selected-view": selected_view,
            "filename": selected_file,
        }


class ShapeNetSparse(ShapeNet):
    def __init__(self, path: str | None = None, split: str = "train", sample_size: int = 5_000, categories: list[str]|None = None, load_renders: bool = True, n_steps=1024) -> None:
        super().__init__(path, split, sample_size, categories, load_renders)
        
        self.set_voxel_size()
        self.set_scheduler(DDIMScheduler(steps=n_steps))
        
    def set_scheduler(self, scheduler):
        self.noise_scheduler = scheduler
    
    def set_voxel_size(self, voxel_size: float = 1e-8) -> None:
        self.voxel_size = voxel_size
        
    def __getitem__(self, idx) -> Any:
        data = super().__getitem__(idx)
        
        pc = data["pc"]
        # render = data["render"]
        render_features = data["render-features"]
        selected_view = data["selected-view"]
        filename = data["filename"]
        
        noisy_pc, t, noise = self.noise_scheduler(pc)
        
        noisy_pc = noisy_pc.numpy()
        noise = noise.numpy()
        
        coords = noisy_pc - np.min(noisy_pc, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.voxel_size, return_index=True)
        
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(noisy_pc[indices], dtype=torch.float)
        noise = torch.tensor(noise[indices], dtype=torch.float)
        
        noisy_pc = SparseTensor(coords=coords, feats=feats)
        noise = SparseTensor(coords=coords, feats=noise)
        t = torch.tensor(t)
         
        return {
            "input": noisy_pc,
            "pc": pc,
            "t": t,
            "noise": noise,
            "render-features": render_features,
            "selected-view": selected_view,
            "filename": filename,
        }
        
def get_dataloaders(path: str, batch_size: int = 32, sample_size: int = 2048, num_workers: int = 4, categories: list[str] | None = None, load_renders: bool = True, n_steps=1024) -> tuple[DataLoader, DataLoader]:
    train_dataset = ShapeNetSparse(path, "train", sample_size, categories, load_renders, n_steps)
    test_dataset = ShapeNetSparse(path, "test", sample_size, categories, load_renders, n_steps)
    val_dataset = ShapeNetSparse(path, "val", sample_size, categories, load_renders, n_steps)
    
    train_dataset.set_voxel_size(1e-5)
    test_dataset.set_voxel_size(1e-5)
    val_dataset.set_voxel_size(1e-5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=sparse_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sparse_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sparse_collate_fn)
    
    return train_loader, test_loader, val_loader