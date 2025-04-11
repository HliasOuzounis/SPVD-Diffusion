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

class ModelNet40(Dataset):
    def __init__(self, path: str | None = None, split: str = "train", sample_size: int = 5_000, categories: list[str]|None = None, load_renders: bool = True) -> None:
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        self.split = split
        
        self.path = path if path is not None else "./data/ModelNet40"
        self.sample_size = sample_size

        self.categories = categories if categories is not None else []

        self.load_renders = load_renders
        
        self.load_data(self.path)

    
    def load_data(self, path: str) -> None:
        pc_path = os.path.join(path, "pointclouds")
        renders_path = os.path.join(path, "processed_renders")
        
        self.pointclouds = []
        self.render_features = []
        self.filenames = []
        
        for category in os.listdir(pc_path):
            if self.categories and category not in self.categories:
                continue
            
            desc = f"Loading renders for {category}" if self.load_renders else f"Loading pointclouds for {category}"
            for file in tqdm(os.listdir(os.path.join(pc_path, category, self.split)), desc=desc):
                model = os.path.join(pc_path, category, self.split, file)
                pointcloud = np.load(model)
                
                self.pointclouds.append(pointcloud)

                file, _ = os.path.splitext(file)
                self.filenames.append(file)

                if self.load_renders:
                    render_features = torch.load(os.path.join(renders_path, category, self.split, f"{file}.pt"), weights_only=True)
                    self.render_features.append(render_features)

        self.pointclouds = np.array(self.pointclouds)
        # Normalize and standardize the pointclouds
        mean = np.mean(self.pointclouds.reshape(-1), axis=0).reshape(1, 1, 1)
        std = np.std(self.pointclouds.reshape(-1), axis=0).reshape(1, 1, 1)

        self.pointclouds = (self.pointclouds - mean) / std
            
    
    def __len__(self) -> int:
        return len(self.pointclouds)
    
    
    def __getitem__(self, idx) -> Any:
        pc = self.pointclouds[idx]
        
        idxs = np.random.choice(pc.shape[0], self.sample_size, replace=False)
        pc = pc[idxs, :]

        selected_file = self.filenames[idx]
        render_features = selected_vew = None

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
    
class ModelNet40Sparse(ModelNet40):
    def __init__(self, path: str | None = None, split: str = "train", sample_size: int = 5_000, categories: list[str]|None = None, load_renders: bool = True) -> None:
        super().__init__(path, split, sample_size, categories)
        
        self.set_voxel_size()
        self.set_scheduler()
        
    def set_scheduler(self, beta_min=0.0001, beta_max=0.02, n_steps=1024, mode='linear'):
        self.noise_scheduler = DDPMScheduler(beta_min, beta_max, n_steps, mode)
    
    def set_voxel_size(self, voxel_size: float = 1e-8) -> None:
        self.voxel_size = voxel_size
        
    def __getitem__(self, idx) -> Any:
        data = super().__getitem__(idx)
        
        pc = data["pc"]
        # render = data["render"]
        render_features = data["render-features"]
        selected_view = data["selected-view"]
        filename = data["filename"]
        
        pc, t, noise = self.noise_scheduler(pc)
        
        pc = pc.numpy()
        noise = noise.numpy()
        
        coords = pc - np.min(pc, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.voxel_size, return_index=True)
        
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(pc[indices], dtype=torch.float)
        noise = torch.tensor(noise[indices], dtype=torch.float)
        
        noisy_pc = SparseTensor(coords=coords, feats=feats)
        noise = SparseTensor(coords=coords, feats=noise)
        t = torch.tensor(t)
         
        return {
            "input": noisy_pc,
            "t": t,
            "noise": noise,
            "render-features": render_features,
            "selected-view": selected_view,
            "filename": filename,
        }
        
def get_dataloaders(path: str, batch_size: int = 32, num_workers: int = 4, categories: list[str] | None = None, load_renders: bool = True) -> tuple[DataLoader, DataLoader]:
    sample_size = 2048
    train_dataset = ModelNet40Sparse(path, "train", sample_size, categories, load_renders)
    test_dataset = ModelNet40Sparse(path, "test", sample_size, categories, load_renders)
    
    train_dataset.set_voxel_size(1e-5)
    test_dataset.set_voxel_size(1e-5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=sparse_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sparse_collate_fn)
    
    return train_loader, test_loader
        

def main():
    tr, te = get_dataloaders("./data/ModelNet40")
    
    print(next(iter(tr)))

if __name__ == "__main__":
    main()