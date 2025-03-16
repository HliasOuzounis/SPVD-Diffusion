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
from PIL import Image

from ..utils import NoiseSchedulerDDPM, VisualTransformer
from my_schedulers.ddpm_scheduler import DDPMScheduler

class ModelNet40(Dataset):
    def __init__(self, path: str | None = None, split: str = "train", sample_size: int = 5_000, categories: list[str]|None = None) -> None:
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        self.split = split

        self.visual_transformer = VisualTransformer()
        
        self.path = path if path is not None else "./data/ModelNet40"
        self.sample_size = sample_size

        self.categories = categories if categories is not None else []
        
        self.load_data(self.path)

    
    def load_data(self, path: str) -> None:
        pc_path = os.path.join(path, "pointclouds")
        renders_path = os.path.join(path, "renders")
        
        self.pointclouds = []
        self.renders = []
        # self.render_features = []
        
        for category in os.listdir(pc_path):
            if self.categories and category not in self.categories:
                continue
            
            for file in tqdm(os.listdir(os.path.join(pc_path, category, self.split)), desc=f"Loading renders for {category}"):
                model = os.path.join(pc_path, category, self.split, file)
                pointcloud = np.load(model)
                
                self.pointclouds.append(pointcloud)

                file, _ = os.path.splitext(file)
    
                # model_views = []
                # for view in os.listdir(os.path.join(renders_path, category, self.split, file)):
                #     image = Image.open(os.path.join(renders_path, category, self.split, file, view)).convert("RGB")
                #     preprocessed = self.visual_transformer.preprocess(image)['pixel_values'][0]
                #     model_views.append(preprocessed)
                # model_views = torch.stack(model_views)
                # self.renders.append(model_views)
                # self.render_features.append(self.visual_transformer(model_views))
            
    
    def __len__(self) -> int:
        return len(self.pointclouds)
    
    
    def __getitem__(self, idx) -> Any:
        pc = self.pointclouds[idx]
        
        idxs = np.random.choice(pc.shape[0], self.sample_size, replace=False)
        pc = pc[idxs, :]

        # renders = self.render_features[idx]
        # selected_render_idx = np.random.randint(0, renders.shape[0])
        # selected_render = self.renders[idx][selected_render_idx].cpu().numpy()
        # render_features = renders[selected_render_idx]
        
        std = 0.02
        noise = np.random.normal(0, std, pc.shape)

        pc += noise
        pc = torch.tensor(pc, dtype=torch.float)
        
        return {
            "idx": idx,
            "pc": pc,
            # "render": selected_render,
            # "render_features": render_features,
        }
    
class ModelNet40Sparse(ModelNet40):
    def __init__(self, path: str | None = None, split: str = "train", sample_size: int = 5_000, categories: list[str]|None = None) -> None:
        super().__init__(path, split, sample_size, categories)
        
        self.set_voxel_size()
        self.set_scheduler()
        
    def set_scheduler(self, beta_min=0.0001, beta_max=0.02, n_steps=1024, mode='linear'):
        # self.noise_scheduler = DDPMScheduler(beta_min, beta_max, n_steps, mode)
        self.noise_scheduler = NoiseSchedulerDDPM(beta_min, beta_max, n_steps, mode)
    
    def set_voxel_size(self, voxel_size: float = 1e-8) -> None:
        self.voxel_size = voxel_size
        
    def __getitem__(self, idx) -> Any:
        data = super().__getitem__(idx)
        
        pc = data["pc"]
        # render = data["render"]
        # render_features = data["render_features"]
        
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
            # "render": render,
            # "render-features": render_features
        }
        
def get_dataloaders(path: str, batch_size: int = 32, num_workers: int = 4, categories: list[str] | None = None) -> tuple[DataLoader, DataLoader]:
    sample_size = 2048
    train_dataset = ModelNet40Sparse(path, "train", sample_size, categories)
    test_dataset = ModelNet40Sparse(path, "test", sample_size, categories)
    
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