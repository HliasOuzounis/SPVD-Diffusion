import os
import trimesh
from tqdm import tqdm
import zipfile

import numpy as np
import torch

from transformers import ViTImageProcessor, ViTModel
from PIL import Image


class VisualTransformer:
    model_name = "google/vit-base-patch16-224-in21k"  # Example model
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to("cuda")

    def preprocess(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.to("cuda")

    def __call__(self, x):
        with torch.no_grad():
            outputs = self.model(x)
        return outputs.last_hidden_state.cpu()


class ModelNet40Downloader:
    def __init__(self, path: str | None = None, split: str = "train") -> None:
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        self.split = split
        
        self.path = path if path is not None else "./data/ModelNet40/"
        
        if not os.path.exists(self.path):
            downlaod(self.path)

        self.path += "off"

    def __len__(self) -> int:
        return len(self.trimesh_data)

    def __getitem__(self, idx) -> tuple[trimesh.Trimesh, str]:
        return self.trimesh_data[idx], self.labels[idx]
    
    def download(self, path: str | None = None) -> None:
        if path is None:
            path = "./data/ModelNet40"

        os.makedirs(path, exist_ok=True)

        print(f"Downloading ModelNet40 dataset at {path}")

        zipname = "ModelNet40.zip"
        cmd = f"wget -q --show-progress http://modelnet.cs.princeton.edu/{zipname} -P {path}"
        os.system(cmd)

        with zipfile.ZipFile(os.path.join(path, zipname), "r") as zf:
            for member in tqdm(zf.infolist(), desc="Extracting"):
                try:
                    if member.is_dir():
                        continue
                    member.filename = member.filename.replace("ModelNet40/", "off/")
                    zf.extract(member, path)
                except zipname.error as e:
                    pass

        # cmd = f"rm {os.path.join(path, zipname)}"
        os.system(cmd)


    def save_to_stl(self, path: str) -> None:
        for i, category in enumerate(os.listdir(self.path)):
            os.makedirs(os.path.join(path, category, self.split), exist_ok=True)

            files = os.listdir(os.path.join(self.path, category, self.split))
            
            for file in tqdm(files, desc=f"Saving {category} to stl", unit="mesh"):
                with open(os.path.join(self.path, category, self.split, file), "r") as f:
                    mesh = trimesh.load(f, file_type="off")
                    
                    filename = os.path.join(path, f"{category}/{self.split}/{file.replace('off', 'stl')}")
                    mesh.export(filename)
            
    
    def save_to_pointcloud(self, path: str, num_points: int = 4096):
        for i, category in enumerate(os.listdir(self.path)):
            files = os.listdir(os.path.join(self.path, category, self.split))

            os.makedirs(os.path.join(path, category, self.split), exist_ok=True)
            
            for file in tqdm(files, desc=f"Saving {category} to pointcloud", unit="mesh"):
                with open(os.path.join(self.path, category, self.split, file), "r") as f:
                    mesh = trimesh.load(f, file_type="off")
                    
                    filename = os.path.join(path, f"{category}/{self.split}/{file.replace('off', 'npy')}")
                    
                    pointcloud = np.array(mesh.sample(num_points))
                    
                    pointcloud -= np.mean(pointcloud, axis=0)
                    pointcloud /= np.max(np.linalg.norm(pointcloud, axis=1))
                    
                    np.save(filename, pointcloud)
    
    def parse_renders(self, path: str) -> None:
        vit = VisualTransformer()
        renders_path = os.path.join("./data/ModelNet40/", "renders")

        for i, category in enumerate(os.listdir(self.path)):            
            for file in tqdm(os.listdir(os.path.join(renders_path, category, self.split)), desc=f"Loading renders for {category}"):
                model_views = []
                for view in os.listdir(os.path.join(renders_path, category, self.split, file)):
                    try:
                        image = Image.open(os.path.join(renders_path, category, self.split, file, view)).convert("RGB")
                        preprocessed = vit.preprocess(image)['pixel_values'][0]
                        model_views.append(preprocessed)
                    except:
                        print(f"Error loading {category}/{self.split}/{file}/{view}")
                        exit()

                model_views = torch.stack(model_views)
                model_views = vit(model_views)

                os.makedirs(os.path.join(path, category, self.split), exist_ok=True)
                
                save_path = os.path.join(path, f"{category}/{self.split}/{file}.npy")
                np.save(save_path, model_views.numpy())


if __name__ == "__main__":
    for split in ["test", "train"]:
        dtst = ModelNet40Downloader(split=split)
        # dtst.save_to_stl(path="./data/ModelNet40/stl_models")
        # dtst.save_to_pointcloud(path="./data/ModelNet40/pointclouds", num_points=15_000)
        dtst.parse_renders(path="./data/ModelNet40/processed_renders")