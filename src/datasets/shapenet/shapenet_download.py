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


class ShapeNetDownloader:
    def __init__(self, path: str | None = None) -> None:
        self.path = path if path is not None else "./data/ShapeNetCore/"
        
        if not os.path.exists(self.path + "obj"):
            self.download(self.path + "obj")
        
    def download(self, path: str | None = None) -> None:
        if path is None:
            path = "./data/ShapeNetCore/obj"
        
        os.makedirs(path, exist_ok=True)
        
        print(f"Downloading ShapeNet dataset at {path}")
        
        zipname = "ShapeNetCore.v2.zip"
        cmd = f"./src/datasets/shapenet/manual_download.sh"
        os.system(cmd)
        
        # Extract the downloaded zip file
        # with zipfile.ZipFile(zipname, 'r') as zip_ref:
        #     zip_ref.extractall(path)
        
        # Rename folders based on synsetid_to_cate mapping
        for synset_id, category in synsetid_to_cate.items():
            src_folder = os.path.join(path, synset_id)
            dest_folder = os.path.join(path, category)
            if os.path.exists(src_folder):
                os.rename(src_folder, dest_folder)

if __name__ == "__main__":
    downloader = ShapeNetDownloader()