import os
from tqdm.auto import tqdm

parent_path = "data/ShapeNet"
for category in tqdm(os.listdir(os.path.join(parent_path, "pointclouds"))):
    train = set(os.listdir(os.path.join(os.path.join(parent_path, "pointclouds"), category, "train")))
    test = set(os.listdir(os.path.join(os.path.join(parent_path, "pointclouds"), category, "test")))
    val = set(os.listdir(os.path.join(os.path.join(parent_path, "pointclouds"), category, "val")))

    train = {file.split(".")[0] for file in train}
    test = {file.split(".")[0] for file in test}
    val = {file.split(".")[0] for file in val}

    
    for folder in ("embed_renders", "embed_sketches", "renders", "sketches"):
        os.makedirs(os.path.join(parent_path, folder, category, "train"), exist_ok=True)
        os.makedirs(os.path.join(parent_path, folder, category, "test"), exist_ok=True)
        os.makedirs(os.path.join(parent_path, folder, category, "val"), exist_ok=True)
        for file in os.listdir(os.path.join(parent_path, folder, category)):
            if file in train:
                os.rename(os.path.join(parent_path, folder, category, file), os.path.join(parent_path, folder, category, "train", file))
            elif file in test:
                os.rename(os.path.join(parent_path, folder, category, file), os.path.join(parent_path, folder, category, "test", file))
            elif file in val:
                os.rename(os.path.join(parent_path, folder, category, file), os.path.join(parent_path, folder, category, "val", file))

            