# Dataset Class for ShapeNet Point Cloud
#
# This file defines a dataset class designed to load the ShapeNet Point Cloud dataset.
# It is an adapted version of the original implementation from PointFlow:
# (https://github.com/stevenygd/PointFlow.git).
#
# Enhancements and Objectives:
# 1. Improved Readability:
#    - The code has been refactored for better clarity and maintainability.
# 2. Additional Data Loading:
#    - Added functionality to load supplementary data, such as:
#      - Associated image renders.
#      - Latent space representations.
#
# Notes on Modifications:
# - Comments throughout the code explain deviations and design choices
#   made compared to the original implementation.


import os
import torch
import random
from tqdm import tqdm 

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn


def get_raw_file_names(directory, suffix):
    """
    Args:
        directory (str): The path to the directory.
        suffix (str): The file suffix to filter by (e.g., ".npy").

    Returns:
        list: A list of file names without the suffix.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The specified path is not a directory: {directory}")

    raw_file_names = [
        os.path.splitext(file)[0]  # Extract the name without the extension
        for file in os.listdir(directory)  # List all files in the directory
        if file.endswith(suffix)  # Filter by the specified suffix
    ]

    return raw_file_names


def np_save_load(pc_path, synset_id=None, uid=None):
    """
    Practically calls np.load on pc_path.

    Args:
        pc_path (str): Path to the .npy file.
        synset_id (str, optional): Synset ID for the point cloud (category or class identifier).
        uid (str, optional): Unique identifier for the point cloud (e.g., file name or ID).

    Returns:
        np.ndarray: Loaded point cloud array.

    Raises:
        Exception: If the file cannot be loaded, with an optimized error message.
    """
    try:
        return np.load(pc_path)
    except Exception as e:
        message = f"Could not load point cloud from file: {pc_path}\n"
        if synset_id is not None:
            message += f" - Synset ID: {synset_id}\n"
        if uid is not None:
            message += f" - UID: {uid}\n"
        message += f"Error: {str(e)}"
        raise Exception(message)


# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(
        self,
        root_dir,  # path to the ShapeNet dataset (or dataset with similar structure)
        subdirs,  # contains the UIDs of the classes to load
        tr_sample_size=10000,
        te_sample_size=None,
        split="train",
        transforms=[],  # list of transforms to apply to the point clouds
        normalize_per_shape=False,
        random_subsample=False,
        normalize_std_per_axis=False,
        all_points_mean=None,  # use predefined mean value
        all_points_std=None,  # use predefined std value
        # NOTE: we have removed scale from the arguments, as it wasn't used.
    ):
        self.root_dir = root_dir
        self.subdirs = subdirs
        self.split = split
        self.tr_sample_size, self.te_sample_size = tr_sample_size, te_sample_size
        self.transforms = transforms
        self.random_subsample = random_subsample

        # --- Data Loading --- #
        # for each point cloud we want to store:
        #   1. the points of the point cloud
        self.all_points = []
        #   2. the category of the point cloud (in case we train with multiple categories)
        self.all_categories = []
        #   3. the uid of the point cloud, that is its unique identity
        self.all_uids = []

        # Parse all subdirectories (subdirectory <-> category)
        for subd in self.subdirs:
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)

            # get all the uids for the point clouds of this category
            subd_uids = get_raw_file_names(sub_path, ".npy")

            # load all point cloud files

            for uid in tqdm(subd_uids, desc=f"Loading point clouds for category {synsetid_to_cate[subd]}"):
                pc_path = os.path.join(sub_path, uid + ".npy")
                # This function practically calls np.load, but in case of an error,
                # it will display a prettier error message :P
                pc = np_save_load(pc_path, subd, uid)
                # The current dataset version contains point clouds with 15000 points
                assert pc.shape[0] == 15000

                # storing the information for each point cloud
                self.all_points.append(pc)
                self.all_categories.append(subd)
                self.all_uids.append(uid)

        # NOTE: The original dataset implementation shuffles the order of the points
        #       We skip this step, as we let the dataloader shuffle the data

        # Represent all points as a numpy array
        self.all_points = np.stack(self.all_points)

        # --- Normalization --- #
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis

        # If mean value is predefined, then std should be as well and viseversa.
        assert (all_points_mean is None and all_points_std is None) or (
            all_points_mean is not None and all_points_std is not None
        ), "Either both 'all_points_mean' and 'all_points_std' must be None, or neither."
        if all_points_mean is not None and all_points_std is not None:
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
            print("Using global mean and variance for all point clouds.")
            print(" - Ignoring the following arguments:")
            print(f"   - normalize_per_shape: {normalize_per_shape}")
            print(f"   - normalize_std_per_axis: {normalize_std_per_axis}")
        elif self.normalize_per_shape:  # normalize each shape independantly
            # if normalize per shape: calculcate mean and var for each shape of the dataset
            B, N, C = self.all_points.shape
            self.all_points_mean = self.all_points.mean(
                axis=1, keepdims=True
            )  # [B, 1, 3]
            if self.normalize_std_per_axis:
                self.all_points_std = self.all_points.std(
                    axis=1, keepdims=True
                )  # [B, 1, 3]
            else:
                self.all_points_std = (
                    self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
                )
        else:  # normalize all shapes across the dataset
            B, N, C = self.all_points.shape
            self.all_points_mean = (
                self.all_points.reshape(-1, C).mean(0).reshape(1, 1, C)
            )
            if self.normalize_std_per_axis:
                self.all_points_std = (
                    self.all_points.reshape(-1, C).std(axis=0).reshape(1, 1, C)
                )
            else:
                self.all_points_std = (
                    self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
                )

        # Normalize the point coordinates
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        # NOTE: Following the dataset from PointFlow, they perform a split where, from the 15k points
        #       the 10k first are considered to be `train` points, while the last 5k `test` points
        self.train_points = self.all_points[:, :10000]
        if self.te_sample_size is not None:
            self.test_points = self.all_points[:, 10000:]

        if self.tr_sample_size > 10000:
            print(
                f"Maximum tr_sample_size is 10k. Given value {self.tr_sample_size} will be clamped."
            )
            self.tr_sample_size = 10000
        if self.te_sample_size is None:
            print(
                "`te_sample_size` was set to None, dataset will not have test points."
            )
        elif self.te_sample_size > 5000:
            print(
                f"Maximum te_sample_size is 5k. Given value {self.te_sample_size} will be clamped."
            )
            self.te_sample_size = 5000

    def get_pc_stats(self, idx):
        # This function returns the mean and std for the indexed point cloud,
        # so that the reverse normalization can be applied to it.
        if self.normalize_per_shape:
            m = self.all_points_mean[idx]
            s = self.all_points_std[idx]
        else:
            m = self.all_points_mean[
                0
            ]  # shape of self.all_points_mean: [1, 1, 3] -->  m.shape = [1, 3]
            s = self.all_points_std[
                0
            ]  # shape of self.all_points_mean: [1, 1, 1/3] --> s.shape = [1, 1/3]

        return m, s

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):

        # get the train point cloud
        train_pc = self.train_points[idx]  # shape: 10000 x 3

        if self.random_subsample:  # select a random subsample from the first 10k points
            train_idxs = np.random.choice(train_pc.shape[0], self.tr_sample_size)
        else:  # keep the K-first points
            train_idxs = np.arange(self.tr_sample_size)

        train_pc = torch.from_numpy(train_pc[train_idxs, :]).float()

        # get the test point cloud
        if self.te_sample_size is not None:
            test_pc = self.test_points[idx]
            if self.random_subsample:
                test_idxs = np.random.choice(test_pc.shape[0], self.te_sample_size)
            else:
                test_idxs = np.arange(self.te_sample_size)

            test_pc = torch.from_numpy(test_pc[test_idxs, :]).float()
        else:
            test_pc = None

        # get mean and std to be able to perform reverse normalization
        m, s = self.get_pc_stats(idx)

        # get the id of the category
        synset_id = self.all_categories[idx]

        # get the uid of the object
        uid = self.all_uids[idx]

        # TODO: If test points is None, don't add it to the dictionary! (Raises error)
        sample = {
            "idx": idx,
            "train_points": train_pc,
            "test_points": test_pc,
            "mean": m,
            "std": s,
            "synset_id": synset_id,
            "uid": uid,
        }

        for t in self.transforms:
            sample = t(sample)

        return sample


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(
        self,
        root_dir="data/ShapeNetCore.v2.PC15k",
        categories=["airplane"],
        tr_sample_size=10000,
        te_sample_size=None,
        split="train",
        transforms=[],
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=False,
        all_points_mean=None,
        all_points_std=None,
    ):

        if not isinstance(categories, (tuple, list)):
            categories = [categories]
        self.categories = categories
        if "all" in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.categories]

        # as defined in the PointFlow version
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super().__init__(
            root_dir,
            self.synset_ids,
            tr_sample_size,
            te_sample_size,
            split,
            transforms,
            normalize_per_shape,
            random_subsample,
            normalize_std_per_axis,
            all_points_mean,
            all_points_std,
        )


class ShapeNet15kPointCloudsViTEmbs(ShapeNet15kPointClouds):
    def __init__(
        self,
        root_dir,
        embed_dir,
        embed_type="patch_emb",  # patch_emb or global_emb
        categories=["airplane"],
        tr_sample_size=10000,
        te_sample_size=None,
        split="train",
        transforms=[],
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=False,
        all_points_mean=None,
        all_points_std=None,
    ):

        assert embed_type in [
            "patch_emb",
            "global_emb",
        ], "Invalid embedding type. Choose from ['patches', 'global_vec']"
        self.embed_type = embed_type

        super().__init__(
            root_dir,
            categories,
            tr_sample_size,
            te_sample_size,
            split,
            transforms,
            normalize_per_shape,
            normalize_std_per_axis,
            random_subsample,
            all_points_mean,
            all_points_std,
        )

        # load the embeddings
        self.vit_embeddings = {}
        for cat_id, uid in tqdm(zip(self.all_categories, self.all_uids), desc=f'Loading renders'):
            # folder that contains the embeddings
            # TODO: I should also add split folders to the path
            path = os.path.join(embed_dir, cat_id, uid)
            self.vit_embeddings[uid] = (path, self.get_embeddings(path))
        # self.vit_embeddings = torch.stack(vit_embeddings)
        assert len(self.vit_embeddings) == len(self.all_points)

    def get_embeddings(self, path):

        # load all files in the embedding folder
        all_files = [
            file
            for file in os.listdir(path)
            if os.path.isfile(os.path.join(path, file))
        ]

        # filter based on the embed_type
        if self.embed_type == "patch_emb":
            filtered_files = [file for file in all_files if "patch_embs" in file]
        elif self.embed_type == "global_emb":
            filtered_files = [file for file in all_files if "pooled_emb" in file]

        return filtered_files
        # load the embeddings
        embs = []
        for file in filtered_files:
            embs.append(torch.load(os.path.join(path, file), weights_only=True))
        embs = torch.stack(embs)

        return embs

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        path, files = self.vit_embeddings[sample["uid"]]
        idx = random.randint(0, len(files))
        file = files[idx]        
        vit_emb = torch.load(os.path.join(path, file), weights_only=True)
        sample["vit_emb"] = vit_emb[idx]

        # load the ViT Embedding
        # vit_emb = self.vit_embeddings[sample["uid"]]
        # num_embedding = vit_emb.shape[0]
        # idx = torch.randint(0, num_embedding, (1,)).item()
        # sample["vit_emb"] = vit_emb[idx]
        
        return sample


def get_dataloaders(
    path: str,
    batch_size: int = 32,
    sample_size: int = 2048,
    num_workers: int = 4,
    categories: list[str] | None = None,
    transforms=[]
) -> tuple[DataLoader, DataLoader]:
    
    pc_path = path + '/pointclouds'
    emb_path = path + '/embed_renders'
    
    train_dataset = ShapeNet15kPointCloudsViTEmbs(
        pc_path, emb_path,
        categories= categories, 
        split='train',
        tr_sample_size=sample_size,
        te_sample_size=sample_size, 
        random_subsample=True,
        transforms=transforms,
    )
    val_dataset = ShapeNet15kPointCloudsViTEmbs(
        pc_path, emb_path,
        categories= categories, 
        split='val',
        tr_sample_size=sample_size,
        te_sample_size=sample_size, 
        random_subsample=True,
        transforms=transforms,
    )
    
    
    train_dataset.set_voxel_size(1e-5)
    train_dataset.set_noise_params(beta_max=0.02)

    val_dataset.set_voxel_size(1e-5)
    val_dataset.set_noise_params(beta_max=0.02)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=sparse_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sparse_collate_fn,
    )

    return train_loader, val_loader


# TODO:
# [✓] Create the ShapeNet15kPointClouds dataset
# [ ] Create the image version of the dataset
# [ ] Clean the codespace
# [ ] Pass Voxelixation and DDPM noisification as transforms and modify the dataset to
#     accept transforms (CODING PRACTICE!)
# [ ] (BONUS) Release this dataset on the SPVD_Lightning repo for better readability
#
