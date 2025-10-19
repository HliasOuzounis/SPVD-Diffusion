import torch
import random 
import numpy as np
from models.sparse_utils import sparse_quantize_torch
from torchsparse import SparseTensor

class DDPMNoisify:

    def __init__(self, beta_min, beta_max, n_steps, mode='linear'):
        self.n_steps, self.beta_min, self.beta_max = n_steps, beta_min, beta_max
        
        if mode == 'linear':
            self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
            self.alpha = 1. - self.beta 
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        elif mode == "warm0.1":
            self.beta = beta_max * torch.ones(n_steps, dtype=torch.float)
            warmup_time = int(0.1 * n_steps)
            self.beta[:warmup_time] = torch.linspace(beta_min, beta_max, warmup_time, dtype=torch.float)
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def __call__(self, sample):
        
        # load the training points from the sample dictionary
        x0 = sample['train_points']

        # select random timestep
        t = random.randint(0, self.n_steps-1)

        # random noise
        noise = torch.randn(x0.shape)

        # interpolate noise
        alpha_hat_t = self.alpha_hat[t]
                
        xt = torch.sqrt(alpha_hat_t)*x0 \
                + torch.sqrt(1-alpha_hat_t)*noise


        # update the sample dictionary with the new noisy points
        sample['xt'] = xt
        sample['t'] = t
        sample['noise'] = noise

        return sample
    
class TorchSparseVoxelize:

    def __init__(self, voxel_size=1e-5):
        self.voxel_size = voxel_size
    
    def __call__(self, sample):

        pts, t, noise = sample['xt'], sample['t'], sample['noise']

        coords = pts - torch.min(pts, dim=0, keepdim=True).values
        coords, indices = sparse_quantize_torch(coords, self.voxel_size, return_index=True)

        coords = coords.int()
        feats = pts[indices].float()
        noise = noise[indices].float()

        noisy_pts = SparseTensor(coords=coords, feats=feats)
        noise = SparseTensor(coords=coords, feats=noise)
        
        sample['xt'] = noisy_pts
        sample['noise'] = noise
        sample['t'] = torch.tensor(t)

        return sample

class UnitSphereNormalize:
    def __call__(self, sample):
        if isinstance(sample, dict):
            points = sample['train_points']
        else:
            points = sample

        # Center the points
        center = torch.mean(points, dim=0)
        centered_points = points - center

        # Normalize to unit sphere
        scale = torch.max(torch.norm(centered_points, dim=1))
        normalized_points = centered_points / scale

        if isinstance(sample, dict):
            sample['train_points'] = normalized_points
            sample['center'] = center
            sample['scale'] = scale
            return sample
        else:
            return normalized_points

class DropPointsByLabel:
    """
    Randomly drops labels from a set of points. The `min_keep_labels` specifies the minimum
    number of labels that should remain. 
    If there is only one label, half of the points will be dropped. 
    Otherwise, if the total number of labels is less than or equal to `min_keep_labels`,
    exactly one label is dropped. 
    In all other cases, the number of dropped labels is chosen randomly from the range
    [1, max_labels - min_keep_labels].
    """

    def __init__(self, min_keep_labels=1):
        self.min_keep_labels = min_keep_labels

    def __call__(self, sample):
        if isinstance(sample, dict):
            points = sample['train_points']
            labels = sample['train_labels']
        else:
            points, labels = sample

        unique_labels = torch.unique(labels)
        max_labels = len(unique_labels)

        # Special case: only one label, drop half of the points
        if max_labels == 1:
            num_points = points.shape[0]
            keep_count = num_points // 2
            keep_indices = torch.randperm(num_points, device=points.device)[:keep_count]
            points = points[keep_indices]
            labels = labels[keep_indices]
        else:
            # Determine how many labels to drop
            if max_labels <= self.min_keep_labels:
                drop_count = 1
            else:
                drop_count = torch.randint(
                    1, max_labels - self.min_keep_labels + 1, (1,), device=points.device
                ).item()

            # Randomly select which labels to drop
            perm = torch.randperm(max_labels, device=points.device)
            drop_labels = unique_labels[perm][:drop_count]

            # Compute a mask of labels to keep
            keep_mask = ~torch.isin(labels, drop_labels)

            # Keep only the points/labels we want
            points = points[keep_mask]
            labels = labels[keep_mask]

        # Return data in the same format as received
        if isinstance(sample, dict):
            sample['train_points'] = points
            sample['train_labels'] = labels
            return sample
        else:
            return points, labels

class PadTrainPoints:
    """
    Pads the 'train_points' and 'train_labels' tensors in the sample to a specified number of points.
    If the number of points is less than 'num_points', it pads with zeros for points and -1 for labels.
    """

    def __init__(self, num_points=2048):
        self.num_points = num_points

    def __call__(self, sample):
        train_points = sample['train_points']
        train_labels = sample['train_labels']
        num_points = train_points.shape[0]

        if num_points < self.num_points:
            # Pad points with zeros
            pad_points = train_points.new_zeros((self.num_points - num_points, train_points.shape[1]))
            train_points = torch.cat([train_points, pad_points], dim=0)

            # Pad labels with -1
            pad_labels = train_labels.new_full((self.num_points - num_points,), -1)
            train_labels = torch.cat([train_labels, pad_labels], dim=0)

        sample['train_points'] = train_points
        sample['train_labels'] = train_labels

        return sample

class GetPoints:

    '''
    Gets the point information from 
    
    '''
    def __call__(self, sample):
        return sample['train_points']