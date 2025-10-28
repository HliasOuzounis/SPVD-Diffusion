from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch.nn as nn
import torch

"""
Module for computing point cloud metrics such as Chamfer Distance and Earth Mover's Distance (EMD).
Portions of this module were adapted from the RGB2Point repository:
https://github.com/JaeLee18/RGB2point
"""

def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
    first point cloud
    y: numpy array [n_points_y, n_dims]
    second point cloud
    metric: string or callable, default ‘l2’
    metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
    direction of Chamfer distance.
    'y_to_x': computes average minimal distance from every point in y to x
    'x_to_y': computes average minimal distance from every point in x to y
    'bi': compute both
    Returns
    -------
    chamfer_dist: float
    computed bidirectional Chamfer distance:
    sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, pred, target):
        # pred and target are expected to have shape (batch_size, 1024, 3)
        assert pred.shape == target.shape
        assert pred.shape[1] == 1024 and pred.shape[2] == 3

        batch_size = pred.shape[0]
        num_points = pred.shape[1]

        # Compute pairwise distances between all points
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        dist = torch.sum(diff**2, dim=-1)

        # Solve the assignment problem using Hungarian algorithm
        # Note: This is a simplified version and may not be the most efficient for large point clouds
        assignment = torch.zeros_like(dist)
        for b in range(batch_size):
            _, indices = torch.topk(dist[b], k=num_points, largest=False, dim=1)
            assignment[b] = torch.scatter(assignment[b], 1, indices, 1)

        # Compute the EMD
        emd = torch.sum(dist * assignment, dim=[1, 2]) / num_points

        return emd.mean()