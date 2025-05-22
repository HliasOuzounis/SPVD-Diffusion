import torch

def process_ckpt(ckpt):
    if "state_dict" not in ckpt:
        return ckpt

    new_ckpt = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("teacher."):
            continue
        if k.startswith("student."):
            k = k.replace("student.", "")
            new_ckpt[k] = v
        if k.startswith("model."):
            new_ckpt[k] = v
    
    return new_ckpt
    
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

def standardize(bacthed_poitns: torch.Tensor) -> torch.Tensor:
    """
    Standardize a batched tensor of 3D points.
    
    Args:
        batched_points: (B, N, 3) tensor, where B = batch size, N = num points.
    
    Returns:
        (B, N, 3) tensor, where all points are standardized.
    """
    # Compute mean and std for each batch
    mean = torch.mean(bacthed_poitns, dim=1, keepdim=True)  # (B, 1, 3)
    std = torch.std(bacthed_poitns, dim=1, keepdim=True)  # (B, 1, 3)

    # Standardize
    standardized = (bacthed_poitns - mean) / (std + 1e-8)  # Avoid division by zero

    return standardized

def normalize_to_unit_cube(batched_points: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batched tensor of 3D points to the unit cube.
    
    Args:
        batched_points: (B, N, 3) tensor, where B = batch size, N = num points.
    
    Returns:
        (B, N, 3) tensor, where all points lie within the unit cube.
    """
    # Find min and max for each batch
    min_vals = torch.min(batched_points, dim=1, keepdim=True).values  # (B, 1, 3)
    max_vals = torch.max(batched_points, dim=1, keepdim=True).values  # (B, 1, 3)

    # Compute range
    ranges = max_vals - min_vals + 1e-8  # Avoid division by zero

    # Normalize to [-1, 1]
    normalized = (batched_points - min_vals) / ranges  # (B, N, 3)

    return normalized