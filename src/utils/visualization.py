import k3d
import numpy as np
import matplotlib.pyplot as plt

def display_pointclouds_grid(pointclouds, offset=8.0, point_size=0.2, grid_dims=None):
    """
    Display multiple point clouds in a grid layout using k3d.
    
    Args:
        *pointclouds: Variable number of point clouds (each as numpy array of shape [N,3])
        offset: Distance between point clouds (default: 1.0)
        point_size: Size of points in visualization (default: 0.1)
        grid_dims: Optional (rows, cols) for grid layout. Auto-calculated if None.
    """
    if len(pointclouds) == 0:
        raise ValueError("At least one point cloud must be provided")
    
    # Calculate grid dimensions if not provided
    if grid_dims is None:
        grid_cols = int(np.ceil(np.sqrt(len(pointclouds))))
        grid_rows = int(np.ceil(len(pointclouds) / grid_cols))
    else:
        grid_rows, grid_cols = grid_dims

    
    # Create plot
    plot = k3d.plot()

    stacked_points = []
    for idx, points in enumerate(pointclouds):
        if points.shape[1] != 3:
            raise ValueError(f"Point cloud {idx} must be shape [N,3], got {points.shape}")
        
        # Calculate grid position
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Apply offset
        offset_vec = np.array([-col * offset, 0, row * offset])
        offset_points = points + offset_vec
        
        # Create k3d object
        point_cloud = k3d.points(
            positions=offset_points.astype(np.float32),
            # color=color,
            point_size=point_size
        )
        plot += point_cloud

        stacked_points.append(offset_points)
    
    camera_target = np.mean(np.vstack(stacked_points), axis=0)
    camera_pos = camera_target - np.array([-3 * offset, -2 * offset, +3 * offset])
    
    # Adjust camera to see all point clouds
    plot.grid_visible = False
    plot.camera_auto_fit=False
    plot.camera = np.hstack([
            camera_pos,    # Camera position [x,y,z]
            camera_target,     # Target position [x,y,z]
            [0, 1, 0]         # Up vector
        ]).tolist()
    
    plot.display()


def display_renders_grid(images, grid_dims=None):
    """
    Plot a grid of images using matplotlib.

    Args:
        images: A batch of images (numpy array of numpy arrays) in (N, H, W, C) format.
        grid_dims: Tuple (rows, cols) specifying the grid dimensions.
        figsize: Tuple specifying the figure size.
    """
    if len(images) == 0:
        raise ValueError("At least one image must be provided")

    if grid_dims is None:
        grid_cols = int(np.ceil(np.sqrt(len(images))))
        grid_rows = int(np.ceil(len(images) / grid_cols))
    else:
        grid_rows, grid_cols = grid_dims
    
    fig, axes = plt.subplots(grid_cols, grid_rows, figsize=(12, 12))
    images = iter(images)

    for i in range(grid_cols):
        for j in range(grid_rows):
            img = next(images)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()