import k3d
import numpy as np

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


def plot_renders(batch, grid=(8, 4)):
    import matplotlib.pyplot as plt
    
    print(batch.shape)
    fig, axes = plt.subplots(grid[0], grid[1], figsize=(grid[1] * 2, grid[0] * 2))
    batch = batch.detach().cpu().clone()
    k = 0
    for i in range(grid[0]):
        for j in range(grid[1]):
            if k >= batch.shape[0]:
                axes[i, j].axis('off')
                continue
            
            img = batch[k].permute(1, 2, 0)  # assuming batch is in (N, C, H, W) format
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            k += 1
    
    plt.tight_layout()
    plt.show()