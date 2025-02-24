import k3d
    
def visualize_notebook(batch, grid=(8, 4), x_offset=2.5, y_offset=2.5, point_size=0.2):
    batch = batch.detach().cpu().clone()
    
    assert len(grid) == 2
            
    x_offset_start = - x_offset * grid[0] // 2
    x_offset_start = x_offset_start + x_offset / 2 if grid[0] % 2 == 0 else x_offset_start
    
    y_offset_start = - y_offset * grid[1] // 2
    y_offset_start = y_offset_start + y_offset / 2 if grid[1] % 2 == 0 else y_offset_start
    
    
    plot = k3d.plot(camera_auto_fit=False)
    # pos, target, up, fov
    plot.camera = [0, 0, 12, 0, 0, 0, 0, 1, 0]
    plot.grid_visible = False
    
    k = 0
    for i in range(grid[0]):
        for j in range(grid[1]):
            
            # get point cloud to cpu
            pc = batch[k]
            
            # translate the point cloud properly
            pc[:, 0] += x_offset_start + i * x_offset
            pc[:, 1] += y_offset_start + j * y_offset
            
            # turn in into a k3d point cloud
            
            plot += k3d.points(pc, point_size=point_size)
            
            k += 1
            if k > batch.shape[0] - 1:
                break
        else:
            continue
        break
        
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