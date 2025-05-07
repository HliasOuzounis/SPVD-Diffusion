
from .scheduling_strategies import DDPM, DDIM, LinearScheduling, WarmupScheduling
from .torch2sparse import Torch2TorchsparseCPU, Torch2TorchsparseGPU
from .schedulers import SparseScheduler, DualModelSparseScheduler

def create_sparse_scheduler(
        scheduler_strategy='DDPM',
        scheduler_type='Sparse',
        device_type='gpu',
        beta_min=0.0001, 
        beta_max=0.02,
        n_steps=1000,
        pres=1e-5,
        scheduling_method='linear',
        save_process=False,
        sparse_backend = 'torchsparse', # currently the only one supported
        **kwargs
):
    """
    Factory function to create a scheduler.

    Args:
        scheduler_type: Type of scheduler ('DDPM' or 'DDIM').
        device_type: Device type ('cpu' or 'gpu').
        beta_min: Minimum beta value.
        beta_max: Maximum beta value.
        n_steps: Number of steps in the diffusion process.
        pres: Precision for sparse quantization.
        scheduling_method: The scheduler method for the `beta` values ('linear', 'warmup')
        save_process: Whether to save intermediate results.
        **kwargs: Additional keyword arguments for the scheduling strategy.

    Returns:
        An instance of SparseScheduler.
    """
    
    # Select the scheduling method
    if scheduling_method == 'linear':
        scheduling_method = LinearScheduling()
    elif scheduling_method == 'warmup':
        scheduling_method = WarmupScheduling()
    else:
        raise ValueError(f'Unknown scheduling method: {scheduling_method}')

    # Select the scheduling strategy (DDPM, DDIM, etc)
    if scheduler_strategy == "DDPM":
        strategy = DDPM(
            beta_min=beta_min,
            beta_max=beta_max,
            n_steps=n_steps,
            scheduling_method=scheduling_method,
            **kwargs,
        )
    elif scheduler_strategy == "DDIM":
        strategy = DDIM(
            beta_min=beta_min,
            beta_max=beta_max,
            n_steps=n_steps,
            scheduling_method=scheduling_method,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Select the appropriate Torch2Sparse method
    if sparse_backend == 'torchsparse':
        if device_type == 'cpu':
            torch2sparse = Torch2TorchsparseCPU()
        elif device_type == 'gpu':
            torch2sparse = Torch2TorchsparseGPU()
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    else: 
        raise ValueError(f"Unknown sparse backend: {sparse_backend}")
    
    # Create the scheduler
    if scheduler_type == 'Sparse':
        # Create the scheduler
        scheduler = SparseScheduler(
            strategy=strategy, 
            torch2sparse=torch2sparse, 
            pres=pres, 
            save_process=save_process
        )
    elif scheduler_type == 'DualModelSparse':
        scheduler = DualModelSparseScheduler(
            strategy=strategy, 
            torch2sparse=torch2sparse, 
            pres=pres, 
            save_process=save_process
        )
    else: 
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler