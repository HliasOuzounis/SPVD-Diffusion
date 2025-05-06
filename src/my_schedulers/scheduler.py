from abc import ABC, abstractmethod
import torch
from tqdm.auto import tqdm

class Scheduler(ABC):
    def __init__(self, init_steps: int|None = None, steps: int = 1000, beta_min: float = 0.0001, beta_max: float = 0.02, mode: str = 'linear'):
        """
        Args:
            steps: Number of steps for the scheduler
        """
        
        assert steps > 0, "Number of steps must be positive"
        self.steps = steps

        if init_steps is None:
            init_steps = steps
        
        if mode == 'linear':
            self.beta = torch.linspace(beta_min, beta_max, init_steps)
        elif mode == 'warm0.1':
            warmup_time = int(0.1 * init_steps)
            self.beta = beta_max * torch.ones(init_steps)
            self.beta[:warmup_time] = torch.linspace(beta_min, beta_max, warmup_time)
        else:
            raise NotImplementedError(f"Scheduler mode {mode} not implemented")

        self.t_steps = list(reversed(range(steps))) 

    def sample(self, model, num_samples: int, num_points: int, num_features: int = 3, starting_noise=None, reference=None, stochastic=True, device='cuda', progress_bar=True):
        """
        Args:
            model: The model to sample from
            num_samples: The number of samples to generate
            num_points: The number of points in each sample
            num_features: The number of features in each point
        Returns:
            The generated samples of shape (num_samples, num_points, num_features)
        """
        shape = (num_samples, num_points, num_features)
        x_t = self.create_noise(shape, device) if starting_noise is None else starting_noise

        if reference is not None:
            assert len(reference) == num_samples, "Reference image batch size must match the number of samples"
        
        with torch.no_grad():
            t_steps = tqdm(self.t_steps, desc="Sampling", leave=False) if progress_bar else self.t_steps
            for t in t_steps:
                x_t = self.sample_step(model, x_t, t, shape, device, reference=reference, stochastic=stochastic)
        
        x_t = self.post_process(x_t)
        
        return x_t.reshape(shape)

    def sample_step(self, model, x, t, shape, device, reference=None, stochastic=True):
        if isinstance(t, int) or (t.numel() == 1 and shape[0] != 1):
            t_batch = torch.full((shape[0],), t, device=device)
        else:
            t_batch = t

        t_batch = torch.clamp(t_batch, min=0)

        # noise_prediction = model((x, t_batch), reference)
        noise_prediction = model((x, t_batch, reference))

        new_x = self.update(x, t_batch, noise_prediction, shape, stochastic=stochastic)
        return new_x

    def post_process(self, x):
        # Post-process the generated samples
        # This can include clamping, normalization, etc.
        return x
        
    @abstractmethod
    def update(self, x, t, noise):
        pass

    @abstractmethod
    def add_noise(self, x, t):
        pass

    @abstractmethod
    def create_noise(self, shape, device):
        pass

    @abstractmethod
    def get_params(self, t, bs, device):
        pass