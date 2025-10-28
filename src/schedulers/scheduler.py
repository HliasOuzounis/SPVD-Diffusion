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
        self.init_steps = init_steps

        if mode == 'linear':
            self.beta = torch.linspace(beta_min, beta_max, init_steps)
        elif mode == "warm0.1":
            self.beta = beta_max * torch.ones(init_steps)
            warmup_time = int(0.1 * init_steps)
            self.beta[:warmup_time] = torch.linspace(
                beta_min, beta_max, warmup_time
            )
        else:
            raise NotImplementedError(f"Scheduler mode {mode} not implemented") 

        self.step_size = 1
        self.t_steps = list(reversed(range(init_steps)))

        while steps != len(self.t_steps):
            if steps > len(self.t_steps):
                raise ValueError("Can't reach the desired number of steps by halving the starting steps")
            self.t_steps = self.t_steps[::2]
            self.step_size *= 2
            
        self.t_steps = torch.tensor(self.t_steps, dtype=torch.int64)

    def sample(self, model, num_samples: int, num_points: int, num_features: int = 3, starting_noise=None, reference=None, stochastic=True, device='cuda', save=False, guidance_scale=1):
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
    
        logs = []
        with torch.no_grad():
            for t in tqdm(self.t_steps, desc="Sampling", leave=False):
                x_t = self.sample_step(model, x_t, t, shape, device, reference=reference, stochastic=stochastic, save=save, guidance_scale=guidance_scale)
                if save:
                    x_t, x0 = x_t
                    logs.append(x0.cpu())
        
        x_t = self.post_process(x_t)
        
        return x_t.reshape(shape) if not save else (x_t.reshape(shape), logs)

    def sample_step(self, model, x, t, shape, device, reference=None, stochastic=True, save=False, guidance_scale=1):
        if isinstance(t, int) or (t.numel() == 1):
            t_batch = torch.full((shape[0],), t, device=device)
        else:
            t_batch = t

        t_batch = torch.clamp(t_batch, min=0)

        noise_prediction = model((x, t_batch, reference))
        
        if reference is not None and guidance_scale != 1:
            noise_prediction_unguided = model((x, t_batch, None))
            noise_prediction = noise_prediction_unguided + guidance_scale * (noise_prediction - noise_prediction_unguided)

        new_x = self.update(x, t_batch, noise_prediction, shape, stochastic=stochastic, save=save)
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