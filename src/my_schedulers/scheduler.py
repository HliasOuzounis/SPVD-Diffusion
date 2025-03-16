from abc import ABC, abstractmethod
import torch
from tqdm import tqdm

class Scheduler(ABC):
    def sample(self, model, num_samples: int, num_points: int, num_features: int = 3):
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
        x_t = self.create_noise(shape, model.device)
        steps = list(reversed(range(self.steps)))
        for t in tqdm(steps, desc="Sampling"):
            x_t = self.sample_step(model, x_t, t, shape, model.device)
        return x_t.reshape(shape)

    @torch.no_grad()
    def sample_step(self, model, x, t, shape, device):
        if isinstance(t, int) or t.numel() == 1:
            t_batch = torch.full((shape[0],), t, device=device)
        else:
            t_batch = t

        noise_prediction = model((x, t_batch))

        return self.update(x, t_batch, noise_prediction, shape)
        
    @abstractmethod
    def update(self, x, t, noise):
        pass

    @abstractmethod
    def denoise(self, x, noise, a_t_cumprod):
        pass

    @abstractmethod
    def add_noise(self, x, t):
        pass

    @abstractmethod
    def create_noise(self, shape, device):
        pass