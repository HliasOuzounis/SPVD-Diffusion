import torch
from abc import ABC, abstractmethod
from .scheduling_strategies import SchedulingStrategy
from .torch2sparse import Torch2Sparse


class SchedulerBase(ABC):
    """
    Abstract base class for schedulers.
    """

    def __init__(self, strategy: SchedulingStrategy, save_process=False):
        self.strategy = strategy
        self.save_process = save_process

    def get_pc(self, x_t, shape):
        return x_t.detach().cpu().reshape(shape)

    @abstractmethod
    def sample(
        self,
        model,
        bs,
        n_points=2048,
        nf=3,
        cond_emb=None,
        uncond_emb=None,
        guidance_scale=1.0,
        mode="guided",
        save_process=False,
    ):
        pass

    @abstractmethod
    def create_noise(self, shape, device):
        pass


class SparseScheduler(SchedulerBase):
    """
    Scheduler for sparse tensors with classifier-free guidance.
    """

    def __init__(
            self, 
            strategy: SchedulingStrategy, 
            torch2sparse: Torch2Sparse, 
            pres = 1e-5, 
            save_process=False
    ):
        super().__init__(strategy, save_process)
        self.pres = pres
        self.torch2sparse = torch2sparse
    
    def create_noise(self, shape, device):
        noise = torch.randn(shape, device=device)
        return self.torch2sparse(noise, shape, self.pres)
    
    def get_pc(self, x_t, shape):
        return x_t.F.detach().cpu().reshape(shape) # TODO: move this to the torch2sparse - handle different sparse libraries in the future.
    
    def update_rule(self, x_t, noise_pred, t, i, shape, device):
        x_t = x_t.F  # TODO: In the torch2sparse add a function `get_features` that handles this
        x_t = self.strategy.update_rule(x_t, noise_pred, t, i, shape, device)
        return self.torch2sparse(x_t, shape, self.pres)
    
    @torch.no_grad()
    def sample(
        self, 
        model, 
        bs, 
        n_points=2048, 
        nf=3, 
        cond_emb=None, 
        uncond_emb=None, 
        guidance_scale=1.0, 
        mode='guided',
        save_process=None
    ):
        """
        Generates samples using the diffusion model with optional guidance.

        Args:
            model: The neural network model for noise prediction.
            bs: Batch size.
            n_points: Number of points per point cloud.
            nf: Number of features (default is 3 for XYZ coordinates).
            cond_emb: Conditional embedding (tensor).
            uncond_emb: Unconditional embedding (tensor).
            guidance_scale: Guidance scale factor.
            mode: 'guided', 'conditional', or 'unconditional'.
            save_process: Whether to save intermediate point clouds, if set to None, 
                          it will use the value set at the __init__.

        Returns:
            Generated point clouds.
        """
        device = next(model.parameters()).device
        shape = (bs, n_points, nf)
        x_t = self.create_noise(shape, device)
        if save_process is None: 
            save_process = self.save_process
        preds = [self.get_pc(x_t, shape)] if save_process else None

        for i, t in enumerate(self.strategy.steps):
            x_t = self.sample_step(
                model, 
                x_t, 
                t, 
                i, 
                cond_emb, 
                uncond_emb, 
                shape, 
                device, 
                guidance_scale,
                mode
            )
            if save_process:
                preds.append(self.get_pc(x_t, shape))

        return preds if save_process else self.get_pc(x_t, shape)
    
    def sample_step(
            self,
            model, 
            x_t, 
            t, 
            i, 
            cond_emb, 
            uncond_emb, 
            shape, 
            device, 
            guidance_scale, 
            mode
    ):
        bs = shape[0]
        
        # create the time embedding
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        if mode == "unconditional":
            # Unconditional generation
            noise_pred = model((x_t, t_batch, uncond_emb))
        elif mode == "conditional":
            # Fully conditional generation
            noise_pred = model((x_t, t_batch, cond_emb))
        elif mode == "guided":
            # Classifier-free guidance
            # Conditional prediction
            noise_pred_cond = model((x_t, t_batch, cond_emb))
            # Unconditional prediction
            noise_pred_uncond = model((x_t, t_batch, uncond_emb))
            # Combine predictions
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Update x_t values
        x_t = self.update_rule(x_t, noise_pred, t, i, shape, device)
        
        return x_t
    

class DualModelSparseScheduler(SparseScheduler):
    """
    This scheduler is similar to the SparseScheduler but uses different models 
    for handling conditional and unconditional noise predictions.
    """

    @torch.no_grad()
    def sample(self, 
               cond_model, 
               uncond_model, 
               bs, 
               n_points=2048, 
               nf=3, 
               cond_emb=None, 
               uncond_emb=None, 
               guidance_scale=1, 
               mode='guided', 
               save_process=None):

        cond_model_device = next(cond_model.parameters()).device
        uncond_model_device = next(uncond_model.parameters()).device
        # models should be on the same device for this version
        assert cond_model_device == uncond_model_device, "Models should be on the same device"
        device = cond_model_device

        shape = (bs, n_points, nf)
        x_t = self.create_noise(shape, device)
        if save_process is None: 
            save_process = self.save_process
        preds = [self.get_pc(x_t, shape)] if save_process else None

        for i, t in enumerate(self.strategy.steps):
            x_t = self.sample_step(
                cond_model, 
                uncond_model, 
                x_t, 
                t, 
                i, 
                cond_emb, 
                uncond_emb, 
                shape, 
                device, 
                guidance_scale,
                mode
            )

    def sample_step(
            self,
            cond_model,
            uncond_model, 
            x_t, 
            t, 
            i, 
            cond_emb, 
            uncond_emb, 
            shape, 
            device, 
            guidance_scale, 
            mode
    ):
        bs = shape[0]

        # create the time embedding
        t_batch = torch.full((bs,), t, device=device, dtype=torch.long)

        if mode == "unconditional":
            # Unconditional generation
            noise_pred = uncond_model((x_t, t_batch))
        elif mode == "conditional":
            # Fully conditional generation
            noise_pred = cond_model((x_t, t_batch, cond_emb))
        elif mode == "guided":
            # Classifier-free guidance
            # Conditional prediction
            noise_pred_cond = cond_model((x_t, t_batch, cond_emb))
            # Unconditional prediction
            noise_pred_uncond = uncond_model((x_t, t_batch))
            # Combine predictions
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # Update x_t values
        x_t = self.update_rule(x_t, noise_pred, t, i, shape, device)

        return x_t

