import torch
from abc import ABC, abstractmethod


#############################################################
###                   Scheduling Method                   ###
#############################################################

class SchedulingMethod(ABC):
    """
    Abstract base class for beta scheduling methods.
    """

    @abstractmethod
    def compute_schedule(self, strategy):
        pass


class LinearScheduling(SchedulingMethod):
    """
    Linear beta scheduling method.
    """

    def compute_schedule(self, strategy):
        inds = torch.arange(strategy.n_steps)
        beta = torch.linspace(strategy.beta_min, strategy.beta_max, strategy.n_steps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        return inds, beta, alpha, alpha_hat


class WarmupScheduling(SchedulingMethod):
    """
    Warmup beta scheduling method.
    """

    def compute_schedule(self, strategy):
        inds = torch.arange(strategy.n_steps)
        beta = strategy.beta_max * torch.ones(strategy.n_steps, dtype=torch.float)
        warmup_time = int(0.1 * strategy.n_steps) #TODO: warmup percentage could be set as a parameter
        beta[:warmup_time] = torch.linspace(
            strategy.beta_min, strategy.beta_max, warmup_time, dtype=torch.float
        )
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        return inds, beta, alpha, alpha_hat
    
#############################################################
###                  Scheduling Strategy                  ###
#############################################################

class SchedulingStrategy(ABC):
    """
    Abstract base class for scheduling strategies.
    """

    @abstractmethod
    def update_rule(self, x_t, noise_pred, t, i, shape, device):
        pass

    @property
    @abstractmethod
    def steps(self):
        pass

class DDPMBase(SchedulingStrategy):
    '''
    This is a Base method for all conventional schedulers like DDPM and DDIM that interpolate between a beta_min and beta_max value
    '''
    def __init__(
        self,
        beta_min=0.0001,
        beta_max=0.02,
        n_steps=1000,
        scheduling_method=LinearScheduling() ,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_steps = n_steps
        self.scheduling_method = scheduling_method

        self.inds, self.beta, self.alpha, self.alpha_hat = self.scheduling_method.compute_schedule(
            self
        )
        self.inds = list(reversed(self.inds))

    @property
    def steps(self):
        return self.inds

    @abstractmethod
    def update_rule(self, x_t, noise_pred, t, i, shape, device):
        pass

class DDPM(DDPMBase):
    """
    DDPM scheduling strategy.
    """

    def __init__(
        self,
        beta_min=0.0001,
        beta_max=0.02,
        n_steps=1000,
        scheduling_method=None,
        sigma_type="beta",
    ):
        super().__init__(beta_min, beta_max, n_steps, scheduling_method)

        assert sigma_type in ["beta", "coef_beta"], f"Invalid sigma type: {sigma_type}"
        if sigma_type == "beta":
            self.sigma = torch.sqrt(self.beta)
        else:
            alpha_hat_prev = torch.cat([torch.tensor([1.0]), self.alpha_hat[:-1]])
            self.sigma = torch.sqrt(
                self.beta * (1 - alpha_hat_prev) / (1 - self.alpha_hat)
            )

    def update_rule(self, x_t, noise_pred, t, i, shape, device):
        # create noise from Gaussian destribution, with the same shape as x_t
        z = (
            torch.randn_like(x_t).to(device) if t > 0 else torch.zeros_like(x_t).to(device)
        )
        
        # get the parameters for the current timestep
        a_t, ahat_t, s_t = self.alpha[t], self.alpha_hat[t], self.sigma[t]

        x_t = (
            (1 / torch.sqrt(a_t))
            * (x_t - ((1 - a_t) / torch.sqrt(1 - ahat_t)) * noise_pred)
            + s_t * z
        )
        return x_t

class DDIM(DDPMBase):
    """
    DDIM scheduling strategy.
    """

    def __init__(
        self,
        beta_min=0.0001,
        beta_max=0.02,
        n_steps=1000,
        scheduling_method=None,
        s_steps=100,
    ):
        super().__init__(beta_min, beta_max, n_steps, scheduling_method)
        self.s_steps = s_steps
        self.inds = torch.floor(torch.linspace(0, n_steps - 1, s_steps + 1)).long()
        self.prev_inds = list(reversed(self.inds[:-1]))
        self.inds = list(reversed(self.inds[1:]))

    def update_rule(self, x_t, noise_pred, t, i, shape, device):
        t_prev = self.prev_inds[i]
        ahat_t = self.alpha_hat[t]
        ahat_t_prev = self.alpha_hat[t_prev]

        x_t = (
            torch.sqrt(ahat_t_prev)
            * ((x_t - torch.sqrt(1 - ahat_t) * noise_pred) / torch.sqrt(ahat_t))
            + torch.sqrt(1 - ahat_t_prev) * noise_pred
        )
        return x_t