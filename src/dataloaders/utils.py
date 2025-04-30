import torch
import random 
import numpy as np

from transformers import ViTImageProcessor, ViTModel


class NoiseSchedulerDDPM:

    def __init__(self, beta_min, beta_max, n_steps, mode='linear'):
        self.n_steps, self.beta_min, self.beta_max = n_steps, beta_min, beta_max
        
        if mode == 'linear':
            self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
            self.α = 1. - self.beta 
            self.ᾱ = torch.cumprod(self.α, dim=0)
            #self.σ = self.beta.sqrt()
        elif mode == "warm0.1":
            self.beta = beta_max * torch.ones(n_steps, dtype=torch.float)
            warmup_time = int(0.1 * n_steps)
            self.beta[:warmup_time] = torch.linspace(beta_min, beta_max, warmup_time, dtype=torch.float)
            self.α = 1. - self.beta
            self.ᾱ = torch.cumprod(self.α, dim=0)
            #self.σ = self.beta.sqrt()
        
        self.beta = self.beta.numpy()
        self.α = self.α.numpy()
        self.ᾱ = self.ᾱ.numpy() 
        #self.σ = self.σ.numpy()

    def __call__(self, x0):

        # select random timestep
        t = random.randint(0, self.n_steps-1)

        # random noise
        ε = torch.randn(x0.shape)

        # interpolate noise
        ᾱ_t = self.ᾱ[t]
                
        xt = np.sqrt(ᾱ_t)*x0 + np.sqrt(1-ᾱ_t)*ε

        return xt, t, ε

class VisualTransformer:
    model_name = "google/vit-base-patch16-224-in21k"  # Example model
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to("cuda")

    def preprocess(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.to("cuda")

    def __call__(self, x):
        with torch.no_grad():
            outputs = self.model(x)
        return outputs.last_hidden_state.cpu()

if __name__ == "__main__":
    V = VisualTransformer()
    from PIL import Image
    image_path = "data/test.png"
    image = Image.open(image_path).convert("RGB")
    print(V(image).shape)
