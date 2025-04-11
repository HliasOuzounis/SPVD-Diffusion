import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from abc import ABC, abstractmethod

class Task(ABC):
    @abstractmethod
    def prep_data(self, batch):
        pass
    @abstractmethod
    def loss_fn(self, pred, target):
        pass

class SparseGeneration(Task):
    def prep_data(self, batch):
        noisy_data, t, noise = batch['input'], batch['t'], batch['noise']
        render_features = batch['render-features']
        inp = (noisy_data, t)
        return inp, noise.F, render_features
    
    def loss_fn(self, preds, target, snr):
        # return F.mse_loss(preds, target)
        snr = snr.view(-1, 1, 1)
        batch = snr.shape[0]
        preds = preds.view(batch, -1, 3)
        target = target.view(batch, -1, 3)
        return (snr * (preds - target).pow(2)).mean()

class DiffusionBase(L.LightningModule):

    def __init__(self, model, task=SparseGeneration(), lr=0.0002):
        super().__init__()
        self.model = model
        # self.student = model
        self.task = task
        self.learning_rate = lr
        
    def set_noise_scheduler(self, noise_scheduler):
        self.noise_scheduler = noise_scheduler
        
    def forward(self, x, render_features=None):
        return self.model(x, render_features)
    
    def training_step(self, batch, batch_idx):
        # get data from the batch
        inp, target, render_features = self.task.prep_data(batch)
        x, t = inp

        if torch.rand(1) < 0.1: # Random unconditional training
            render_features = None

        # activate the network for noise prediction
        preds = self(inp, render_features)

        # calculate the loss
        snr = self.noise_scheduler.snr_weight(t).to(preds.device)
        loss = self.task.loss_fn(preds, target, snr)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return loss

    def validation_step(self, batch, batch_idx):
        inp, target, render_features = self.task.prep_data(batch)
        x, t = inp

        preds = self(inp, render_features)

        snr = self.noise_scheduler.snr_weight(t).to(preds.device)
        loss = self.task.loss_fn(preds, target, snr)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))


    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.05)

        # Create a dummy scheduler (we will update `total_steps` later)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=1)

        # Return optimizer and scheduler (scheduler will be updated in `on_fit_start`)
        return [optimizer], [{'scheduler': self.lr_scheduler, 'interval': 'step'}]

    # Setting the OneCycle scheduler correct number of steps at the start of the fit loop, where the dataloaders are available.
    def on_train_start(self):
        # Access the dataloader and calculate total steps
        train_loader = self.trainer.train_dataloader  # Access the dataloader from the trainer
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        # Update the scheduler's `total_steps` dynamically
        self.lr_scheduler.total_steps = total_steps

        # Read the batch size for logging
        self.tr_batch_size = self.trainer.train_dataloader.batch_size

    def on_validation_start(self):
        val_loader = self.trainer.val_dataloaders
        if val_loader:
            self.vl_batch_size = val_loader.batch_size
