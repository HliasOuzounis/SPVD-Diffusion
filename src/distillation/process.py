import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from models.lightningBase import Task

class DistillationTask(Task):
    def prep_data(self, batch):
        noisy_data, t, _ = batch['input'], batch['t'], batch['noise']
        inp = (noisy_data, t)
        return inp, noisy_data
    
    def loss_fn(self, pred, target):
        return F.mse_loss(pred, target)
    

class DistillationProcess(L.LightningModule):
    def __init__(self, task=DistillationTask(), lr=0.0002):
        super().__init__()
        self.task = task
        self.learning_rate = lr

        self.teacher = None
        self.student = None
    
    def set_teacher(self, teacher):
        self.teacher = teacher
        
    def set_student(self, student):
        self.student = student
    
    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        inp, _ = self.task.prep_data(batch)

        with torch.no_grad():
            teacher_preds = self.teacher(inp)
        
        student_preds = self(inp)
        
        loss = self.task.loss_fn(student_preds, teacher_preds)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inp, target = self.task.prep_data(batch)

        with torch.no_grad():
            teacher_preds = self.teacher(inp)
        
        student_preds = self(inp)
        
        loss = self.task.loss_fn(student_preds, teacher_preds)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
    
    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.05)

        # Create a dummy scheduler (we will update `total_steps` later)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=1)

        # Return optimizer and scheduler (scheduler will be updated in `on_fit_start`)
        return [optimizer], [{'scheduler': self.lr_scheduler, 'interval': 'step'}]
    
    def on_train_start(self):
        # Access the dataloader and calculate total steps
        train_loader = self.trainer.train_dataloader  # Access the dataloader from the trainer
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        # Update the scheduler's `total_steps` dynamically
        self.lr_scheduler.total_steps = total_steps

        # Read the batch size for logging
        self.tr_batch_size = self.trainer.train_dataloader.batch_size