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