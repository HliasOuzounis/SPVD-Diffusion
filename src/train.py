from datasets.modelnet40.modelnet40_loader import get_dataloaders, ModelNet40
from my_models.spvd import SPVUnet
import models

import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

def train(model, training_set, test_set, epochs=40, checkpoint_path=None, lr=1e-4):
    model = models.DiffusionBase(model, lr=lr)

    if checkpoint_path is not None:
        checkpoint_path, file_name = os.path.split(checkpoint_path)
        file_name = os.path.splitext(file_name)[0]
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename=file_name,
            save_last=True
        )
        
    trainer = L.Trainer(
        max_epochs=epochs,
        gradient_clip_val=10.0,
        callbacks=[] if checkpoint_path is None else [checkpoint_callback]
    )

    trainer.fit(model=model, train_dataloaders=training_set, val_dataloaders=test_set)

    return model

def main():
    path = "../data/ModelNet40"
    categories = ['bottle']
    tr, te = get_dataloaders(path, categories=categories)

    down_blocks = [{
        "features_list": [32, 32, 192, 256],
        "num_layers_list": 1,
        "attn_heads": None
    }]
    up_blocks = [
        {
            "features_list": [256, 192, 32, 32],
            "num_layers_list": 2,
            "attn_heads": None
        },
    ]
    t_emb_features = 64

    model = SPVUnet(down_blocks, up_blocks, t_emb_features)

    lr = 1e-4
    model = models.DiffusionBase(model, lr=lr)

    checkpoint_path = '../checkpoints/ModelNet/testing/{epoch}-{val_loss:.2f}'
    train(model, checkpoint_path, tr, te)