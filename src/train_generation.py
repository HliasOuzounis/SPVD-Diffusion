import argparse
import torch
import lightning as L
from models import *
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets.shapenet.shapenet_pointflow_sparse import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="SPVD Training Script")

    # Add arguments
    parser.add_argument('--version', type=str, choices=['S', 'M', 'L'], default='S',
                        help='Model version: S, M, or L (default: S)')
    parser.add_argument('--categories', type=str, nargs='+', default=['airplane'],
                        help='List of categories (default: ["airplane"])')
    parser.add_argument('--ckpt_name', type=str, default='SPVD',
                        help='Checkpoint name (default: SPVD)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--path', type=str, default='/home/ubuntu/ShapeNetPC/',
                        help='Path to the dataset (default: /home/ubuntu/ShapeNetPC/)')
    parser.add_argument('--precision', type=str, choices=['medium', 'high'], default='medium',
                        help='Floating point precision: medium or high (default: medium)')

    return parser.parse_args()

def main():
    args = parse_args()

    # Set floating point precision based on argument
    if args.precision == 'medium':
        torch.set_float32_matmul_precision('medium')
    elif args.precision == 'high':
        torch.set_float32_matmul_precision('high')

    # Load model based on the version argument
    if args.version == 'S':
        m = SPVD_S()
    elif args.version == 'M':
        m = SPVD()
    elif args.version == 'L':
        m = SPVD_L()

    # Initialize model
    model = DiffusionBase(m, lr=args.lr)

    # Get dataloaders
    # tr_dl, te_dl = get_dataloaders(args.path, args.categories)
    
    from datasets.modelnet40.modelnet40_loader import get_dataloaders
    path = "./data/ModelNet40/pointclouds"
    tr_dl, te_dl = get_dataloaders(path)

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/ModelNet40', filename=args.ckpt_name)

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=args.epochs, 
        gradient_clip_val=10.0, 
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model=model, train_dataloaders=tr_dl, val_dataloaders=te_dl)

if __name__ == "__main__":
    main()
