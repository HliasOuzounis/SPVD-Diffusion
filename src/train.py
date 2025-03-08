from datasets.modelnet40.modelnet40_loader import get_dataloaders, ModelNet40
from my_models.spvd import SPVUnet
import models

import torch
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

def main():
    path = "../data/ModelNet40"
    categories = ['bottle']
    tr, te = get_dataloaders(path, categories=categories)
    
    
    down_blocks = [{
        "features_list": [32, 64, 128],
        "num_layers_list": 1,
        "attn_heads": (None, None)
    }]
    up_blocks = [
        # {
        #     "features_list": [256, 192, 192],
        #     "num_layers_list": 2,
        #     "attn_heads": 8
        # },
        {
            "features_list": [128, 64, 32],
            "num_layers_list": 2,
            "attn_heads": None
        },
    ]
    t_emb_features = 64

    model = SPVUnet(down_blocks, up_blocks, t_emb_features)

    import models

    lr = 1e-4
    model = models.DiffusionBase(model, lr=lr)

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='../checkpoints/ModelNet/testing',
        filename='{epoch}-{val_loss:.2f}',
        save_last=True  # Save the last checkpoint
    )
    trainer = L.Trainer(
        max_epochs=40,
        gradient_clip_val=10.0,
        callbacks=[checkpoint_callback]
    )
    # trainer.fit(model=model, train_dataloaders=tr, val_dataloaders=te)
    
    from utils.schedulers import DDPMSparseSchedulerGPU

    ddpm_sched = DDPMSparseSchedulerGPU(n_steps=1000, beta_min=0.0001, beta_max=0.02, pres=1e-5)

    preds = ddpm_sched.sample(model.cuda().eval(), 16, 2048)

    # ----------------------------
    # model2 = models.SPVD() 
    # model2 = models.DiffusionBase(model2, lr=lr)

    # trainer.fit(model=model2, train_dataloaders=tr, val_dataloaders=te)

    # preds = ddpm_sched.sample(model2.cuda().eval(), 16, 2048)


    from utils.visualization import visualize_notebook
    visualize_notebook(preds, x_offset=2.5, y_offset=2.5, point_size=0.025)

if __name__ == "__main__":
    main()

