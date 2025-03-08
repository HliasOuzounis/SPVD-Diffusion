from datasets.modelnet40.modelnet40_loader import get_dataloaders, ModelNet40
from my_models.spvd import SPVUnet
import models

import torch
torch.cuda.empty_cache()

def main():
    path = "../data/ModelNet40"
    categories = ['bottle']
    # tr, te = get_dataloaders(path, categories=categories)
    
    down_blocks = [{
        "features_list": [32, 64, 128, 192, 192, 256],
        "num_layers_list": 1,
        "attn_heads": (None, None, None, 8, 8)
    }]
    up_blocks = [
        {
            "features_list": [256, 192, 192],
            "num_layers_list": 2,
            "attn_heads": 8
        },
        {
            "features_list": [192, 128, 64, 32],
            "num_layers_list": 2,
            "attn_heads": None
        },
    ]
    t_emb_features = 64

    model = SPVUnet(down_blocks, up_blocks, t_emb_features)
    model = models.DiffusionBase(model, lr=1e-4)

    weights_path = '../checkpoints/ModelNet/testing/unsafe.ckpt'
    state_dict = torch.load(weights_path, weights_only=True)['state_dict']
    model.load_state_dict(state_dict)
    
    from utils.schedulers import DDPMSparseSchedulerGPU

    ddpm_sched = DDPMSparseSchedulerGPU(n_steps=1000, beta_min=0.0001, beta_max=0.02, pres=1e-5)

    preds = ddpm_sched.sample(model.cuda().eval(), 16, 2048)

if __name__ == '__main__':
    main()
