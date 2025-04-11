import torch
import torch.nn as nn
import lightning as L

from .modules import Group, Mask, TransformerWithEmbeddings

from transformers.optimization import get_cosine_schedule_with_warmup
from metrics.chamfer_dist import ChamferDistanceL2

class PointMAE(L.LightningModule):

    def __init__(self, 
                 embed_dim,
                 num_groups, 
                 group_size, 
                 mask_ratio, 
                 mask_type,
                 encoder_depth, 
                 encoder_num_heads, 
                 encoder_drop_path_rate,
                 decoder_depth, 
                 decoder_num_heads, 
                 decoder_drop_path_rate,
                 training_steps = -1 # Required only for training purposes
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.training_steps = training_steps 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.loss_func = ChamferDistanceL2()

        # Setting the networks
        self.group_devider = Group(group_size=group_size, 
                                   num_group=num_groups)
        
        self.mask_generator = Mask(mask_ratio=mask_ratio, 
                                   mask_type=mask_type)

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim=embed_dim, 
            depth=encoder_depth, 
            num_heads=encoder_num_heads, 
            drop_path_rate=encoder_drop_path_rate, 
            feature_embed=True
        )

        self.MAE_decoder = TransformerWithEmbeddings(
            embed_dim=embed_dim, 
            depth=decoder_depth, 
            num_heads=decoder_num_heads, 
            drop_path_rate=decoder_drop_path_rate, 
            feature_embed=False
        )

        self.increase_dim = nn.Conv1d(embed_dim, 3 * group_size, 1)

    def forward(self, pts):
        # The forward pass receives as input a point cloud and returns its latent space representation.
        # pts: shape BxNx3
        neighborhood, center = self.group_devider(pts)
        x_vis = self.MAE_encoder(neighborhood, center)
        return x_vis
    

    def shared_step(self, batch, batch_idx):
        # `shared_step` is the common part of the training and validation steps.
        # -- Description -- 
        # The shared step:
        # 1. Encodes the partial point cloud, 
        # 2. Adds the masked centers and tokens,
        # 3. Activates the decoder to reconstruct the missing areas, 
        # 4. Returns the reconstruction loss 

        # 
        neighborhood, center = self.group_devider(batch)

        #
        mask = self.mask_generator(center)

        # center: B x N x 3
        # neighborhood : B x N x M x 3
        B, _, M, _ = neighborhood.shape
        masked_center = center[~mask].reshape(B, -1, 3)
        masked_neighborhood = neighborhood[~mask].reshape(B, -1, M, 3)

        # Encode the point cloud to a latent space representation 
        x_vis = self.MAE_encoder(masked_neighborhood, masked_center)

        # x_vis: B x P x F
        # where
        #   - B: batch_size
        #   - P: number of patched that we keep
        #   - F: feature dim
        #
        # mask : B x N x F
        # where
        #   - N: total number of patches
        # mask has value 1 to the masked out points

        B, _, C = x_vis.shape

        # stacking centers [encoded centers, not encoded centers]
        vis_centers = center[~mask].reshape(B, -1, 3)
        mask_centers = center[mask].reshape(B, -1, 3)
        pos_full = torch.cat([vis_centers, mask_centers], dim=1)

        # repeating mask token to pass to the decoder
        _, M, _ = mask_centers.shape
        mask_token = self.mask_token.expand(B, M, -1)

        # concatenating actual features with mask features
        # to pass to the decoder
        x_full = torch.cat([x_vis, mask_token], dim=1)

        # activating the decoder
        x_rec = self.MAE_decoder(x_full, pos_full)

        # seperating the decoded features
        x_rec = x_rec[:, -M:, :]

        # passing patch embedding to final mlp to extract the actual point possitions
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        
        # getting the ground truth points
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        # computing the loss
        loss = self.loss_func(rebuild_points, gt_points)

        return loss


    def training_step(self, batch, batch_idx):

        loss = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self.shared_step(batch, batch_idx)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.001, weight_decay=0.05)
        # NOTE: The learning rate scheduler differs slightly from the original implementation.
        # Here, we use `transformers.get_cosine_schedule_with_warmup`, which updates the schedule
        # after each batch rather than after each epoch. Some parameters are also change slightly.
        
        sched = get_cosine_schedule_with_warmup(opt, 
                                               num_warmup_steps=int(0.1 * self.training_steps),
                                               num_training_steps=self.training_steps, 
                                               num_cycles=1.)

        scheduler = {
            'scheduler': sched,
            'interval': 'step'  # Update the scheduler after each step
        }
        return [opt], [scheduler]

    



