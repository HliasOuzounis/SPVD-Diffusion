import lightning as L
import torch
import torch.nn.functional as F
from torchsparse import SparseTensor

class GSPVD(L.LightningModule):

    def __init__(self, model, lr=1e-3, training_steps=-1, uncond_prob=0.1):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.uncond_prob = uncond_prob
        self.training_steps = training_steps

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch):
        # Get the input and target of the batch
        xt, t, cond_emb = (batch['xt'], batch['t'], batch['vit_emb']) # Input
        target = batch['noise'].F if isinstance(batch['noise'], SparseTensor) else batch['noise'] # Target 
        
        B = len(t)

        # Randomly set some embeddings to the unconditional embedding, based on the self.uncond_prob
        embed_mask = torch.rand((B,)) < self.uncond_prob
        cond_emb[embed_mask] = self.model.uncond_embedding

        # activate the network for noise prediction
        inp = (xt, t, cond_emb)
        noise_pred = self(inp)

        # Compute the loss
        loss = F.mse_loss(noise_pred, target)

        return loss

    def training_step(self, batch, batch_idx):
        # shared_step computes the loss
        loss = self.shared_step(batch)
        # Get the batch_size for logging(Usefull for sparse networks)
        B = len(batch['t'])
        # Log the training loss
        self.log('train_loss', loss, batch_size = B, prog_bar=True)
        # Return loss for training
        return loss

    def validation_step(self, batch, batch_idx):
        # shared_step computes the loss
        loss = self.shared_step(batch)
        # Get the batch size for logging
        B = len(batch['t'])
        # Log the validation loss
        self.log('valid_loss', loss, batch_size = B, prog_bar=True)
    
    def configure_optimizers(self):
        # SPVD originally uses Adam optimizer 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=self.training_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def load_checkpoint(self, path):
        loaded_successfully = False
        try:
            self.load_state_dict(torch.load(path)['state_dict'])
            self.eval().cuda()
            print(f"Checkpoint loaded successfully: {path}")
            loaded_successfully = True
        except RuntimeError as e:
            print(f"Error loading checkpoint with default loading method: {e}")

        # Migration from previous version
        if not loaded_successfully:
            try: 
                state_dict = torch.load(path)['state_dict']
                # Rename the keys, so each key whose prefix is 'guidance_model.' should be renamed to 'model.'
                state_dict = {k.replace('guidance_model.', 'model.'): v for k, v in state_dict.items()}
                self.load_state_dict(state_dict)
                self.eval().cuda()
                loaded_successfully = True
            except RuntimeError as e:
                print(f"Error loading checkpoint with migration method: {e}")

        if not loaded_successfully:
            try: 
                state_dict = torch.load(path)['state_dict']
                # Rename the keys, so each key whose prefix is 'guidance_model.' should be renamed to 'model.'
                state_dict = {k.replace('diffusion_model.', 'model.'): v for k, v in state_dict.items()}
                print(state_dict)
                self.load_state_dict(state_dict)
                self.eval().cuda()
                loaded_successfully = True
            except RuntimeError as e:
                print(f"Error loading checkpoint with migration method: {e}")

        if not loaded_successfully:
            raise Exception("Error loading checkpoint. Make sure you are using a compatible checkpoint file.")
        
    @property
    def uncond_embedding(self):
        return self.model.uncond_embedding