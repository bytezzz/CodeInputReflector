import torch.nn as nn
import lightning.pytorch as pl
from triplet_loss import *
import lightning.pytorch as pl
import torch.nn as nn

from triplet_loss import *

# Common Network Architecture for both SiameseModel and QuadrupletModel, Difference lies on the loss function.
# Reference: https://github.com/cs-sun/InputReflector/blob/main/train.py

class Siamese(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        #Freeze feature_extractor's parameters,
        for param in feature_extractor.parameters():
            param.requires_grad = False
        
        dense_layers = [
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
        ]

        self.dense = nn.Sequential(*dense_layers)
    
    def forward(self, data):
        data = self.feature_extractor(data, attention_mask=data.ne(1), output_hidden_states=True).hidden_states[-1]
        return self.dense(data)

    def forward_quad(self, anchor, positive, negative):
        anchor = self.feature_extractor(anchor, attention_mask=anchor.ne(1), output_hidden_states=True).hidden_states[-1]
        positive = self.feature_extractor(positive, attention_mask=positive.ne(1), output_hidden_states=True).hidden_states[-1]
        negative = self.feature_extractor(negative, attention_mask=negative.ne(1), output_hidden_states=True).hidden_states[-1]
        anchor = self.dense(anchor)
        positive = self.dense(positive)
        negative = self.dense(negative)
        return anchor, positive, negative


class SiameseModel(pl.LightningModule):
    
    def quadruplet_loss(self, batch, model):
        margin1 = 0.5
        margin2 = 1.0
        margin3 = 1.5
        (x_train, x_train_tr, x_train_ood), y_train = batch
        x_train = x_train.to(self.device)
        x_train_tr = x_train_tr.to(self.device)
        x_train_ood = x_train_ood.to(self.device)
        y_train = y_train.to(self.device)
        anchor_output, positive_output, negative_output = self.model.forward_quad(x_train, x_train_tr, x_train_ood)
        return batch_hard_triplet_loss_c1c2_c1c(y_train, anchor_output, anchor_output,
                                                positive_output,
                                                margin=margin1,
                                                squared=False) + batch_hard_triplet_loss_c1c2_c1c(
            y_train, anchor_output, anchor_output, negative_output, margin=margin2,
            squared=False) + batch_hard_triplet_loss_c1c_c1c2(
            y_train, anchor_output, positive_output, anchor_output, margin=margin2, squared=False)
    
    def __init__(self, feature_extractor):
        super().__init__()
        self.model = Siamese(feature_extractor)
        self.save_hyperparameters(ignore=['feature_extractor'])

    def forward(self, x):
        return self.model(x.to(self.device))

    def training_step(self, batch, batch_idx):
        loss = self.quadruplet_loss(batch, self.model)
        self.log('train_loss', loss, on_epoch=True, on_step = False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.quadruplet_loss(batch, self.model)
        self.log('val_loss', loss, on_epoch=True, on_step = False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=torch.sqrt(torch.tensor(0.1)),
                    cooldown=0,
                    mode='min',
                    patience=5,
                    verbose=True,
                    min_lr=0.5e-20
                ),
                "monitor": "val_loss",
            }
        }

    
class QuadrupletModel(pl.LightningModule):
    
    def quadruplet_loss(self, batch, model):
        margin1 = 0.5
        margin2 = 1.0
        margin3 = 1.5
        (x_train, x_train_tr, x_train_ood), y_train = batch
        x_train = x_train.to(self.device)
        x_train_tr = x_train_tr.to(self.device)
        x_train_ood = x_train_ood.to(self.device)
        y_train = y_train.to(self.device)
        anchor_output, positive_output, negative_output = self.model.forward_quad(x_train, x_train_tr, x_train_ood)
        return  batch_hard_triplet_loss(y_train, anchor_output, anchor_output, anchor_output,
                                                      margin=margin1, squared=False) + batch_hard_triplet_loss_cde(
                y_train, anchor_output, anchor_output, anchor_output, margin=margin2, squared=False)
    
    def __init__(self, feature_extractor):
        super().__init__()
        self.model = Siamese(feature_extractor)
        self.save_hyperparameters(ignore=['feature_extractor'])

    def forward(self, x):
        return self.model(x.to(self.device))

    def training_step(self, batch, batch_idx):
        loss = self.quadruplet_loss(batch, self.model)
        self.log('train_loss', loss, on_epoch=True, on_step = False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self.quadruplet_loss(batch, self.model)
        self.log('val_loss', loss, on_epoch=True, on_step = False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=torch.sqrt(torch.tensor(0.1)),
                    cooldown=0,
                    mode='min',
                    patience=5,
                    verbose=True,
                    min_lr=0.5e-20
                ),
                "monitor": "val_loss",
            }
        }
