"""Implementation of the Lightning modules to control the train/test loops
"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class ClassifierModule(pl.LightningModule):
    """Module for training and testing classification models"""

    def __init__(self, model):
        super().__init__()
        self.model = model

        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log(f"train_loss", loss, prog_bar=True)
        self.log(f"train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)
        self.log(f"test_loss", loss, prog_bar=True)
        self.log(f"test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
