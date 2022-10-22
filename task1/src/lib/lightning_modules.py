"""Implementation of the Lightning modules to control the train/test loops
"""
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class ClassifierModule(pl.LightningModule):
    """Module for training and testing classification models"""

    def __init__(self, model, labels: List[str]):
        super().__init__()
        self.model = model
        self.labels = labels

        # Metrics
        n_labels = len(self.labels)
        self.train_acc = torchmetrics.classification.MultilabelAccuracy(n_labels)
        self.val_acc = torchmetrics.classification.MultilabelAccuracy(n_labels)
        self.test_acc = torchmetrics.classification.MultilabelAccuracy(n_labels)
        self.train_f1 = torchmetrics.classification.MultilabelF1Score(n_labels)
        self.val_f1 = torchmetrics.classification.MultilabelF1Score(n_labels)
        self.test_f1 = torchmetrics.classification.MultilabelF1Score(n_labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.train_acc(logits, y)
        self.log(f"train_loss", loss, prog_bar=True)
        self.log(f"train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.val_acc(logits, y)
        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.test_acc(logits, y)
        self.log(f"test_loss", loss, prog_bar=True)
        self.log(f"test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
