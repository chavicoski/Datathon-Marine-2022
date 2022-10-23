"""Implementation of the Lightning modules to control the train/test loops
"""
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import (MultilabelAccuracy, MultilabelF1Score,
                                         MultilabelPrecision, MultilabelRecall)


class ClassifierModule(pl.LightningModule):
    """Module for training and testing classification models"""

    def __init__(self, model, labels: List[str]):
        super().__init__()
        self.model = model
        self.labels = labels

        # Accuracy
        n_labels = len(self.labels)
        self.train_acc = MultilabelAccuracy(n_labels)
        self.val_acc = MultilabelAccuracy(n_labels)
        self.test_acc = MultilabelAccuracy(n_labels)
        # Precision score
        self.train_precision = MultilabelPrecision(n_labels)
        self.val_precision = MultilabelPrecision(n_labels)
        self.test_precision = MultilabelPrecision(n_labels)
        # Recall score
        self.train_recall = MultilabelRecall(n_labels)
        self.val_recall = MultilabelRecall(n_labels)
        self.test_recall = MultilabelRecall(n_labels)
        # F1 score
        self.train_f1 = MultilabelF1Score(n_labels)
        self.val_f1 = MultilabelF1Score(n_labels)
        self.test_f1 = MultilabelF1Score(n_labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.train_acc(logits, y)
        self.train_precision(logits, y)
        self.train_recall(logits, y)
        self.train_f1(logits, y)
        self.log(f"train_loss", loss, prog_bar=True)
        self.log(f"train_acc", self.train_acc, prog_bar=True)
        self.log(f"train_precision", self.train_acc, prog_bar=False)
        self.log(f"train_recall", self.train_acc, prog_bar=False)
        self.log(f"train_f1", self.train_f1, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.val_acc(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_f1(logits, y)
        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_acc", self.val_acc, prog_bar=True)
        self.log(f"val_precision", self.val_acc, prog_bar=False)
        self.log(f"val_recall", self.val_acc, prog_bar=False)
        self.log(f"val_f1", self.val_acc, prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.test_acc(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)
        self.test_f1(logits, y)
        self.log(f"test_loss", loss, prog_bar=True)
        self.log(f"test_acc", self.test_acc, prog_bar=True)
        self.log(f"test_precision", self.test_acc, prog_bar=False)
        self.log(f"test_recall", self.test_acc, prog_bar=False)
        self.log(f"test_f1", self.test_acc, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)
