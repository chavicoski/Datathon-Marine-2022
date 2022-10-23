"""Implementation of the data loaders
"""
import os
from random import random
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MarineSoundDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        audio_dir: str,
        labels: List[str],
        drop_silence: bool = False,
    ):
        """Dataset constructor

        Args:
            annotations_file (str): Audio annotations
            audio_dir (str): Path to the root folder of the audio data
            labels (List[str]): List of labels to use from the dataset
            drop_silence (bool): To drop the samples without any event
        """
        self.drop_silence = drop_silence
        if self.drop_silence:
            if "silence" in annotations.columns:
                valid_samples = annotations["silence"] == 1
            else:
                valid_samples = annotations[labels].apply(
                    lambda x: bool(x.sum()), axis="columns"
                )
            print(f"Going to drop {(~valid_samples).sum()} silence samples")
            self.annotations = annotations[valid_samples]
        else:
            self.annotations = annotations
        self.audio_dir = audio_dir
        self.labels = labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        feature_path, label_path = self._get_sample_paths(index)
        features = torch.load(feature_path)
        label = torch.load(label_path).astype(np.float32)
        return features, label

    def _get_sample_paths(self, index) -> Tuple[str, str]:
        """Returns the paths to the feature and label tensors"""
        sample_annot = self.annotations.iloc[index]
        features_path = os.path.join(self.audio_dir, sample_annot.feature_path)
        label_path = os.path.join(self.audio_dir, sample_annot.mask_path)
        return features_path, label_path

    def _get_label(self, index) -> np.ndarray:
        """Returns the one-hot vector of the selected labels"""
        onehot_arr = self.annotations[self.labels].iloc[index].values
        return torch.from_numpy(onehot_arr).float()


class MarineSoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        labels: List[str],
        batch_size: int = 32,
        num_workers: int = 32,
        mixup_prob: float = 0.0,
        drop_silence: bool = False,
    ):
        """DatasetModule constructor

        Args:
            annotations_file (str): Path to the TSV with annotations
            audio_dir (str): Path to the root folder of the audio data
            labels (List[str]): List of labels to use from the dataset
            batch_size (int): Number of audio samples per batch
            num_workers (int): Worker threads to use for data loading
            mixup_prob (float): Probability to mixup the samples in a batch
                                (only durin fit)
            drop_silence (bool): To drop the samples without any event
                                 (in train dataset)
        """
        super().__init__()
        self.annotations = pd.read_csv(annotations_file, sep="\t")
        self.audio_dir = audio_dir
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_prob = mixup_prob
        self.drop_silence = drop_silence
        if self.mixup_prob:
            self.mixup_transform = MixUpTransform()

    def setup(self, stage: str):
        # Split in train and test data
        train_annotations, test_annotations = train_test_split(
            self.annotations, test_size=0.2, shuffle=True
        )
        # Split the train data into train and validation
        train_split_annotations, val_split_annotations = train_test_split(
            train_annotations, test_size=0.2, shuffle=True
        )
        self.train_dataset = MarineSoundDataset(
            train_split_annotations,
            self.audio_dir,
            self.labels,
            self.drop_silence,
        )
        self.val_dataset = MarineSoundDataset(
            val_split_annotations,
            self.audio_dir,
            self.labels,
            False,
        )
        self.test_dataset = MarineSoundDataset(
            test_annotations,
            self.audio_dir,
            self.labels,
            False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # Apply the mixup by probability and only during training (Trainer.fit)
        if (random() < self.mixup_prob) and (self.trainer.state.fn == "fit"):
            return self.mixup_transform(batch)
        return batch


class MixUpTransform(object):
    """Batch tansformation top apply mixup by permuting the samples in the batch
    and mixing them (fetures and labels)"""

    def __init__(
        self, alpha: float = 0.2, beta: float = 0.2, mixup_type: str = "hard"
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.mixup_type = mixup_type

    def __call__(self, batch: Any) -> Any:
        x, y = batch

        # Permute the samples in the batch to mix them randomly
        batch_size = x.shape[0]
        permutation = torch.randperm(batch_size)

        # Get the mix factor
        if self.mixup_type == "hard":
            # c in range [0.3, 0.7]
            c = np.random.beta(self.alpha, self.beta) * 0.4 + 0.3
            mixed_y = torch.clamp(y + y[permutation], min=0, max=1)
        elif self.mixup_type == "soft":
            c = np.random.beta(self.alpha, self.beta)
            mixed_y = torch.clamp(c * y + (1 - c) * y[permutation], min=0, max=1)
        else:
            raise ValueError(f"Unexpected mixup type ('{self.mixup_type}')")

        mixed_x = c * x + (1 - c) * x[permutation]

        return mixed_x, mixed_y
