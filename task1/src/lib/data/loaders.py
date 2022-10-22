"""Implementation of the data loaders
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MarineSoundDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, audio_dir: str, labels: List[str]):
        """Dataset constructor

        Args:
            annotations_file (str): Audio annotations
            audio_dir (str): Path to the root folder of the audio data
            labels (List[str]): List of labels to use from the dataset
        """
        self.annotations = annotations
        self.audio_dir = audio_dir
        self.labels = labels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        features = torch.load(self._get_features_path(index))
        label = self._get_label(index)
        return features, label

    def _get_features_path(self, index) -> str:
        features_tensor_name = self.annotations.path.iloc[index]
        return os.path.join(self.audio_dir, features_tensor_name)

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
    ):
        """DatasetModule constructor

        Args:
            annotations_file (str): Path to the TSV with annotations
            audio_dir (str): Path to the root folder of the audio data
            labels (List[str]): List of labels to use from the dataset
            batch_size (int): Number of audio samples per batch
            num_workers (int): Worker threads to use for data loading
        """
        super().__init__()
        self.annotations = pd.read_csv(annotations_file, sep="\t")
        self.audio_dir = audio_dir
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        )
        self.val_dataset = MarineSoundDataset(
            val_split_annotations,
            self.audio_dir,
            self.labels,
        )
        self.test_dataset = MarineSoundDataset(
            test_annotations,
            self.audio_dir,
            self.labels,
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
