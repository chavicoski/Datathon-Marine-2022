"""Implementation of the data loaders
"""

import os
from typing import Callable, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MarineSoundDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        audio_dir: str,
        target_sample_rate: int,
        n_samples: int,
        transform: Callable,
    ):
        """Dataset constructor

        Args:
            annotations_file (str): Audio annotations
            audio_dir (str): Path to the root folder of the audio data
            target_sample_rate (int): Sample rate to use (resampling if needed)
            n_samples (int): Samples to take from each audio (cut or pad if needed)
            transform (Callable): Transform function to apply to the signal
        """
        self.annotations = annotations
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        # Load data
        audio_path = self._get_audio_path(index)
        label = self._get_label(index)
        signal, sample_rate = torchaudio.load(audio_path)

        # Preprocess
        signal = self._resample(signal, sample_rate)
        signal = self._stereo_to_mono(signal)
        signal = self._pad_cut(signal)
        signal = self.transform(signal)

        return signal, label

    def _get_audio_path(self, index) -> str:
        audio_annotations = self.annotations.iloc[index]
        fold = f"fold{audio_annotations.fold}"
        audio_path = os.path.join(
            self.audio_dir, fold, audio_annotations.slice_file_name
        )
        return audio_path

    def _get_label(self, index) -> int:
        return self.annotations.iloc[index].classID

    def _resample(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Resamples the signal if its sample rate is not `target_sample_rate`

        Args:
            signal (torch.Tensor): Signal to resample
            sample_rate (int): Original sample rate of the signal

        Returns:
            torch.Tensor: Signal with sample rate equal to `target_sample_rate`
        """
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            return resampler(signal)
        return signal

    def _stereo_to_mono(self, signal: torch.Tensor) -> torch.Tensor:
        """If the signal is not mono averages all the channels to get the mono
        version

        Args:
            signal (torch.Tensor): Signal to convert to mono

        Returns:
            torch.Tensor: Mono signal
        """
        if signal.shape[0] > 1:  # shape: (num_channels, samples)
            return torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _pad_cut(self, signal: torch.Tensor) -> torch.Tensor:
        """Adjusts the signal length to be equal to `n_samples`. Adds right
        padding if it is shorter or cuts it if longer

        Args:
            signal (torch.Tensor): Signal to adjust the length

        Returns:
            torch.Tensor: Signal with length `n_samples`
        """
        signal_length = signal.shape[1]  # shape: (num_channels, samples)
        if signal_length < self.n_samples:
            # Apply right padding
            pad_length = self.n_samples - signal_length
            return torch.nn.functional.pad(signal, (0, pad_length))
        elif signal_length > self.n_samples:
            return signal[:, : self.n_samples]  # Cut the signal

        return signal


class MarineSoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        target_sample_rate: int,
        n_samples: int,
        transform: Callable,
        batch_size: int = 32,
        num_workers: int = 32,
    ):
        """DatasetModule constructor

        Args:
            annotations_file (str): Path to the CSV with annotations
            audio_dir (str): Path to the root folder of the audio data
            target_sample_rate (int): Sample rate to use (resampling if needed)
            n_samples (int): Samples to take from each audio (cut or pad if needed)
            transform (Callable): Transform function to apply to the signal
            batch_size (int): Number of audio samples per batch
        """
        super().__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        # Split in train and test data
        train_annotations, test_annotations = train_test_split(
            self.annotations, test_size=0.2, stratify=self.annotations.classID
        )
        # Split the train data into train and validation
        train_split_annotations, val_split_annotations = train_test_split(
            train_annotations, test_size=0.2, stratify=train_annotations.classID
        )
        self.train_dataset = MarineSoundDataset(
            train_split_annotations,
            self.audio_dir,
            self.target_sample_rate,
            self.n_samples,
            self.transform,
        )
        self.val_dataset = MarineSoundDataset(
            val_split_annotations,
            self.audio_dir,
            self.target_sample_rate,
            self.n_samples,
            self.transform,
        )
        self.test_dataset = MarineSoundDataset(
            test_annotations,
            self.audio_dir,
            self.target_sample_rate,
            self.n_samples,
            self.transform,
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
