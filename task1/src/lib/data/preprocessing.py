"""Implementation of the functionalities to preprocess the raw dataset to
prepare it for training models"""
import os
from abc import ABC, abstractmethod
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


class PreprocPipeline(ABC):
    @abstractmethod
    def preprocess_data(self):
        """Executes the preprocessing"""
        pass


class ChunkPreprocPipeline(object):
    """Pipeline to preprocess the data to perform direct audio classification

    All the audios are divided into short chunks with a one-hot encoded label
    to train a multilabel classifier
    """

    def __init__(
        self,
        audio_dir: str,
        annotations_file: str,
        out_dir: str,
        frame_size: int = 1024,
        hop_size: int = 512,
        sample_rate: int = 50000,
        chunk_size: int = 256,
        spec_type: str = "mel",
        n_mels: int = 64,
        only_events: bool = False,
    ):
        """Initialization of the preprocessing pipeline

        Args:
            audio_dir (str): Path to the raw data audio directory
            annotations_file (str): Path to the raw data annotations file
            out_dir (str): Directory to save the new preprocessed dataset
            frame_size (float): Number of samples per frame
            hop_size (float): Number of samples to hop between frames
            sample_rate (int): Sample rate of the dataset
            chunk_size (int): Number of frames to take for each chunk
            spec_type (str): Type of spectrogram to use. Choices: "mel", "base"
            n_mels (int): Number of mel filters to use (is using spec_type="mel")
            only_events (bool): Creates a dataset with chunks with at least one label
        """
        self.audio_dir = audio_dir
        self.annotations = self._drop_wrong_labels(pd.read_csv(annotations_file))
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.spec_type = spec_type
        self.n_mels = n_mels
        self.only_events = only_events

        # Name of the new directories to store the preprocessed dataset
        self.out_dir = out_dir
        self.dataset_name = (
            f"frame-{self.frame_size}"
            f"_hop-{self.hop_size}"
            f"_chunk-{self.chunk_size}"
            f"_spec-{spec_type}"
        )
        if self.spec_type == "mel":
            self.dataset_name += f"_mels-{self.n_mels}"
        if self.only_events:
            self.dataset_name += "_only-events"

        self.dataset_dir = os.path.join(self.out_dir, self.dataset_name)
        self.tensor_dir = os.path.join(self.dataset_dir, "audio_tensors")

        # Prepare auxiliary variables for data labeling
        self.sorted_labels = self.annotations.label.value_counts().index
        self.n_labels = len(self.sorted_labels)
        self.labels2idx = {l: i for i, l in enumerate(self.sorted_labels)}
        self.sample_duration = 1 / sample_rate
        self.frame_duration = self.sample_duration * self.frame_size
        self.hop_duration = self.sample_duration * self.hop_size

    def _drop_wrong_labels(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """Drop the labels corresponding to long click events"""
        drop_mask = annotations.duration > 30
        drop_mask &= annotations.label == "click"
        return annotations[~drop_mask]

    def _print_start_msg(self):
        print("Preprocessing config:")
        print(f" - frame size: {self.frame_size}")
        print(f" - frame duration (sec): {self.frame_duration}")
        print(f" - hop size: {self.hop_size}")
        print(f" - hop duration (sec): {self.hop_duration}")
        print(f" - chunk size (frames): {self.chunk_size}")
        print(f" - labels: {self.sorted_labels}")

    def preprocess_data(self):
        # Prepare the dataset dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.tensor_dir, exist_ok=True)

        self._print_start_msg()

        # Prepare the kernel to extract the spectrograms
        if self.spec_type == "base":
            spect_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.frame_size,
                win_length=self.frame_size,
                hop_length=self.hop_size,
                center=False,  # To match the number of frames in the labels_mask
            )
        elif self.spec_type == "mel":
            spect_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.frame_size,
                win_length=self.frame_size,
                hop_length=self.hop_size,
                center=False,  # To match the number of frames in the labels_mask
                n_mels=self.n_mels,
            )
        else:
            raise ValueError(f"Invalid spectrogram type ('{self.spec_type})")

        # Create the samples from the chunks
        chunk_tensors = []  # .pt tensors file names
        chunk_labels = []  # One-hot vector labels
        for audio_name, audio_annot in tqdm(self.annotations.groupby("path")):
            # Load the audio data
            audio_file = os.path.join(self.audio_dir, f"{audio_name}.wav")
            signal, sr = torchaudio.load(audio_file)

            if sr != self.sample_rate:
                # The pipeline expects that all the audios are of the same sample_rate
                raise ValueError(f"The sampling rate of {audio_file} is {sr}!")

            # Compute the full audio spectrogram and labels mask
            spectrogram = spect_transform(signal)
            labels_mask = labels_to_mask(
                self.audio_dir,
                audio_name,
                audio_annot,
                self.frame_size,
                self.hop_size,
                self.n_labels,
                self.labels2idx,
            )

            # Extract the chunks of frames that correspond to each sample
            spectrogram_chunks = librosa.util.frame(
                spectrogram,
                frame_length=self.chunk_size,
                hop_length=self.chunk_size // 2,
            )
            mask_chunks = librosa.util.frame(
                labels_mask,
                frame_length=self.chunk_size,
                hop_length=self.chunk_size // 2,
            )

            # Sanity check
            # spectrogram_chunks shape: (channels, freq, chunk_frames, chunks)
            # mask shape: (labels, chunk_frames, chunks)
            assert spectrogram_chunks.shape[-2:] == mask_chunks.shape[-2:]
            _, chunk_frames, chunks = mask_chunks.shape

            # Store each spectrogram chunk and the corrsponding label
            for c in range(chunks):
                # spec_chunk shape: (channels, freq, chunk_frames)
                spec_chunk = spectrogram_chunks[:, :, :, c]
                # mask_chunk shape: (labels, chunk_frames)
                mask_chunk = mask_chunks[:, :, c]

                # Save the features tensor
                tensor_fname = f"{audio_name}_chunk{c}.pt"
                tensor_path = os.path.join(self.tensor_dir, tensor_fname)
                torch.save(spec_chunk, tensor_path)

                # Combine the mask_chunk frames to create a 1D one-hot vector
                # for multilabel classification
                acc_mask_frames = mask_chunk.sum(axis=1)  # Accumulate frames labels
                onehot_label = acc_mask_frames.astype(bool).astype(np.int8)

                if self.only_events and onehot_label.sum() == 0:
                    continue  # Skip chunks without and event

                chunk_tensors.append(tensor_fname)
                chunk_labels.append(onehot_label)

        # Save the labels in a TSV file
        chunk_labels = np.array(chunk_labels)
        label_columns = {l: chunk_labels[:, i] for l, i in self.labels2idx.items()}
        chunk_annotations = pd.DataFrame({"path": chunk_tensors, **label_columns})
        chunk_annot_file = os.path.join(self.dataset_dir, "labels.tsv")
        chunk_annotations.to_csv(chunk_annot_file, sep="\t", index=False)


def labels_to_mask(
    audio_dir: str,
    audio_name: str,
    audio_annot: pd.DataFrame,
    frame_size: int,
    hop_size: int,
    n_labels: int,
    labels2idx: Dict[str, int],
) -> torch.Tensor:
    """Given a DataFrame with all the annotations of an audio file, creates a binary
    2D tensor with shape (n_labels, n_frames) containing the labels at each frame
    (for a given `frame_size` and `hop_size`)

    Args:
        audio_dir (str): Path to the folder with the audio data
        audio_name (str): name of the audio file ("path" column in raw data)
        audio_annot (pd.DataFrame): Annotations of the audio file to mask
        frame_size (int): Frame size (is samples) to create the mask
        hop_size (int): Hop size (is samples) to create the mask
        n_labels (int): Number of labels to use in the mask
        labels2idx (Dict[str, int]): Mapping from label name to index in the mask

    Returns:
        torch.Tensor: Binary mask of the labels
    """
    # Load the audio
    signal, sample_rate = torchaudio.load(os.path.join(audio_dir, f"{audio_name}.wav"))

    # Prepare the mask 2D matrix (labels, frames)
    n_frames = int((signal.shape[-1] - frame_size) / hop_size) + 1
    mask = torch.zeros((n_labels, n_frames))
    # Compute some utility values
    sample_time = 1 / sample_rate
    frame_time = sample_time * frame_size
    for _, row in audio_annot.iterrows():
        start_frame = int(row.start / frame_time)
        end_frame = int(row.end / frame_time)
        mask[labels2idx[row.label], start_frame:end_frame] += 1

    return mask.bool().int()
