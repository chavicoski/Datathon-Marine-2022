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
        frame_duration: float = 0.04,
        hop_duration: float = 0.02,
        sample_rate: int = 50000,
        n_frames: int = 512,
    ):
        """Initialization of the preprocessing pipeline

        Args:
            audio_dir (str): Path to the raw data audio directory
            annotations_file (str): Path to the raw data annotations file
            out_dir (str): Directory to save the new preprocessed dataset
            frame_duration (float): Duration in seconds of a frame
            hop_duration (float): Duration in seconds of the hop
            sample_rate (int): Sample rate of the dataset
            n_frames (int): Frames to take for each chunk to classify
        """
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(annotations_file)
        self.frame_duration = frame_duration
        self.hop_duration = hop_duration
        self.sample_rate = sample_rate
        self.n_frames = n_frames

        # Name of the new directories to store the preprocessed dataset
        self.out_dir = out_dir
        self.dataset_name = f"frame-{frame_duration}_hop-{hop_duration}"
        self.dataset_dir = os.path.join(self.out_dir, self.dataset_name)
        self.tensor_dir = os.path.join(self.dataset_dir, "audio_tensors")

        # Prepare auxiliary variables for data labeling
        self.sorted_labels = self.annotations.label.value_counts().index
        self.n_labels = len(self.sorted_labels)
        self.labels2idx = {l: i for i, l in enumerate(self.sorted_labels)}
        self.sample_duration = 1 / sample_rate
        self.frame_size = int(self.frame_duration / self.sample_duration)
        self.hop_size = int(self.hop_duration / self.sample_duration)

    def preprocess_data(self):
        # Prepare the dataset dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.tensor_dir, exist_ok=True)

        # Prepare the kernel to extract the spectrograms
        spect_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.frame_size,
            win_length=self.frame_size,
            hop_length=self.hop_size,
            center=False,  # To match the number of frames in the labels_mask
        )

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
                spectrogram, frame_length=self.n_frames, hop_length=self.n_frames // 2
            )
            mask_chunks = librosa.util.frame(
                labels_mask, frame_length=self.n_frames, hop_length=self.n_frames // 2
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
                chunk_tensors.append(tensor_fname)

                # Combine the mask_chunk frames to create a 1D one-hot vector
                # for multilabel classification
                acc_mask_frames = mask_chunk.sum(axis=1)  # Accumulate frames labels
                onehot_label = acc_mask_frames.astype(bool).astype(np.int8)
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
