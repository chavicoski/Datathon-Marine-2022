"""Implementation of the functionalities to preprocess the raw dataset to
prepare it for training models"""
from abc import ABC, abstractmethod

import pandas as pd


class PreprocPipeline(ABC):
    @abstractmethod
    def preprocess_data(self):
        """Executes the preprocessing"""
        pass


class ChunkPreprocPipeline(object):
    """Pipeline to preprocess the data to perform direct audio classification

    All the audios are divided into short chunks with a one-hot encoded label
    to train a simple classifier
    """

    def __init__(self, audio_dir: str, annotations_file: str, out_dir: str):
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(annotations_file)
        self.out_dir = out_dir

    def preprocess_data(self):
        # For each audio file in self.annotations
        #   - Load the audio
        #   - Extract features (spectrogram)
        #   - Create labels (2D matrix)
        #   - Split features and labels by frame
        #   - Store the features in tensors and the labels in a CSV
        #
        # Note: Create the folder in self.out_dir to store the preprocessed
        #       fetures and the new labels csv
        raise NotImplementedError()
