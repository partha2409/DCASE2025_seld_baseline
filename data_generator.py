"""
data_generator.py

This module handles the creation of data generators for efficient data loading and preprocessing during training.

Class:
    DataGenerator: A data generator for efficient data loading and preprocessing during training.

Methods:
    __init__(self, params, mode='dev_train'): Initializes the DataGenerator instance.
    __getitem__(self, item): Returns the data for a given index.
    __len__(self): Returns the number of data points.
    get_feature_files(self): Collects the paths to the feature files based on the selected folds and modality.
    get_folds(self): Returns the folds for the given data split.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
import torch
import glob
from torch.utils.data.dataset import Dataset
import numpy as np


class DataGenerator(Dataset):
    def __init__(self, params, mode='dev_train'):
        """
        Initializes the DataGenerator instance.
        Args:
            params (dict): Parameters for data generation.
            mode (str): data split ('dev_train', 'dev_test', 'dev_synth').
        """

        super().__init__()
        self.params = params
        self.mode = mode
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']
        self.modality = params['modality']

        self.folds = self.get_folds()

        # self.video_files will be an empty [] if self.modality == 'audio'
        self.audio_files, self.video_files, self.label_files = self.get_feature_files()

    def __getitem__(self, item):
        """
        Returns the data for a given index.
        Args:
            item (int): Index of the data.
        Returns:
            tuple: A tuple containing audio features, video_features (for audio_visual modality), and labels.
        """
        # TODO: Currently labels are returned with on/off screen. so handle loss accordingly.

        audio_file = self.audio_files[item]
        label_file = self.label_files[item]

        audio_features = torch.load(audio_file)
        labels = torch.load(label_file)

        if not self.params['multiACCDOA']:  # TODO: why masking and multiplication instead of simply omitting the first 13 entries?

            mask = labels[:, :self.params['nb_classes']]
            mask = mask.repeat(1, 4)
            labels = mask * labels[:, self.params['nb_classes']:]

        if self.modality == 'audio_visual':
            video_file = self.video_files[item]
            video_features = torch.load(video_file)
            return (audio_features, video_features), labels
        else:
            return audio_features, labels

    def __len__(self):
        """
        Returns the number of data points.
        Returns:
            int: Number of data points.
        """

        return len(self.audio_files)

    def get_feature_files(self):
        """
        Collects the paths to the feature and label files based on the selected folds and modality.
        Returns:
            tuple: A tuple containing lists of paths to audio feature files, video feature files, and processed label files.
        """
        audio_files, video_files, label_files = [], [], []

        # Loop through each fold and collect files
        for fold in self.folds:
            audio_files += glob.glob(os.path.join(self.feat_dir, f'stereo_dev/{fold}*.pt'))
            label_files += glob.glob(os.path.join(self.feat_dir, 'metadata_dev{}/{}*.pt'.format('_adpit' if self.params['multiACCDOA'] else '', fold)))

            # Only collect video files if modality is 'audio_video'
            if self.modality == 'audio_visual':
                video_files += glob.glob(os.path.join(self.feat_dir, f'video_dev/{fold}*.pt'))

        # Sort files to ensure corresponding audio, video, and label files are in the same order
        audio_files = sorted(audio_files, key=lambda x: x.split('/')[-1])
        label_files = sorted(label_files, key=lambda x: x.split('/')[-1])

        # Sort video files only if modality is 'audio_visual'
        if self.modality == 'audio_visual':
            video_files = sorted(video_files, key=lambda x: x.split('/')[-1])

        # Return the appropriate files based on modality
        if self.modality == 'audio':
            return audio_files, [], label_files
        elif self.modality == 'audio_visual':
            return audio_files, video_files, label_files
        else:
            raise ValueError(f"Invalid modality: {self.modality}. Choose from ['audio', 'audio_visual'].")

    def get_folds(self):
        """
        Returns the folds for the given data split
        Returns:
            list: List of folds.
        """
        if self.mode == 'dev_train':
            return self.params['dev_train_folds']  # fold 3
        elif self.mode == 'dev_test':
            return self.params['dev_test_folds']  # fold 4
        elif self.mode == 'dev_synth':
            assert self.modality == 'audio', "Modality must be set to 'audio' for 'dev_synth' mode, as there are no synthetic videos."
            return self.params['dev_synth_folds']  # fold 1 and fold 2
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from ['dev_train', 'dev_test', 'dev_synth'].")


if __name__ == '__main__':
    # use this space to test if the DataGenerator class works as expected.
    # All the classes will be called from the main.py for actual use.

    from parameters import params
    from torch.utils.data import DataLoader
    params['multiACCDOA'] = False
    dev_train_dataset = DataGenerator(params=params, mode='dev_train')
    dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'], pin_memory=False,  drop_last=True)

    for i, (input_features, labels) in enumerate(dev_train_iterator):
        if params['modality'] == 'audio':
            print(input_features.size())
            print(labels.size())
        elif params['modality'] == 'audio_visual':
            print(input_features[0].size())
            print(input_features[1].size())
            print(labels.size())
        break



