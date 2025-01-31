"""
extract_features.py

This module defines the SELDFeatureExtractor class, which provides functionality to extract features from both audio
and video data. It includes the following key components:

Classes:
    SELDFeatureExtractor: A class that supports the extraction of audio and video features.
    It can extract log Mel spectrogram from audio files and ResNet-based features from video frames.

    Methods:
        - extract_audio_features: Extracts audio features from a specified split of the dataset.
        - extract_video_features: Extracts video features from a specified split of the dataset.
        - extract_features: A high-level function to extract features based on the modality ('audio' or 'audio_visual').

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

import os
import librosa
import librosa.feature
import glob
import numpy as np
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm


class SELDFeatureExtractor():
    def __init__(self, params):
        """
        Initializes the SELDFeatureExtractor with the provided parameters.
        Args:
            params (dict): A dictionary containing various parameters for audio/video feature extraction among others.
        """

        self.params = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']

        self.modality = params['modality']

        # audio feature extraction
        self.sampling_rate = params['sampling_rate']
        self.max_audio_len_in_s = params['max_audio_len_in_s']
        self.hop_length = int(self.sampling_rate * params['hop_length_s'])
        self.win_length = 2 * self.hop_length
        self.n_fft = 2 ** (self.win_length - 1).bit_length()
        self.nb_mels = params['nb_mels']

        # video feature extraction
        if self.modality == 'audio_visual':
            self.fps = params['fps']
            self.max_video_len_in_s = params['max_video_len_in_s']
            # Initialize ResNet model and set to evaluation mode
            self.weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=self.weights)
            self.backbone = torch.nn.Sequential(*(list(self.model.children())[:-2]))
            self.backbone.eval()
            self.preprocess = self.weights.transforms()

    def extract_audio_features(self, split):
        """
        Extracts audio features for a given split (dev/eval).
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        if split == 'dev':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split == 'eval':
            audio_files = glob.glob(os.path.join(self.root_dir, 'stereo_eval', 'eval', '*.wav'))
        else:
            raise AssertionError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, f'stereo_{split}'), exist_ok=True)

        for audio_file in tqdm(audio_files, desc=f"Processing audio files ({split})", unit="file"):
            filename = os.path.splitext(os.path.basename(audio_file))[0] + '.npy'
            feature_path = os.path.join(self.feat_dir, f'stereo_{split}', filename)
            # Check if the feature file already exists
            if os.path.exists(feature_path):
                continue
            # If the feature file doesn't exist, perform extraction
            audio, sr = self.load_audio(audio_file)
            audio_feat = self.extract_log_mel_spectrogram(audio, sr)
            np.save(feature_path, audio_feat, allow_pickle=True)

    def extract_video_features(self, split):
        """
        Extracts video features for a given split (dev/eval).
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """
        if split == 'dev':
            video_files = glob.glob(os.path.join(self.root_dir, 'video_dev', 'dev-*', '*.mp4'))
        elif split == 'eval':
            video_files = glob.glob(os.path.join(self.root_dir, 'video_eval', 'eval', '*.mp4'))
        else:
            raise AssertionError("Split must be either 'dev' or 'eval'.")

        os.makedirs(os.path.join(self.feat_dir, f'video_{split}'), exist_ok=True)

        for video_file in tqdm(video_files, desc=f"Processing video files ({split})", unit="file"):
            filename = os.path.splitext(os.path.basename(video_file))[0] + '.npy'
            feature_path = os.path.join(self.feat_dir, f'video_{split}', filename)

            # Check if the feature file already exists
            if os.path.exists(feature_path):
                continue

            # If the feature file doesn't exist, perform extraction
            video_frames = self.load_video(video_file)
            video_feat = self.extract_resnet_features(video_frames)
            np.save(feature_path, video_feat, allow_pickle=True)

    def extract_features(self, split='dev'):
        """
        Extracts features based on the selected modality ('audio' or 'audio_visual').
        Args:
            split (str): The split for which features need to be extracted ('dev' or 'eval').
        """

        os.makedirs(self.feat_dir, exist_ok=True)

        if self.modality == 'audio':
            self.extract_audio_features(split)
        elif self.modality == 'audio_visual':
            self.extract_audio_features(split)
            self.extract_video_features(split)
        else:
            raise AssertionError("Modality should be one of 'audio' or 'audio_visual'. You can set the modality in params.py")

    def load_audio(self, audio_file):
        """
        Loads an audio file.
        Args:
            audio_file (str): Path to the audio file.
        Returns:
            tuple: (audio_data, sample_rate)
        """

        audio_data, sr = librosa.load(path=audio_file, sr=self.sampling_rate, mono=False)
        return audio_data, sr

    def extract_log_mel_spectrogram(self, audio, sr):
        """
        Extracts log mel spectrogram from audio data.
        Args:
            audio (ndarray): Audio data.
            sr (int): Sample rate of the audio.
        Returns:
            ndarray: Log mel spectrogram.
        """

        linear_stft = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length).T
        linear_stft_mag = np.abs(linear_stft) ** 2
        mel_spec = librosa.feature.melspectrogram(S=linear_stft_mag, sr=sr, n_mels=self.nb_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spec)
        log_mel_spectrogram = log_mel_spectrogram.transpose((0, 2, 1)).reshape(linear_stft.shape[0], -1)
        return log_mel_spectrogram

    def load_video(self, video_file):
        """
        Loads video frames from a video file.
        Args:
            video_file (str): Path to the video file.
        Returns:
            list: List of PIL images (frames).
        """

        cap = cv2.VideoCapture(video_file)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, video_fps // self.fps)

        pil_frames = []
        frame_cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_cnt % frame_interval == 0:
                resized_frame = cv2.resize(frame, (360, 180))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frames.append(pil_frame)
            frame_cnt += 1
        cap.release()
        return pil_frames

    def extract_resnet_features(self, video_frames):
        """
        Extracts ResNet-50 features from video frames.
        Args:
            video_frames (list): List of PIL video frames.
        Returns:
            ndarray: Extracted video features.
        """

        with torch.no_grad():
            # Preprocess images
            preprocessed_images = [self.preprocess(image) for image in video_frames]
            max_batch_size = 1000
            iter = (len(preprocessed_images) - 1) // max_batch_size + 1
            vid_features_part_list = []
            for i in range(iter):
                preprocessed_images_part = torch.stack(preprocessed_images[i * max_batch_size: (i + 1) * max_batch_size], dim=0)
                vid_features_part = self.backbone(preprocessed_images_part)
                vid_features_part = torch.mean(vid_features_part, dim=1)
                vid_features_part_list.append(vid_features_part)
            vid_features = torch.cat(vid_features_part_list, dim=0)
        vid_feat = np.array(vid_features)
        return vid_feat


if __name__ == '__main__':
    pass
    # use this to test if the SELDFeatureExtractor class works as expected after the dataset is ready.
    # All the classes will be called from the main.py for actual use.
