"""
utils.py

This module includes miscellaneous utility functions that support the project,
such as data_preprocessing, logging, file handling, and general-purpose helpers.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
from torch.utils.tensorboard import SummaryWriter
import time
import pickle
import librosa
import librosa.feature
import numpy as np
import cv2
from PIL import Image
import torch


def setup(params):
    """
    Sets up the environment for training by creating directories for model checkpoints
    and logging, saving configuration parameters, and initializing a tensorboard summary writer.
    Args:
        params (dict): Dictionary containing the configuration parameters.
    Returns:
        tuple: A tuple containing the path to the checkpoints folder and the tensorboard summary writer instance.
    """
    # create dir to save model checkpoints
    reference = params['net_type'] + '_' + params['modality'] + str(time.strftime('_%Y%m%d_%H%M%S'))
    checkpoints_folder = os.path.join(params['output_folder'] + 'checkpoints' + reference)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # save the all the config/hyperparams to a pickle file
    pickle_filepath = os.path.join(str(checkpoints_folder), 'config.pkl')
    pickle_file = open(pickle_filepath, 'wb')
    pickle.dump(params, pickle_file)

    # create a tensorboard summary writer for logging and visualization
    log_dir = os.path.join(params['log_dir'], reference)
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=str(log_dir))

    return checkpoints_folder, summary_writer


def load_audio(audio_file, sampling_rate):
    """
    Loads an audio file.
    Args:
        audio_file (str): Path to the audio file.
        sampling_rate (int): Target sampling rate
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(path=audio_file, sr=sampling_rate, mono=False)
    return audio_data, sr


def extract_log_mel_spectrogram(audio, sr, n_fft, hop_length, win_length, nb_mels):
    """
    Computes the log Mel spectrogram from an audio signal.

    Parameters:
        audio (ndarray): NumPy array containing the audio waveform.
        sr (int): The sample rate of the audio signal.
        n_fft (int): Size of the FFT window.
        hop_length (int): Number of samples to shift between successive frames.
        win_length (int): Length of each windowed frame in samples.
        nb_mels (int): Number of Mel filter banks to use.

    Returns:
        ndarray: Array of shape (2, time_frames, nb_mels) - log Mel spectrogram for each channel.
    """

    linear_stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T
    linear_stft_mag = np.abs(linear_stft) ** 2
    mel_spec = librosa.feature.melspectrogram(S=linear_stft_mag, sr=sr, n_mels=nb_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spec)
    log_mel_spectrogram = log_mel_spectrogram.transpose((2, 0, 1))
    return log_mel_spectrogram


def load_video(video_file, fps):
    """
    Loads video frames from a video file.
    Args:
        video_file (str): Path to the video file.
        fps (int): Target frames per second
    Returns:
        list: List of PIL images (frames).
    """

    cap = cv2.VideoCapture(video_file)
    video_fps = 30
    frame_interval = max(1, video_fps // fps)
    pil_frames = []
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_cnt % frame_interval == 0:  # smoothening may not be required since we process the frames individually through a resnet
            resized_frame = cv2.resize(frame, (360, 180))
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            pil_frames.append(pil_frame)
        frame_cnt += 1
    cap.release()
    return pil_frames


def extract_resnet_features(video_frames, resnet_preprocessor, resnet_backbone, device):
    """
    Extracts ResNet-50 features from video frames.
    Args:
        video_frames (list): List of PIL video frames.
        resnet_preprocessor (callable): Preprocessing function to prepare images for ResNet input.
        resnet_backbone (torch.nn.Module): Pre-trained ResNet model to extract features.
        device (str): Device to perform computation on (e.g., 'cuda' or 'cpu').
    Returns:
        tensor: Extracted video features of shape (nb_frames, 7 ,7)
    """

    with torch.no_grad():
        preprocessed_images = [resnet_preprocessor(image) for image in video_frames]
        preprocessed_images = torch.stack(preprocessed_images, dim=0).to(device)
        vid_features = resnet_backbone(preprocessed_images)
        vid_features = torch.mean(vid_features, dim=1)
        return vid_features


def load_labels(label_file, convert_to_cartesian=True):
    label_data = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
        for line in lines:
            values = line.strip().split(',')
            frame_idx = int(values[0])
            data_row = [int(values[1]), int(values[2]), float(values[3]), float(values[4]), int(values[5])]
            if frame_idx not in label_data:
                label_data[frame_idx] = []
            label_data[frame_idx].append(data_row)

    if convert_to_cartesian:
        label_data = convert_polar_to_cartesian(label_data)
    return label_data


def process_labels(_desc_file, _nb_label_frames, _nb_unique_classes):

    se_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    x_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    y_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    dist_label = np.zeros((_nb_label_frames, _nb_unique_classes))
    onscreen_label = np.zeros((_nb_label_frames, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            for active_event in active_event_list:
                # print(active_event)
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[2]
                y_label[frame_ind, active_event[0]] = active_event[3]
                dist_label[frame_ind, active_event[0]] = active_event[4]/100.
                onscreen_label[frame_ind, active_event[0]] = active_event[5]

    label_mat = np.concatenate((se_label, x_label, y_label, dist_label, onscreen_label), axis=1)
    return label_mat


def process_labels_adpit(_desc_file, _nb_label_frames, _nb_unique_classes):

    se_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))  # 50, 6, 13
    x_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))
    y_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))
    dist_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))
    onscreen_label = np.zeros((_nb_label_frames, 6, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
            active_event_list_per_class = []
            for i, active_event in enumerate(active_event_list):
                active_event_list_per_class.append(active_event)
                if i == len(active_event_list) - 1:  # if the last
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4] / 100.
                        onscreen_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4] / 100.
                        onscreen_label[frame_ind, 0, active_event_b0[0]] = active_event_b0[5]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4] / 100.
                        onscreen_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]

                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4] / 100.
                        onscreen_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]

                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4] / 100.
                        onscreen_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4] / 100.
                        onscreen_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]

                elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4] / 100.
                        onscreen_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4] / 100.
                        onscreen_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4] / 100.
                        onscreen_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4] / 100.
                        onscreen_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4] / 100.
                        onscreen_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4] / 100.
                        onscreen_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]
                    active_event_list_per_class = []

    label_mat = np.stack((se_label, x_label, y_label, dist_label, onscreen_label), axis=2)  # [nb_frames, 6, 5(act+XY+dist+onscreen), max_classes]
    return label_mat


def convert_polar_to_cartesian(input_dict):
    output_dict = {}
    for frame_idx in input_dict.keys():
        if frame_idx not in output_dict:
            output_dict[frame_idx] = []
        for tmp_val in input_dict[frame_idx]:
            azi_rad = tmp_val[2]*np.pi/180
            x = np.cos(azi_rad)
            y = np.sin(azi_rad)
            output_dict[frame_idx].append(tmp_val[0:2] + [x, y] + tmp_val[3:])
    return output_dict


def convert_cartesian_to_polar(input_dict):
    output_dict = {}
    for frame_idx in input_dict.keys():
        if frame_idx not in output_dict:
            output_dict[frame_idx] = []
        for tmp_val in input_dict[frame_idx]:
            x = tmp_val[2]
            y = tmp_val[3]
            azi_rad = np.arctan2(y, x)
            azimuth = azi_rad * 180 / np.pi
            output_dict[frame_idx].append(tmp_val[0:2] + [azimuth] + tmp_val[4:])
    return output_dict
