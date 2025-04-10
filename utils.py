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
from scipy import stats
from scipy.optimize import linear_sum_assignment
import warnings


def setup(params):
    """
    Sets up the environment for training by creating directories for model checkpoints
    and logging, saving configuration parameters, and initializing a tensorboard summary writer.
    Args:
        params (dict): Dictionary containing the configuration parameters.
    Returns:
        tuple: A tuple containing the path to the checkpoints folder, output folder and the tensorboard summary writer instance.
    """

    print('You are using the following configuration: \n\n')
    for key, value in params.items():
        print(key, ': ', value)

    # create dir to save model checkpoints
    reference = f"{params['net_type']}_{params['modality']}_{'multiACCDOA' if params['multiACCDOA'] else 'singleACCDOA'}{time.strftime('_%Y%m%d_%H%M%S')}"
    checkpoints_dir = os.path.join(params['checkpoints_dir'], reference)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # save the all the config/hyperparams to a pickle file
    pickle_filepath = os.path.join(str(checkpoints_dir), 'config.pkl')
    pickle_file = open(pickle_filepath, 'wb')
    pickle.dump(params, pickle_file)

    # create a tensorboard summary writer for logging and visualization
    log_dir = os.path.join(params['log_dir'], reference)
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=str(log_dir))

    # create output folder to save the predictions
    output_dir = os.path.join(params['output_dir'], reference)
    os.makedirs(output_dir, exist_ok=True)

    return checkpoints_dir, output_dir, summary_writer


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


def extract_stft(audio, n_fft, hop_length, win_length):
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T
    return stft


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

    linear_stft = extract_stft(audio, n_fft, hop_length, win_length)
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

    se_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    x_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    y_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    dist_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    onscreen_label = torch.zeros((_nb_label_frames, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            for active_event in active_event_list:
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[2]
                y_label[frame_ind, active_event[0]] = active_event[3]
                dist_label[frame_ind, active_event[0]] = active_event[4] / 100.
                onscreen_label[frame_ind, active_event[0]] = active_event[5]

    label_mat = torch.cat((se_label, x_label, y_label, dist_label, onscreen_label), dim=1)
    return label_mat


def process_labels_adpit(_desc_file, _nb_label_frames, _nb_unique_classes):

    se_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))  # 50, 6, 13
    x_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    y_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    dist_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    onscreen_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))

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

    label_mat = torch.stack((se_label, x_label, y_label, dist_label, onscreen_label), dim=2)  # [nb_frames, 6, 5(act+XY+dist+onscreen), max_classes]
    return label_mat


def organize_labels(input_dict, max_frames, max_tracks=10):
    """
    :param input_dict: Dictionary containing frame-wise sound event time and location information
            _pred_dict[frame-index] = [[class-index, source-index, azimuth, distance, onscreen] x events in frame]
    :param max_frames: Total number of frames in the recording
    :param max_tracks: Total number of tracks in the output dict
    :return: Dictionary containing class-wise sound event location information in each frame
            dictionary_name[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
    """
    tracks = set(range(max_tracks))
    output_dict = {x: {} for x in range(max_frames)}
    for frame_idx in range(0, max_frames):
        if frame_idx not in input_dict:
            continue
        for [class_idx, source_idx, az, dist, onscreen] in input_dict[frame_idx]:
            if class_idx not in output_dict[frame_idx]:
                output_dict[frame_idx][class_idx] = {}
            if source_idx not in output_dict[frame_idx][class_idx] and source_idx < max_tracks:
                track_idx = source_idx  # If possible, use source_idx as track_idx
            else:                       # If not, use the first one available
                try:
                    track_idx = list(set(tracks) - output_dict[frame_idx][class_idx].keys())[0]
                except IndexError:
                    warnings.warn("The number of sources of is higher than the number of tracks. "
                                  "Some events will be missed.")
                    track_idx = 0  # Overwrite one event
            output_dict[frame_idx][class_idx][track_idx] = [az, dist, onscreen]

    return output_dict


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


def get_accdoa_labels(logits, nb_classes, modality):
    x, y = logits[:, :, :nb_classes], logits[:, :, nb_classes:2 * nb_classes]
    sed = torch.sqrt(x ** 2 + y ** 2) > 0.5
    distance = logits[:, :, 2 * nb_classes: 3 * nb_classes]
    distance[distance < 0.] = 0.
    if modality == 'audio_visual':
        on_screen = logits[:, :, 3 * nb_classes: 4 * nb_classes]
    else:
        on_screen = torch.zeros_like(distance)  # don't care for audio modality
    dummy_src_id = torch.zeros_like(distance)
    return sed, dummy_src_id, x, y, distance, on_screen


def get_multiaccdoa_labels(logits, nb_classes, modality):
    if modality == 'audio':
        x0, y0 = logits[:, :, :1*nb_classes], logits[:, :, 1*nb_classes:2*nb_classes]
        sed0 = torch.sqrt(x0**2 + y0**2) > 0.5
        dist0 = logits[:, :, 2*nb_classes:3*nb_classes]
        dist0[dist0 < 0.] = 0
        doa0 = logits[:, :, :2*nb_classes]
        dummy_src_id0 = torch.zeros_like(dist0)
        on_screen0 = torch.zeros_like(dist0)

        x1, y1 = logits[:, :, 3*nb_classes:4 * nb_classes], logits[:, :, 4 * nb_classes: 5 * nb_classes]
        sed1 = torch.sqrt(x1 ** 2 + y1 ** 2) > 0.5
        dist1 = logits[:, :, 5 * nb_classes:6 * nb_classes]
        dist1[dist1 < 0.] = 0
        doa1 = logits[:, :, 3*nb_classes:5 * nb_classes]
        dummy_src_id1 = torch.zeros_like(dist1)
        on_screen1 = torch.zeros_like(dist1)

        x2, y2 = logits[:, :, 6*nb_classes:7 * nb_classes], logits[:, :, 7 * nb_classes:8 * nb_classes]
        sed2 = torch.sqrt(x2 ** 2 + y2 ** 2) > 0.5
        dist2 = logits[:, :, 8 * nb_classes:9 * nb_classes]
        dist2[dist2 < 0.] = 0
        doa2 = logits[:, :, 6*nb_classes:8 * nb_classes]
        dummy_src_id2 = torch.zeros_like(dist2)
        on_screen2 = torch.zeros_like(dist2)

        return sed0, dummy_src_id0, doa0,  dist0, on_screen0, sed1, dummy_src_id1, doa1,  dist1, on_screen1, sed2, dummy_src_id2, doa2,  dist2, on_screen2

    else:
        x0, y0 = logits[:, :, :1 * nb_classes], logits[:, :, 1 * nb_classes:2 * nb_classes]
        sed0 = torch.sqrt(x0 ** 2 + y0 ** 2) > 0.5
        dist0 = logits[:, :, 2 * nb_classes:3 * nb_classes]
        dist0[dist0 < 0.] = 0
        doa0 = logits[:, :, :2 * nb_classes]
        dummy_src_id0 = torch.zeros_like(dist0)
        on_screen0 = logits[:, :, 3 * nb_classes:4 * nb_classes]

        x1, y1 = logits[:, :, 4 * nb_classes:5 * nb_classes], logits[:, :, 5 * nb_classes: 6 * nb_classes]
        sed1 = torch.sqrt(x1 ** 2 + y1 ** 2) > 0.5
        dist1 = logits[:, :, 6 * nb_classes:7 * nb_classes]
        dist1[dist1 < 0.] = 0
        doa1 = logits[:, :, 4 * nb_classes:6 * nb_classes]
        dummy_src_id1 = torch.zeros_like(dist1)
        on_screen1 = logits[:, :, 7 * nb_classes:8 * nb_classes]

        x2, y2 = logits[:, :, 8 * nb_classes:9 * nb_classes], logits[:, :, 9 * nb_classes:10 * nb_classes]
        sed2 = torch.sqrt(x2 ** 2 + y2 ** 2) > 0.5
        dist2 = logits[:, :, 10 * nb_classes:11 * nb_classes]
        dist2[dist2 < 0.] = 0
        doa2 = logits[:, :, 8 * nb_classes:10 * nb_classes]
        dummy_src_id2 = torch.zeros_like(dist2)
        on_screen2 = logits[:, :, 11 * nb_classes:12 * nb_classes]

        return sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2


def get_output_dict_format_single_accdoa(sed, src_id, x, y, dist, onscreen, convert_to_polar=True):
    output_dict = {}
    for frame_cnt in range(sed.shape[0]):
        for class_cnt in range(sed.shape[1]):
            if sed[frame_cnt][class_cnt] > 0.5:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                output_dict[frame_cnt].append([class_cnt, src_id[frame_cnt][class_cnt], x[frame_cnt][class_cnt], y[frame_cnt][class_cnt], dist[frame_cnt][class_cnt], onscreen[frame_cnt][class_cnt]])

    if convert_to_polar:
        output_dict = convert_cartesian_to_polar(output_dict)
    return output_dict


def distance_between_cartesian_coordinates(x1, y1, x2, y2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + 1e-10)
    x1, y1, x2, y2 = x1/N1, y1/N1, x2/N2, y2/N2

    # Compute the distance
    dist = x1*x2 + y1*y2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def fold_az_angle(az):
    """
    Project azimuth angle into the range [-90, 90]

    :param az: azimuth angle in degrees
    :return: folded angle in degrees
    """
    # Fold az angles
    az = (az + 180) % 360 - 180  # Make sure az is in the range [-180, 180)
    az_fold = az.copy()
    az_fold[np.logical_and(-180 <= az, az < -90)] = -180 - az[np.logical_and(-180 <= az, az < -90)]
    az_fold[np.logical_and(90 < az, az <= 180)] = 180 - az[np.logical_and(90 < az, az <= 180)]
    return az_fold


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def get_output_dict_format_multi_accdoa(sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2, thresh_unify, nb_classes, convert_to_polar=True):
    output_dict = {}
    for frame_cnt in range(sed0.shape[0]):
        for class_cnt in range(sed0.shape[1]):
            flag_0sim1 = determine_similar_location(sed0[frame_cnt][class_cnt], sed1[frame_cnt][class_cnt], doa0[frame_cnt], doa1[frame_cnt], class_cnt, thresh_unify, nb_classes)
            flag_1sim2 = determine_similar_location(sed1[frame_cnt][class_cnt], sed2[frame_cnt][class_cnt], doa1[frame_cnt], doa2[frame_cnt], class_cnt, thresh_unify, nb_classes)
            flag_2sim0 = determine_similar_location(sed2[frame_cnt][class_cnt], sed0[frame_cnt][class_cnt], doa2[frame_cnt], doa0[frame_cnt], class_cnt, thresh_unify, nb_classes)

            # unify or not unify according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                if sed0[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id0[frame_cnt][class_cnt],
                                                   doa0[frame_cnt][class_cnt],
                                                   doa0[frame_cnt][class_cnt + nb_classes],
                                                   dist0[frame_cnt][class_cnt],
                                                   on_screen0[frame_cnt][class_cnt]])

                if sed1[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id1[frame_cnt][class_cnt],
                                                   doa1[frame_cnt][class_cnt],
                                                   doa1[frame_cnt][class_cnt + nb_classes],
                                                   dist1[frame_cnt][class_cnt],
                                                   on_screen1[frame_cnt][class_cnt]])

                if sed2[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id2[frame_cnt][class_cnt],
                                                   doa2[frame_cnt][class_cnt],
                                                   doa2[frame_cnt][class_cnt + nb_classes],
                                                   dist2[frame_cnt][class_cnt],
                                                   on_screen2[frame_cnt][class_cnt]])

            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if flag_0sim1:
                    if sed2[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id2[frame_cnt][class_cnt],
                                                       doa2[frame_cnt][class_cnt],
                                                       doa2[frame_cnt][class_cnt + nb_classes],
                                                       dist2[frame_cnt][class_cnt],
                                                       on_screen2[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt]) / 2
                    dist_pred_fc = (dist0[frame_cnt] + dist1[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen0[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id0[frame_cnt]
                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nb_classes],dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

                elif flag_1sim2:
                    if sed0[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id0[frame_cnt][class_cnt],
                                                       doa0[frame_cnt][class_cnt],
                                                       doa0[frame_cnt][class_cnt + nb_classes],
                                                       dist0[frame_cnt][class_cnt],
                                                       on_screen0[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa1[frame_cnt] + doa2[frame_cnt]) / 2
                    dist_pred_fc = (dist1[frame_cnt] + dist2[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen1[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id1[frame_cnt]

                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt],
                         doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

                elif flag_2sim0:
                    if sed1[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id1[frame_cnt][class_cnt],
                                                       doa1[frame_cnt][class_cnt],
                                                       doa1[frame_cnt][class_cnt + nb_classes],
                                                       dist1[frame_cnt][class_cnt],
                                                       on_screen1[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa2[frame_cnt] + doa0[frame_cnt]) / 2
                    dist_pred_fc = (dist2[frame_cnt] + dist0[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen2[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id2[frame_cnt]

                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt],
                         doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt] + doa2[frame_cnt]) / 3
                dist_pred_fc = (dist0[frame_cnt] + dist1[frame_cnt] + dist2[frame_cnt]) / 3

                dummy_src_id_pred_fc = dummy_src_id0[frame_cnt]
                on_screen_pred_fc = on_screen0[frame_cnt]  # TODO: How to do this?

                output_dict[frame_cnt].append(
                    [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

    if convert_to_polar:
        output_dict = convert_cartesian_to_polar(output_dict)
    return output_dict


def write_to_dcase_output_format(output_dict, output_dir, filename, split, convert_dist_to_cm=True):
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    file_path = os.path.join(output_dir,split, filename)
    with open(file_path, 'w') as f:
        f.write('frame,class,source,azimuth,distance,onscreen\n')
        # Write data
        for frame_ind, values in output_dict.items():
            for value in values:
                azimuth_rounded = round(float(value[2]))
                dist_rounded = round(float(value[3]) * 100) if convert_dist_to_cm else round(float(value[3]))
                f.write(f"{int(frame_ind)},{int(value[0])},{int(value[1])},{azimuth_rounded},{dist_rounded},{int(value[4])}\n")


def write_logits_to_dcase_format(logits, params, output_dir, filelist, split='dev-test'):
    if not params['multiACCDOA']:
        sed, dummy_src_id, x, y, dist, onscreen = get_accdoa_labels(logits, params['nb_classes'], params['modality'])
        for i in range(sed.size(0)):
            sed_i, dummy_src_id_i, x_i, y_i, dist_i, onscreen_i = sed[i].cpu().numpy(), dummy_src_id[i].cpu().numpy(), x[i].cpu().numpy(), y[i].cpu().numpy(), dist[i].cpu().numpy(), onscreen[i].cpu().numpy()
            output_dict = get_output_dict_format_single_accdoa(sed_i, dummy_src_id_i, x_i, y_i, dist_i, onscreen_i, convert_to_polar=True)
            write_to_dcase_output_format(output_dict, output_dir, os.path.basename(filelist[i])[:-3] + '.csv', split)

    else:
        (sed0, dummy_src_id0, doa0, dist0, on_screen0,
         sed1, dummy_src_id1, doa1, dist1, on_screen1,
         sed2, dummy_src_id2, doa2, dist2, on_screen2) = get_multiaccdoa_labels(logits, params['nb_classes'], params['modality'])

        for i in range(sed0.size(0)):
            sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i = sed0[i].cpu().numpy(), dummy_src_id0[i].cpu().numpy(), doa0[i].cpu().numpy(), dist0[i].cpu().numpy(), on_screen0[i].cpu().numpy()
            sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i = sed1[i].cpu().numpy(), dummy_src_id1[i].cpu().numpy(), doa1[i].cpu().numpy(), dist1[i].cpu().numpy(), on_screen1[i].cpu().numpy()
            sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i = sed2[i].cpu().numpy(), dummy_src_id2[i].cpu().numpy(), doa2[i].cpu().numpy(), dist2[i].cpu().numpy(), on_screen2[i].cpu().numpy()

            output_dict = get_output_dict_format_multi_accdoa(sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i,
                                                              sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i,
                                                              sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i, params['thresh_unify'], params['nb_classes'], convert_to_polar=True)
            write_to_dcase_output_format(output_dict, output_dir, os.path.basename(filelist[i])[:-3] + '.csv', split)


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of azimuth angles. Given a set of ground truth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of ground truth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        ground truth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list: list of ground-truth azimuth angles in degrees
        :param pred_list: list of predicted azimuth angles in degrees
        :return: cost - azimuth distance (after folding them to the range [-90, 90])
        :return: row_ind - row indexes obtained from the Hungarian algorithm
        :return: col_ind - column indexes obtained from the Hungarian algorithm
    """
    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        az1, az2 = gt_list[ind_pairs[:, 0]], pred_list[ind_pairs[:, 1]]
        distances_ang = np.abs(fold_az_angle(az1) - fold_az_angle(az2))
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind


def print_results(f, ang_error, dist_error, rel_dist_error, onscreen_acc, class_wise_scr, params):
    use_jackknife = params['use_jackknife']
    print('\n\n')
    print('F-score: {:0.1f}% {}'.format(
        100 * f[0] if use_jackknife else 100 * f,
        '[{:0.2f}, {:0.2f}]'.format(100 * f[1][0], 100 * f[1][1]) if use_jackknife else ''
    ))

    print('DOA error: {:0.1f} {}'.format(
        ang_error[0] if use_jackknife else ang_error,
        '[{:0.2f}, {:0.2f}]'.format(ang_error[1][0], ang_error[1][1]) if use_jackknife else ''
    ))

    print('Distance error: {:0.2f} {}'.format(
        dist_error[0] if use_jackknife else dist_error,
        '[{:0.2f}, {:0.2f}]'.format(dist_error[1][0], dist_error[1][1]) if use_jackknife else ''
    ))
    print('Relative distance error: {:0.2f} {}'.format(
        rel_dist_error[0] if use_jackknife else rel_dist_error,
        '[{:0.2f}, {:0.2f}]'.format(rel_dist_error[1][0], rel_dist_error[1][1]) if use_jackknife else ''
    ))

    if params['modality'] == 'audio_visual':
        print('Onscreen accuracy: {:0.1f}% {}'.format(
            100 * onscreen_acc[0] if use_jackknife else 100 * onscreen_acc,
            '[{:0.2f}, {:0.2f}]'.format(100 * onscreen_acc[1][0],
                                        100 * onscreen_acc[1][1]) if use_jackknife else ''
        ))

    if params['average'] == 'macro':
        print('Class-wise results on unseen data:')

        if params['modality'] == 'audio_visual':
            print('Class\tF-score\tDOA-Error\tDist-Error\tRelDist-Error\tOnscreenAcc.')
        else:
            print('Class\tF-score\tDOA-Error\tDist-Error\tRelDist-Error')

        for cls_cnt in range(params['nb_classes']):
            if params['modality'] == 'audio_visual':
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                    cls_cnt,
                    class_wise_scr[0][0][cls_cnt] if use_jackknife else class_wise_scr[0][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][0][cls_cnt][0],
                                                class_wise_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][1][cls_cnt] if use_jackknife else class_wise_scr[1][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][1][cls_cnt][0],
                                                class_wise_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][2][cls_cnt] if use_jackknife else class_wise_scr[2][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][2][cls_cnt][0],
                                                class_wise_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][3][cls_cnt] if use_jackknife else class_wise_scr[3][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][3][cls_cnt][0],
                                                class_wise_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][4][cls_cnt] if use_jackknife else class_wise_scr[4][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][4][cls_cnt][0],
                                                class_wise_scr[1][4][cls_cnt][1]) if use_jackknife else ''
                ))
            else:
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                    cls_cnt,
                    class_wise_scr[0][0][cls_cnt] if use_jackknife else class_wise_scr[0][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][0][cls_cnt][0],
                                                class_wise_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][1][cls_cnt] if use_jackknife else class_wise_scr[1][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][1][cls_cnt][0],
                                                class_wise_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][2][cls_cnt] if use_jackknife else class_wise_scr[2][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][2][cls_cnt][0],
                                                class_wise_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    class_wise_scr[0][3][cls_cnt] if use_jackknife else class_wise_scr[3][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(class_wise_scr[1][3][cls_cnt][0],
                                                class_wise_scr[1][3][cls_cnt][1]) if use_jackknife else ''
                ))
