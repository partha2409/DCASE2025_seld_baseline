"""
inference.py

This module provides utilities for running inference using the trained model,

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: March 2025
"""

import utils
from model import SELDModel
import pickle
import os
import torch
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader


def run_inference():

    params_file = os.path.join(model_dir, 'config.pkl')
    f = open(params_file, "rb")
    params = pickle.load(f)

    reference = model_dir.split('/')[-1]
    output_dir = os.path.join(params['output_dir'], reference)

    seld_model = SELDModel(params).to(device)
    model_ckpt = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device, weights_only=False)
    seld_model.load_state_dict(model_ckpt['seld_model'])

    seld_metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

    test_dataset = DataGenerator(params=params, mode='dev_test')
    test_iterator = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=False, drop_last=False)

    seld_model.eval()
    with torch.no_grad():
        for j, (input_features, labels) in enumerate(test_iterator):
            labels = labels.to(device)
            # Handling modalities
            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            # Forward pass
            logits = seld_model(audio_features, video_features)

            # save predictions to csv files for metric calculations
            utils.write_logits_to_dcase_format(logits, params, output_dir, test_iterator.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']])

        test_metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=False)
        test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
        utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':

    model_dir = "checkpoints/SELDnet_audio_multiACCDOA_20250330_170742"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    run_inference()