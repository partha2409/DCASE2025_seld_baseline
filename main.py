"""
main.py

This is the entry point for the project. It orchestrates the training pipeline,
including data preparation, model training, and evaluation.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""


import torch
from parameters import params
from model import SELDModel
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from metrics import SELDMetrics
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor
import utils

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def main(pre_trained_model=None):

    # Set up directories for storing model checkpoints, predictions,  and create a summary writer
    checkpoints_folder, output_dir, summary_writer = utils.setup(params)

    # Feature extraction code.
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')

    # Set up dev_train and dev_test data iterator
    dev_train_dataset = DataGenerator(params=params, mode='dev_train')
    dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'], drop_last=True)

    dev_test_dataset = DataGenerator(params=params, mode='dev_test')
    dev_test_iterator = DataLoader(dataset=dev_test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=False, drop_last=True)

    # create model, optimizer, loss and metrics
    seld_model = SELDModel(params=params).to(device)
    optimizer = torch.optim.Adam(params=seld_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    if params['multiACCDOA']:
        seld_loss = SELDLossADPIT(params=params).to(device)
    else:
        seld_loss = SELDLossSingleACCDOA(params=params).to(device)

    seld_metrics = SELDMetrics(params=params)

    start_epoch = 0
    # load pretrained model if available to continue training
    if pre_trained_model is not None:
        ckpt = torch.load(pre_trained_model)
        seld_model.load_state_dict(ckpt['seld_model'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1

    # Training and validation
    for epoch in range(start_epoch, params['nb_epochs']):

        seld_model.train()
        train_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.

        for i, (input_features, labels) in enumerate(dev_train_iterator):
            optimizer.zero_grad()

            labels = labels.to(device)
            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'. You can set the modality in params.py")

            logits = seld_model(audio_features, video_features)
            loss = seld_loss(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()

        # validation loop after every epoch
        with torch.no_grad():

            seld_model.eval()
            val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.

            for j, (input_features, labels) in enumerate(dev_test_iterator):

                labels = labels.to(device)
                if params['modality'] == 'audio':
                    audio_features, video_features = input_features.to(device), None
                elif params['modality'] == 'audio_visual':
                    audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
                else:
                    raise AssertionError("Modality should be one of 'audio' or 'audio_visual'. You can set the modality in params.py")

                logits = seld_model(audio_features, video_features)
                loss = seld_loss(logits, labels)
                val_loss_per_epoch += loss.item()
                utils.write_logits_to_dcase_format(logits, params, output_dir, dev_test_dataset.label_files[j*params['batch_size']: (j+1)*params['batch_size']])
            print('epoch = {}, training loss = {:.4f}, validation loss = {:.4f}'.format(epoch + 1, train_loss_per_epoch/(i+1), val_loss_per_epoch / (j + 1)))


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main()

