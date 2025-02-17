"""
parameters.py

This module stores all the configurable parameters and hyperparameters used
across the project, ensuring easy tuning and reproducibility.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

params = {

    # choose task
    'modality': 'audio_visual',  # 'audio' or audio_visual'

    # data params
    'root_dir': '../DCASE2025_SELD_dataset',  # parent directory containing the audio, video and labels directory
    'feat_dir': '../DCASE2025_SELD_dataset/features',  # store extracted features here "root_dir/feat_dir"

    'log_dir': 'logs',  # save all logs here like loss and metrics
    'output_dir': 'outputs',  # save the predicted files here.

    # audio feature extraction params
    'sampling_rate': 24000,
    'hop_length_s': 0.02,
    'nb_mels': 64,

    # video feature extraction params
    'fps': 10,
    'resnet_feature_size': 49,  # (7,7) feature_map for every frame from resnet

    # model params
    'nb_conv_blocks': 3,
    'nb_conv_filters': 64,
    'f_pool_size': [4, 4, 2],
    't_pool_size': [5, 1, 1],
    'dropout': 0.05,

    'rnn_size': 128,
    'nb_rnn_layers': 2,

    'nb_self_attn_layers': 2,
    'nb_attn_heads': 8,

    'nb_transformer_layers': 2,

    'nb_fnn_layers': 1,
    'fnn_size':128,

    'max_polyphony': 3,  # for multiaccdoa
    'nb_classes': 13,
    'label_sequence_length': 50,  # 5 seconds with 100ms frames

    # loss params
    'multiACCDOA': False,

    # training params
    'nb_epochs': None,
    'batch_size': 1,
    'nb_workers': 0,
    'shuffle': True,


    # optimizer params
    'learning_rate': None,
    'weight_decay': None,

    # folds for training, testing
    'dev_train_folds': ['fold3'],
    'dev_test_folds': ['fold4'],
    'dev_synth_folds': ['fold1, fold2'],

}
