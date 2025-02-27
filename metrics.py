"""
metrics.py

This module defines evaluation metrics to monitor the model's performance
during training and testing.

Author: David Diaz-Guerra, Audio Research Group, Tampere University
Date: February 2025
"""

import torch
import numpy as np


class SELDMetrics():
    """
    Guidelines:

    The GT csv files are at : '../DCASE2025_SELD_dataset/metadata_dev/' which is os.path.join(self.root_dir, 'metadata_dev').
    It contains the usual 4 sub folders. we will mainly use 'dev-test-tau' and 'dev-test-sony'

    The dev-test predictions from the model are stored at 'outputs/SELDnet_audio_visual_multiACCDOA_20250227_174427/dev-test/*.csv'
     -- > os.listdir(os.path.join(output_dir, split)) will give the csv predictions without any sub folders.

    during eval phase:
        results will be at 'outputs/SELDnet_audio_visual_multiACCDOA_20250227_174427/eval/*.csv'
        so split value will be 'eval'

    The calculate_seld_metrics function can return all the metrics you choose to have.
    add whatever parameters necessary to the parameters.py file as you wish.
    (Note: don't get confused with the output_dir in the params file. The output_dir argument in the init block here
     is 'outputs/followed_by_the_reference_name_like_the_example_above' )
    """

    def __init__(self, params, output_dir, split='dev-test'):
        self.root_dir = params['root_dir']
        self.output_dir = output_dir
        self.split = split

    def calculate_seld_metrics(self):
        pass


if __name__ == '__main__':
    pass
    # use this to test if the metrics class works as expected. All the classes will be called from the main.py for
    # actual use

