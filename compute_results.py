"""
metrics.py

This module defines the class to run the evaluation of the SELD scores
comparing the ground-truth files in ref_files_folder with the results
files in pred_files_path.

Author: David Diaz-Guerra, Audio Research Group, Tampere University
Date: March 2025
"""

import os
from metrics import SELDMetrics
import parameters
import numpy as np
from utils import jackknife_estimation, load_labels, organize_labels


class ComputeSELDResults(object):
    def __init__(self, params, ref_files_folder=None):
        '''
        This class takes care of computing the SELD scores from the reference and predicted csv files.

        :param params: Dictionary containing the parameters of the SELD evaluation.
        :param ref_files_folder: Folder containing the split folders with the reference csv files.
        '''
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['dataset_dir'],
                                                                                            'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']
        self._dist_thresh = params['lad_dist_thresh']
        self._reldist_thresh = params['lad_reldist_thresh']
        self._req_onscreen = params['lad_req_onscreen']
        self._max_polyphony = params['max_polyphony']

        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = load_labels(os.path.join(self._desc_dir, split, ref_file), convert_to_cartesian=False)  # TODO: Last year this would have been true? why?
                nb_ref_frames = max(list(gt_dict.keys())) if len(gt_dict) > 0 else 0
                self._ref_labels[ref_file] = [organize_labels(gt_dict, nb_ref_frames),
                                              nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']
        self._nb_classes = params['nb_classes']

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        '''
        Compute the SELD scores for the predicted csv files in a given folder.

        :param pred_files_path: Folder containing the predicted csv files.
        :param is_jackknife: Whether to compute the Jackknife confidence intervals.
        '''
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELDMetrics(doa_threshold=self._doa_thresh, req_onscreen=self._req_onscreen,
                           dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh,
                           nb_classes=self._nb_classes, average=self._average)
        pred_labels_dict = {}
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = load_labels(os.path.join(pred_files_path, pred_file), convert_to_cartesian=False)  # TODO: Last year this would have been true? why?
            nb_pred_frames = max(list(pred_dict.keys())) if len(pred_dict) > 0 else 0
            nb_ref_frames = self._ref_labels[pred_file][1]
            pred_labels = organize_labels(pred_dict, max(nb_pred_frames, nb_ref_frames))
            # pred_labels[frame-index][class-index][track-index] := [azimuth, distance, onscreen]
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        F, AngE, DistE, RelDistE, OnscreenAq, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [F, AngE, DistE, RelDistE, OnscreenAq]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                eval = SELDMetrics(doa_threshold=self._doa_thresh, req_onscreen=self._req_onscreen,
                                   dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh,
                                   nb_classes=self._nb_classes, average=self._average)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                F, AngE, DistE, RelDistE, OnscreenAq, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [F, AngE, DistE, RelDistE, OnscreenAq, classwise_results]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)

            estimate, bias = [-1] * len(global_values), [-1] * len(global_values)
            std_err, conf_interval = [-1] * len(global_values), [-1] * len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                    global_value=global_values[i],
                    partial_estimates=partial_estimates[:, i],
                    significance_level=0.05
                )
            return ([F, conf_interval[0]], [AngE, conf_interval[1]], [DistE, conf_interval[2]],
                    [RelDistE, conf_interval[3]], [OnscreenAq, conf_interval[4]],
                    [classwise_results, np.array(conf_interval)[5:].reshape(5, 13, 2) if len(classwise_results) else []])

        else:
            return (F, AngE, DistE, RelDistE, OnscreenAq, classwise_results)


if __name__ == "__main__":
    pred_output_files = 'outputs/SELD_fake_estimates/dev-test-tau'  # Path of the DCASEoutput format files
    params = parameters.params
    # Compute just the DCASE final results
    use_jackknife = False
    eval_dist = params['evaluate_distance'] if 'evaluate_distance' in params else False
    score_obj = ComputeSELDResults(params, ref_files_folder='../DCASE2025_SELD_dataset/metadata_simple_header_int_dev')
    F, AngE, DistE, RelDistE, OnscreenAq, classwise_test_scr = score_obj.get_SELD_Results(pred_output_files,
                                                                                          is_jackknife=use_jackknife)
    print('SED F-score: {:0.1f}% {}'.format(100 * F[0] if use_jackknife else 100 * F,
                                            '[{:0.2f}, {:0.2f}]'.format(100 * F[1][0], 100 * F[1][1])
                                                                                             if use_jackknife else ''))
    print('DOA error: {:0.1f} {}'.format(AngE[0] if use_jackknife else AngE,
                                         '[{:0.2f}, {:0.2f}]'.format(AngE[1][0], AngE[1][1])
                                                                                            if use_jackknife else ''))
    print('Distance metrics: Distance error: {:0.2f} {}, Relative distance error: {:0.2f} {}'.format(
                DistE[0] if use_jackknife else DistE,
                '[{:0.2f}, {:0.2f}]'.format(DistE[1][0], DistE[1][1]) if use_jackknife else '',
                RelDistE[0] if use_jackknife else RelDistE,
                '[{:0.2f}, {:0.2f}]'.format(RelDistE[1][0], RelDistE[1][1]) if use_jackknife else '')
        )
    print('Onscreen accuracy: {:0.1f}% {}'.format(100 * OnscreenAq[0] if use_jackknife else 100 * OnscreenAq,
                                                  '[{:0.2f}, {:0.2f}]'.format(100 * OnscreenAq[1][0], 100 * OnscreenAq[1][1])
                                                                                             if use_jackknife else ''))
    if params['average'] == 'macro':
        print('Classwise results on unseen test data')
        print('Class\tF\tAngE\tDistE\tRelDistE\tOnscreenAq')
        for cls_cnt in range(params['nb_classes']):
            print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                cls_cnt,
                classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                            classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))

