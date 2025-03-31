"""
metrics.py

This module defines evaluation metrics to monitor the model's performance
during training and testing.

Author: David Diaz-Guerra, Audio Research Group, Tampere University
Date: February 2025
"""


import numpy as np
from utils import least_distance_between_gt_pred, jackknife_estimation, load_labels, organize_labels
import os
import warnings


class SELDMetrics(object):

    def __init__(self, doa_threshold=20, dist_threshold=np.inf, reldist_threshold=np.inf, req_onscreen=True,
                 nb_classes=13, average='macro'):
        """
        This class implements both the class-sensitive localization and location-sensitive detection metrics.

        :param doa_threshold: DOA error threshold for location sensitive detection.
        :param dist_threshold: Distance error threshold for location sensitive detection.
        :param reldist_threshold: Relative distance error threshold for location sensitive detection.
        :param req_onscreen: Require correct onscreen estimation for localization sensitive detection.
        :param nb_classes: Number of sound classes.
        :param average: Whether 'macro' or 'micro' aggregate the results
        """
        self._nb_classes = nb_classes

        # Variables for Location-sensitive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._ang_T = doa_threshold
        self._dist_T = dist_threshold
        self._reldist_T = reldist_threshold
        self._req_onscreen = req_onscreen

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_AngE = np.zeros(self._nb_classes)
        self._total_DistE = np.zeros(self._nb_classes)
        self._total_RelDistE = np.zeros(self._nb_classes)
        self._total_OnscreenCorrect = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        assert average in ['macro', 'micro'], "Only 'micro' and 'macro' average are supported"
        self._average = average

    def compute_seld_scores(self):
        """
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores:
            F score, angular error, distance error, relative distance error, onscreen accuracy, and classwise results
        """
        eps = np.finfo(float).eps
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (
                        eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            AngE = self._total_AngE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan
            DistE = self._total_DistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan
            RelDistE = self._total_RelDistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan
            OnscreenAq = self._total_OnscreenCorrect.sum() / float(
                self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            AngE = self._total_AngE / (self._DE_TP + eps)
            AngE[self._DE_TP == 0] = np.nan
            DistE = self._total_DistE / (self._DE_TP + eps)
            DistE[self._DE_TP == 0] = np.nan
            RelDistE = self._total_RelDistE / (self._DE_TP + eps)
            RelDistE[self._DE_TP == 0] = np.nan
            OnscreenAq = self._total_OnscreenCorrect / (self._DE_TP + eps)
            OnscreenAq[self._DE_TP == 0] = np.nan

            classwise_results = np.array([F, AngE, DistE, RelDistE, OnscreenAq])
            F, AngE = F.mean(), np.nanmean(AngE)
            DistE, RelDistE = np.nanmean(DistE), np.nanmean(RelDistE)
            OnscreenAq = np.nanmean(OnscreenAq)

        else:
            raise NotImplementedError('Only micro and macro averaging are supported.')

        return F, AngE, DistE, RelDistE, OnscreenAq, classwise_results

    def update_seld_scores(self, pred, gt):
        """
        Computes the SELD scores given a prediction and ground truth labels.

        :param pred: dictionary containing the predictions for every frame
            pred[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        :param gt: dictionary containing the ground truth for every frame
            gt[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        """
        eps = np.finfo(float).eps

        for frame_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of reference tracks for each class
                nb_gt_doas = len(gt[frame_cnt][class_cnt]) if class_cnt in gt[frame_cnt] else None
                nb_pred_doas = len(pred[frame_cnt][class_cnt]) if class_cnt in pred[frame_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # True positives

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm on the azimuth estimation and then compute the average
                    # spatial distance between the associated reference-predicted tracks.

                    gt_values = np.array(list(gt[frame_cnt][class_cnt].values()))
                    gt_az, gt_dist, gt_onscreeen = gt_values[:, 0], gt_values[:, 1], gt_values[:, 2]
                    pred_values = np.array(list(pred[frame_cnt][class_cnt].values()))
                    pred_az, pred_dist, pred_onscreeen = pred_values[:, 0], pred_values[:, 1], pred_values[:, 2]

                    # Reference and predicted track matching
                    doa_err_list, row_inds, col_inds = least_distance_between_gt_pred(gt_az, pred_az)
                    dist_err_list = np.abs(gt_dist[row_inds] - pred_dist[col_inds])
                    rel_dist_err_list = dist_err_list / (gt_dist[row_inds] + eps)
                    onscreen_correct_list = (gt_onscreeen[row_inds] == pred_onscreeen[col_inds])

                    # https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#evaluation
                    Pc = len(pred_az)
                    Rc = len(gt_az)
                    FNc = max(0, Rc - Pc)
                    FPcinf = max(0, Pc - Rc)
                    Kc = min(Pc, Rc)
                    TPc = Kc
                    Lc = np.sum(np.any((doa_err_list > self._ang_T,
                                        dist_err_list > self._dist_T,
                                        rel_dist_err_list > self._reldist_T,
                                        np.logical_and(np.logical_not(onscreen_correct_list), self._req_onscreen)),
                                       axis=0))
                    FPct = Lc
                    FPc = FPcinf + FPct
                    TPct = Kc - FPct
                    assert Pc == TPct + FPc
                    assert Rc == TPct + FPct + FNc

                    self._total_AngE[class_cnt] += doa_err_list.sum()
                    self._total_DistE[class_cnt] += dist_err_list.sum()
                    self._total_RelDistE[class_cnt] += rel_dist_err_list.sum()
                    self._total_OnscreenCorrect[class_cnt] += onscreen_correct_list.sum()

                    self._TP[class_cnt] += TPct
                    self._DE_TP[class_cnt] += TPc

                    self._FP[class_cnt] += FPcinf
                    self._DE_FP[class_cnt] += FPcinf
                    self._FP_spatial[class_cnt] += FPct
                    loc_FP += FPc

                    self._FN[class_cnt] += FNc
                    self._DE_FN[class_cnt] += FNc
                    loc_FN += FNc

                elif class_cnt in gt[frame_cnt] and class_cnt not in pred[frame_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas
                else:
                    # True negative
                    pass


class ComputeSELDResults(object):
    def __init__(self, params, ref_files_folder=None):
        """
        This class takes care of computing the SELD scores from the reference and predicted csv files.

        :param params: Dictionary containing the parameters of the SELD evaluation.
        :param ref_files_folder: Folder containing the split folders with the reference csv files.
        """
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['root_dir'],
                                                                                            'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']
        self._dist_thresh = params['lad_dist_thresh']
        self._reldist_thresh = params['lad_reldist_thresh']
        self._req_onscreen = params['lad_req_onscreen']

        if params['modality'] == 'audio' and params['lad_req_onscreen']:
            warnings.warn("'lad_req_onscreen' is set to True, but 'modality' is 'audio'. "
                          "Onscreen estimation for detection metrics is not applicable to an audio-only model. "
                          "Resetting 'lad_req_onscreen' To False.")
            self._req_onscreen = False

        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = load_labels(os.path.join(self._desc_dir, split, ref_file), convert_to_cartesian=False)
                nb_ref_frames = max(list(gt_dict.keys())) if len(gt_dict) > 0 else 0
                self._ref_labels[ref_file] = [organize_labels(gt_dict, nb_ref_frames),
                                              nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']
        self._nb_classes = params['nb_classes']

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        """
        Compute the SELD scores for the predicted csv files in a given folder.

        :param pred_files_path: Folder containing the predicted csv files.
        :param is_jackknife: Whether to compute the Jackknife confidence intervals.
        """
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELDMetrics(doa_threshold=self._doa_thresh, req_onscreen=self._req_onscreen,
                           dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh,
                           nb_classes=self._nb_classes, average=self._average)
        pred_labels_dict = {}
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = load_labels(os.path.join(pred_files_path, pred_file), convert_to_cartesian=False)
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


if __name__ == '__main__':
    # use this to test if the metrics class works as expected. All the classes will be called from the main.py for
    # actual use
    pred_output_files = 'outputs/SELD_fake_estimates/dev-test'  # Path of the DCASE output format files
    from parameters import params
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
                                                  '[{:0.2f}, {:0.2f}]'.format(100 * OnscreenAq[1][0],
                                                                              100 * OnscreenAq[1][1])
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


