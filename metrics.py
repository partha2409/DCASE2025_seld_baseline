"""
metrics.py

This module defines evaluation metrics to monitor the model's performance
during training and testing.

Author: David Diaz-Guerra, Audio Research Group, Tampere University
Date: March 2025
"""

import numpy as np
from utils import least_distance_between_gt_pred


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

    def __init__(self, doa_threshold=20, dist_threshold=np.inf, reldist_threshold=np.inf, req_onscreen=True,
                 nb_classes=13, average='macro'):
        '''
        This class implements both the class-sensitive localization and location-sensitive detection metrics.

        :param doa_threshold: DOA error threshold for location sensitive detection.
        :param dist_threshold: Distance error threshold for location sensitive detection.
        :param reldist_threshold: Relative distance error threshold for location sensitive detection.
        :param req_onscreen: Require correct onscreen estimation for localization sensitive detection.
        :param nb_classes: Number of sound classes.
        :param average: Whether 'macro' or 'micro' aggregate the results
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
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
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores:
            F score, angular error, distance error, relative distance error, onscreen accuracy, and classwise results
        '''
        eps = np.finfo(float).eps
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            AngE = self._total_AngE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN
            DistE = self._total_DistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN
            RelDistE = self._total_RelDistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN
            OnscreenAq = self._total_OnscreenCorrect.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.NaN

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            AngE = self._total_AngE / (self._DE_TP + eps)
            AngE[self._DE_TP==0] = np.NaN
            DistE = self._total_DistE / (self._DE_TP + eps)
            DistE[self._DE_TP==0] = np.NaN
            RelDistE = self._total_RelDistE / (self._DE_TP + eps)
            RelDistE[self._DE_TP==0] = np.NaN
            OnscreenAq = self._total_OnscreenCorrect / (self._DE_TP + eps)
            OnscreenAq[self._DE_TP==0] = np.NaN

            classwise_results = np.array([F, AngE, DistE, RelDistE, OnscreenAq])
            F, AngE = F.mean(), np.nanmean(AngE)
            DistE, RelDistE = np.nanmean(DistE), np.nanmean(RelDistE)
            OnscreenAq = np.nanmean(OnscreenAq)

        else:
            raise NotImplementedError('Only micro and macro averaging are supported.')
            
        return F, AngE, DistE, RelDistE, OnscreenAq, classwise_results

    def update_seld_scores(self, pred, gt):
        '''
        Computes the SELD scores given a prediction and ground truth labels.

        :param pred: dictionary containing the predictions for every frame
            pred[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        :param gt: dictionary containing the groundtruth for every frame
            gt[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        '''
        eps = np.finfo(float).eps

        for frame_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class
                nb_gt_doas = len(gt[frame_cnt][class_cnt]) if class_cnt in gt[frame_cnt] else None
                nb_pred_doas = len(pred[frame_cnt][class_cnt]) if class_cnt in pred[frame_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # True positives or False positive case         TODO: Is FP really included in this case?

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


if __name__ == '__main__':
    pass
    # use this to test if the metrics class works as expected. All the classes will be called from the main.py for
    # actual use

