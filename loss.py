"""
loss.py

This module implements the loss functions used for training the model.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""


# TODO : Implement ADPIT loss including on/off screen predictions.
#  General idea: Choose the permutation with the lowest MSE loss based on doa. Then use the corresponding dist and on/off pred as it is.
#  Guidelines:
#  output:
#       audio (batch_size, 50, 117) -> 117 = 3 (tracks) x 3 (x,y,dist), 13 (classes)
#       audio_visual (batch_size, 50, 156) -> 156 = 3 (tracks) x 4 (x,y,dist, on/off), 13 (classes)
#  target:
#       audio (batch_size, 50, 6, 4, 13) -> 6 (tracks) x 4 (sed, x, y, dist), 13 (classes)
#       audio_visual (batch_size, 50, 6, 5, 13) -> 6 (tracks) x 5 (sed, x, y, dist,on/off), 13 (classes)


import torch
import torch.nn as nn


class SELDLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params  # feel free to add params to the parameters.py file if something extra is needed.
        self.modality = params['modality']  # 'audio' or 'audio_visual'

        self._each_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss()


    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=2)  # class-wise frame-level

    def forward(self, output, target):
        # check the guidelines at the beginning for shapes
        pass


# class SELDLoss(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self._each_loss = nn.MSELoss(reduction='none')
#
#     def _each_calc(self, output, target):
#         return self._each_loss(output, target).mean(dim=2)  # class-wise frame-level
#
#     def forward(self, output, target):
#         """
#         Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
#         Args:
#             output: [batch_size, frames, num_track*num_axis*num_class=3*3*13]
#             target: [batch_size, frames, num_track_dummy=6, num_axis=3, num_class=13]
#         Return:
#             loss: scalar
#         """
#         target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZD)=4, num_class=12]
#         target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
#         target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
#         target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
#         target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
#         target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2
#
#         target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*4, num_class=12]
#         target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
#         target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
#         target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
#         target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
#         target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
#         target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
#         target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
#         target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
#         target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
#         target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
#         target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
#         target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)
#
#         output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*4, num_class=12]
#         pad4A = target_B0B0B1 + target_C0C1C2
#         pad4B = target_A0A0A0 + target_C0C1C2
#         pad4C = target_A0A0A0 + target_B0B0B1
#
#         loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
#         loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
#         loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
#         loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
#         loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
#         loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
#         loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
#         loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
#         loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
#         loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
#         loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
#         loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
#         loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)
#
#         loss_min = torch.min(
#             torch.stack((loss_0,
#                          loss_1,
#                          loss_2,
#                          loss_3,
#                          loss_4,
#                          loss_5,
#                          loss_6,
#                          loss_7,
#                          loss_8,
#                          loss_9,
#                          loss_10,
#                          loss_11,
#                          loss_12), dim=0),
#             dim=0).indices
#
#         loss = (loss_0 * (loss_min == 0) +
#                 loss_1 * (loss_min == 1) +
#                 loss_2 * (loss_min == 2) +
#                 loss_3 * (loss_min == 3) +
#                 loss_4 * (loss_min == 4) +
#                 loss_5 * (loss_min == 5) +
#                 loss_6 * (loss_min == 6) +
#                 loss_7 * (loss_min == 7) +
#                 loss_8 * (loss_min == 8) +
#                 loss_9 * (loss_min == 9) +
#                 loss_10 * (loss_min == 10) +
#                 loss_11 * (loss_min == 11) +
#                 loss_12 * (loss_min == 12)).mean()
#
#         return loss


if __name__ == '__main__':
    # use this to test if the loss class works as expected. The all the classes will be called from the main.py for
    # actual use
    from parameters import params
    params['multiACCDOA'] = True

    # switch between the two to test
    params['modality'] = 'audio'
    params['modality'] = 'audio_visual'

    seld_loss = SELDLoss(params)
    if params['modality'] == 'audio':
        dummy_pred = torch.rand(8, 50, 117)
        dummy_target = torch.rand(8, 50, 6, 4, 13)
    else:
        dummy_pred = torch.rand(8, 50, 156)
        dummy_target = torch.rand(8, 50, 6, 5, 13)

    loss = seld_loss(dummy_pred, dummy_target)

