"""
loss.py

This module implements the loss functions used for training the model.

Author: Kazuki Shimada, Sony Corporation, Tokyo
Date: February 2025
"""

import torch
import torch.nn as nn


class SELDLossADPIT(nn.Module):
    """
      General idea: Choose the permutation with the lowest MSE loss based on doa. Then use the corresponding dist and on/off pred as it is.
      output:
           audio (batch_size, 50, 117) -> 117 = 3 (tracks) x 3 (x,y,dist), 13 (classes)
           audio_visual (batch_size, 50, 156) -> 156 = 3 (tracks) x 4 (x,y,dist, on/off), 13 (classes)
      target:
           audio (batch_size, 50, 6, 4, 13) -> 6 (tracks) x 4 (sed, x, y, dist), 13 (classes)
           audio_visual (batch_size, 50, 6, 5, 13) -> 6 (tracks) x 5 (sed, x, y, dist,on/off), 13 (classes)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.modality = params['modality']

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')

    def _mse(self, output, target):
        return self.mse_loss(output, target).mean(dim=2)  # class-wise frame-level

    def _bce(self, output, target):
        return self.bce_loss(output, target).mean(dim=2)  # class-wise frame-level

    def _make_target_perms(self, target_A0, target_B0, target_B1, target_C0, target_C1, target_C2):
        """
        Make 13 (=1+6+6) possible target permutations

        Args:
            target_XX: (batch_size, frames, n (elements), classes): e.g., (batch_size, frames, 2 (act*x, act*y), classes)
        Return:
            list_target_perms: an element is (batch_size, frames, 3 (tracks) x n (elements), classes)
        """
        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        pad4A = target_B0B0B1 + target_C0C1C2  # pad for target_A0A0A0 using target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        pad4B = target_A0A0A0 + target_C0C1C2  # pad for target_BxBxBx using target_A0A0A0 and target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1  # pad for target_CxCxCx using target_A0A0A0 and target_B0B0B1

        list_target_perms = [target_A0A0A0 + pad4A,
                             target_B0B0B1 + pad4B, target_B0B1B0 + pad4B, target_B0B1B1 + pad4B,
                             target_B1B0B0 + pad4B, target_B1B0B1 + pad4B, target_B1B1B0 + pad4B,
                             target_C0C1C2 + pad4C, target_C0C2C1 + pad4C, target_C1C0C2 + pad4C,
                             target_C1C2C0 + pad4C, target_C2C0C1 + pad4C, target_C2C1C0 + pad4C]
        return list_target_perms

    def forward(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible permutations

        Args:
            output:
                audio:        (batch_size, 50, 117) -> 117 = 3 (tracks) x 3 (act*x, act*y, dist) x 13 (classes)
                audio_visual: (batch_size, 50, 156) -> 156 = 3 (tracks) x 4 (act*x, act*y, dist, on/off) x 13 (classes)
            target:
                audio:        (batch_size, 50, 6, 4, 13) -> 6 (dummy tracks), 4 (act, x, y, dist), 13 (classes)
                audio_visual: (batch_size, 50, 6, 5, 13) -> 6 (dummy tracks), 5 (act, x, y, dist, on/off), 13 (classes)
        Return:
            loss: scalar
        """

        # preliminaries for reshaping output
        num_bs = target.shape[0]
        num_frame = target.shape[1]
        num_track = 3
        num_element = target.shape[3] - 1  # E.g., 4 (act, x, y, dist) for target -> 3 (act*x, act*y, dist) for output
        num_class = target.shape[4]
        num_permutation = 13

        # calculate ACCDOA loss for each target permutation and store them to list_loss_accdoa
        output_accdoa = output.reshape(num_bs, num_frame, num_track, num_element, num_class)[:, :, :, 0:3, :]  # use accdoa + dist elements
        output_accdoa = output_accdoa.reshape(num_bs, num_frame, -1, num_class)  # the same shape of each target_accdoa permutation: (batch_size, frames, 3 (tracks) x 3 (act*x, act*y, dist), classes)
        # NOTE: these reshapes are from my temporal assumption, we need to check how to reshape output
        target_accdoa_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:4, :]  # A0, no ov from the same class: (batch_size, frames, 1 (act), classes) * (batch_size, frames, 3 (x, y, dist), classes)
        target_accdoa_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:4, :]  # B0, ov with 2 sources from the same class
        target_accdoa_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:4, :]  # B1
        target_accdoa_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:4, :]  # C0, ov with 3 sources from the same class
        target_accdoa_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:4, :]  # C1
        target_accdoa_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:4, :]  # C2
        # NOTE: I assume target consists of A0, ..., C2, we need to check how to make target
        list_target_accdoa_perms = self._make_target_perms(target_accdoa_A0, target_accdoa_B0, target_accdoa_B1,
                                                           target_accdoa_C0, target_accdoa_C1, target_accdoa_C2)
        list_loss_accdoa = []
        for each_target_accdoa in list_target_accdoa_perms:
            list_loss_accdoa.append(self._mse(output_accdoa, each_target_accdoa))

        # if audiovisual, calculate on/off loss for each target permutation and store them to list_loss_screen
        if self.modality == 'audio':
            list_loss_screen = [0,] * num_permutation  # dummy
        else:
            output_screen = output.reshape(num_bs, num_frame, num_track, num_element, num_class)[:, :, :, 3:4, :]  # use on/off element
            output_screen = output_screen.reshape(num_bs, num_frame, -1, num_class)  # the same shape of each target_screen permutation: (batch_size, frames, 3 (tracks) x 1 (on/off), classes)
            target_screen_A0 = target[:, :, 0, 4:5, :]  # (batch_size, frames, 1 (on/off), classes)
            target_screen_B0 = target[:, :, 1, 4:5, :]
            target_screen_B1 = target[:, :, 2, 4:5, :]
            target_screen_C0 = target[:, :, 3, 4:5, :]
            target_screen_C1 = target[:, :, 4, 4:5, :]
            target_screen_C2 = target[:, :, 5, 4:5, :]
            list_target_screen_perms = self._make_target_perms(target_screen_A0, target_screen_B0, target_screen_B1,
                                                               target_screen_C0, target_screen_C1, target_screen_C2)
            list_loss_screen = []
            for each_target_screen in list_target_screen_perms:
                list_loss_screen.append(self._bce(output_screen, each_target_screen))
                # NOTE: how do we set gt on/off when no source?, we may use masked bce for on/off loss

        # choose the permutation with the lowest ACCDOA loss
        loss_accdoa_min = torch.min(torch.stack(list_loss_accdoa, dim=0), dim=0).indices

        # use the corresponding dist and on/off losses in addition to the ACCDOA loss
        loss_sum = 0
        for i in range(num_permutation):
            loss_sum += (list_loss_accdoa[i] + list_loss_screen[i]) * (loss_accdoa_min == i)
        loss = loss_sum.mean()
        return loss


class SELDLossSingleACCDOA(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target):
        if self.params['modality'] == 'audio':
            loss = self.mse_loss(output, target)
        else:
            mse_loss = self.mse_loss(output[:, :, :-self.params['nb_classes']], target[:, :, :-self.params['nb_classes']])
            bce_loss = self.bce_loss(output[:, :, 3 * self.params['nb_classes']:], target[:, :, 3 * self.params['nb_classes']:])
            loss = (mse_loss + bce_loss) / 2
        return loss


if __name__ == '__main__':
    # use this to test if the loss class works as expected. The all the classes will be called from the main.py for
    # actual use
    pass
