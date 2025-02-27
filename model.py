"""
model.py

This module defines the architecture of the SELD deep learning model.

Classes:
    ConvBlock: A convolutional block for feature extraction from audio input.
    SELDModel: The main SELD model combining ConvBlock, recurrent, attention, and fusion layers.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
    Designed for feature extraction from audio input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=(5, 4), dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SELDModel(nn.Module):
    """
    SELD (Sound Event Localization and Detection) model combining ConvBlock, recurrent, and attention-based layers.
    Supports audio-only and audio_visual input.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Conv layers
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            self.conv_blocks.append(ConvBlock(in_channels=params['nb_conv_filters'] if conv_cnt else 2,  # stereo
                                              out_channels=params['nb_conv_filters'],
                                              pool_size=(params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]),
                                              dropout=params['dropout']))

        # GRU layers
        self.gru_input_dim = params['nb_conv_filters'] * int(np.floor(params['nb_mels'] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'], num_layers=params['nb_rnn_layers'],
                                batch_first=True, dropout=params['dropout'], bidirectional=True)

        # Self attention layers
        self.mhsa_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=params['rnn_size'], num_heads=params['nb_attn_heads'],
                                  dropout=params['dropout'], batch_first=True) for _ in range(params['nb_self_attn_layers'])])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(params['rnn_size']) for _ in range(params['nb_self_attn_layers'])])

        # Fusion layers
        if params['modality'] == 'audio_visual':
            self.visual_embed_to_d_model = nn.Linear(in_features=params['resnet_feature_size'], out_features=params['rnn_size'])
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=params['rnn_size'], nhead=params['nb_attn_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=params['nb_transformer_layers'])

        self.fnn_list = torch.nn.ModuleList()

        for fc_cnt in range(params['nb_fnn_layers']):
            self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'], params['fnn_size'], bias=True))

        if params['multiACCDOA']:
            if params['modality'] == 'audio_visual':
                self.output_dim = params['max_polyphony'] * 4 * params['nb_classes']  # 4 => (x,y), distance, on/off
            else:
                self.output_dim = params['max_polyphony'] * 3 * params['nb_classes']  # 3 => (x,y), distance
        else:
            if params['modality'] == 'audio_visual':
                self.output_dim = 4 * params['nb_classes']  # 4 => (x,y), distance, on/off
            else:
                self.output_dim = 3 * params['nb_classes']  # 3 => (x,y), distance
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], self.output_dim, bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        if params['modality'] == 'audio_visual':
            self.onscreen_act = nn.Sigmoid()

    def forward(self, audio_feat, vid_feat=None):
        """
        Forward pass for the SELD model.
        audio_feat: Tensor of shape (batch_size, 2, 251, 64) (stereo spectrogram input).
        vid_feat: Optional tensor of shape (batch_size, 50, 7, 7) (visual feature map).
        Returns:  Tensor of shape
                  (batch_size, 50, 117) - audio - multiACCDOA.
                  (batch_size, 50, 39)  - audio - singleACCDOA.
                  (batch_size, 50, 156) - audio_visual - multiACCDOA.
                  (batch_size, 50, 52) - audio_visual - singleACCDOA.

        """
        # audio feat - B x 2 x 251 x 64
        for conv_block in self.conv_blocks:
            audio_feat = conv_block(audio_feat)  # B x 64 x 50 x 2

        audio_feat = audio_feat.transpose(1, 2).contiguous()  # B x 50 x 64 x 2
        audio_feat = audio_feat.view(audio_feat.shape[0], audio_feat.shape[1], -1).contiguous()  # B x 50 x 128

        (audio_feat, _) = self.gru(audio_feat)
        audio_feat = torch.tanh(audio_feat)
        audio_feat = audio_feat[:, :, audio_feat.shape[-1] // 2:] * audio_feat[:, :, :audio_feat.shape[-1] // 2]

        for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
            audio_feat_in = audio_feat
            audio_feat, _ = mhsa(audio_feat_in, audio_feat_in, audio_feat_in)
            audio_feat = audio_feat + audio_feat_in  # Residual connection
            audio_feat = ln(audio_feat)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            fused_feat = self.transformer_decoder(audio_feat, vid_feat)
        else:
            fused_feat = audio_feat

        for fnn_cnt in range(len(self.fnn_list) - 1):
            fused_feat = self.fnn_list[fnn_cnt](fused_feat)
        pred = self.fnn_list[-1](fused_feat)

        if self.params['modality'] == 'audio':
            if self.params['multiACCDOA']:
                # pred shape is batch,50,117 - 117 is 3 tracks x 3 (doa-x, doa-y, dist) x 13 classes
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 3, 13)
                doa_pred = pred[:, :, :, 0:2, :]
                dist_pred = pred[:, :, :, 2:3, :]
                doa_pred = self.doa_act(doa_pred)
                dist_pred = self.dist_act(dist_pred)
                pred = torch.cat((doa_pred, dist_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                # pred shape is batch,50,39 - 39 is 3 (doa-x, doa-y, dist) x 13 classes
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 13)
                doa_pred = pred[:, :,  0:2, :]
                dist_pred = pred[:, :, 2:3, :]
                doa_pred = self.doa_act(doa_pred)
                dist_pred = self.dist_act(dist_pred)
                pred = torch.cat((doa_pred, dist_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
        else:
            if self.params['multiACCDOA']:
                # pred shape is batch,50,156 - 156 is 3 tracks x 4 (doa-x, doa-y, dist, onscreen) x 13 classes
                pred = pred.reshape(pred.size(0), pred.size(1), 3, 4, 13)
                doa_pred = pred[:, :, :, 0:2, :]
                dist_pred = pred[:, :, :, 2:3, :]
                onscreen_pred = pred[:, :, :, 3:4, :]

                doa_pred = self.doa_act(doa_pred)
                dist_pred = self.dist_act(dist_pred)
                onscreen_pred = self.onscreen_act(onscreen_pred)

                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=3)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)
            else:
                # pred shape is batch,50,52 - 52 is 4 (doa-x, doa-y, dist, onscreen) x 13 classes
                pred = pred.reshape(pred.size(0), pred.size(1), 4, 13)
                doa_pred = pred[:, :, 0:2, :]
                dist_pred = pred[:, :, 2:3, :]
                onscreen_pred = pred[:, :, 3:4, :]

                doa_pred = self.doa_act(doa_pred)
                dist_pred = self.dist_act(dist_pred)
                onscreen_pred = self.onscreen_act(onscreen_pred)

                pred = torch.cat((doa_pred, dist_pred, onscreen_pred), dim=2)
                pred = pred.reshape(pred.size(0), pred.size(1), -1)

        return pred


if __name__ == '__main__':
    # use this to test if the SELDModel class works as expected.
    # All the classes will be called from the main.py for actual use.
    from parameters import params

    params['multiACCDOA'] = True
    #params['multiACCDOA'] = False

    params['modality'] = 'audio'
    params['modality'] = 'audio_visual'

    test_audio_feat = torch.rand([8, 2, 251, 64])
    test_video_feat = torch.rand([8, 50, 7, 7])
    # test_video_feat = None  # set to none for audio modality

    test_model = SELDModel(params)
    doa = test_model(test_audio_feat, test_video_feat)
    print(doa.size())




