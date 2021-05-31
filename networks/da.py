import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from neuralnets.networks.unet import UNet2D, UNetDecoder2D, UNetEncoder2D
from neuralnets.networks.cnn import CNN2D
from neuralnets.util.losses import CrossEntropyLoss, L2Loss
from neuralnets.data.datasets import LabeledVolumeDataset

from networks.base import UNetDA2D, UNetDA2DClassifier, data_from_range, feature_regularization_loss, ReverseLayerF


class UNetMMD2D(UNetDA2D):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, lambda_mmd=0):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, return_features=True)

        self.lambda_mmd = lambda_mmd

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, f_src = self(x_src)
        y_tar_pred, f_tar = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_mmd = feature_regularization_loss(f_src, f_tar, method='mmd')
        loss = loss_src + loss_tar + self.lambda_mmd * loss_mmd

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        self.log('train/loss_mmd', loss_mmd, prog_bar=True)
        self.log('train/loss', loss)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='train_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='train_tar')

        return loss

    def validation_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, f_src = self(x_src)
        y_tar_pred, f_tar = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_mmd = feature_regularization_loss(f_src, f_tar, method='mmd')
        loss = loss_src + loss_tar + self.lambda_mmd * loss_mmd

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_src', loss_src)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss_mmd', loss_mmd, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.val_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss


class UNetMMD2DClassifier(UNetDA2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='lo