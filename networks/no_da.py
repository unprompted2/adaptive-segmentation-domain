import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from networks.base import UNetDA2D, UNetDA2DClassifier, data_from_range

from neuralnets.data.datasets import LabeledVolumeDataset


class UNetNoDA2D(UNetDA2D):

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred = self(x_src)
        y_tar_pred = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss = loss_src + loss_tar

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/l