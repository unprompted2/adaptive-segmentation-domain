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
        tar_