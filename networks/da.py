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
        super().__init__(input_shape=input_shape, in_channels=in_channels