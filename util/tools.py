import shutil
from torch.utils.data import DataLoader

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.util.tools import parse_params as parse_params_base
from neuralnets.util.io import print_frm
from neuralnets.util.augmentation import *


def get_transforms(tfs, coi=None):
    """
    Builds a transform object based on a list of desired augmentations

    :param tfs: list of augmentations, options: rot90, flipx, flipy, contrast, deformation, noise
    :param coi: classes of interest (only required if deformations are included)
    :return: transform object that implements the desired augmentations
    """

    # dictionary that maps augmentation strings to transform objects
    mapper = {'rot90': Rotate90(),
              'flipx': Flip(prob=0.5, dim=0),
              'flipy': Flip(prob=0.5, dim=1),
              'contrast': ContrastAdjust(adj=0.1),
              'deformation': RandomDeformation(),
              'noise': AddNoise(sigma_max=0.05)}

    # build the transforms
    tf_list = []
    for k