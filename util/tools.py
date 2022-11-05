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

    # dictionary that maps augmen