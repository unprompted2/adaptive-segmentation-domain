import shutil
from torch.utils.data import DataLoader

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.util.tools import parse_params as parse_params_base
from neuralnets.util.io import print_frm
from neuralnets.util.augmentation import *


def get_transforms(tfs, coi=None):
    """
    Build