
"""
    This is a script that illustrates semi-supervised domain adaptive training
"""

"""
    Necessary libraries
"""
import argparse
import yaml
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from neuralnets.util.io import print_frm, mkdir
from neuralnets.util.tools import set_seed

from util.tools import parse_params, get_dataloaders, rmdir, mv, cp
from networks.factory import generate_model
from train.base import train, validate

from multiprocessing import freeze_support


