"""
    This is a script that illustrates segmentation of a dataset with a particular model
"""
import torch

"""
    Necessary libraries
"""
import argparse
import yaml
import time

from neuralnets.util.io import print_frm, read_pngseq
from neuralnets.util.tools import set_seed
from neuralnets.util.validation import segment_read, segment_ram

from util.tools import parse_params, process_seconds
from networks.factory import generate_model

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the network configuration file", type=st