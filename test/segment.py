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
    parser.add_argument("--config", "-c", help="Path to the network configuration file", type=str,
                        default='../train_supervised.yaml')
    parser.add_argument("--model", "-m", help="Path to the network parameters", type=str, required=True)
    parser.add_argument("--dataset", "-d", help="Path to the dataset that needs to be segmented", type=str,
                        required=True)
    parser.add_argument("--block_wise", "-bw", help="Flag that specifies to compute block wise or not",
                        action='store_true', default=False)
    parser.add_argument("--output", "-o", help="Path to store the output segmentation", type=str, required=True)
    parser.add_argument("--gpu", "-g", help="GPU device for computations", type=int, defaul