# This code builds the scripts that will be launched on HPC

import os
import argparse

from neuralnets.util.io import mkdir

EPFL = 'EPFL'
UROCELL = 'UroCell'
PO936Q = 'po936q'
MITOEM_H = 'MitoEM-H'
MITOEM_R = 'MitoEM-R'
MIRA = 'MiRA'
KASTHURI = 'Kasthuri'
VNC = 'VNC'
EMBL_HELA = 'EMBL'
VIB_EVHELA = 'evhela'
DOMAINS = [VIB_EVHELA, VNC, MITOEM_H, EPFL, KASTHURI]
SRC_DOMAINS = '"' + '" "'.join(DOMAINS) + '"'

# methods
NO_DA = 'no-da'
MMD = 'mmd'
DAT = 'dat'
YNET = 'ynet'
UNET_TS = 'unet-ts'
METHODS = [NO_DA, MMD, DAT, YNET, UNET_TS]

# available labels
AVAILABLE_LABELS = [0.05, 0.10, 0.20, 0.50, 1.00]


parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base script", required=True, type=str)
parser.add_argument("--target_dir", "-t", help="Path to the directory where the scripts will be saved", required=True,
                    type=str)
parser.add