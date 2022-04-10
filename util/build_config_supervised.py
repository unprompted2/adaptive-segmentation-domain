
import os
import argparse
import yaml
from yaml.loader import SafeLoader

# domains
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


def _default_params():

    params = {'train_val_test_split': {}, 'split_orientation': {}, 'input_si