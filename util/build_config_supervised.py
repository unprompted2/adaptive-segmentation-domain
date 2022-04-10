
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

    params = {'train_val_test_split': {}, 'split_orientation': {}, 'input_size': {}, 'coi': {}}

    # train/val/test split parameters
    params['train_val_test_split'][EPFL] = '0.40,0.50'
    for DOM in [UROCELL, PO936Q, MITOEM_H, MITOEM_R, VIB_EVHELA]:
        params['train_val_test_split'][DOM] = '0.48,0.60'
    params['train_val_test_split'][VNC] = '0.30,0.50'
    params['train_val_