
from networks.no_da import UNetNoDA2DClassifier
from networks.da import *

from neuralnets.util.io import print_frm


def generate_model(name, params):

    if name == 'u-net' or name == 'no-da':
        net = UNetDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                       dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                       activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'])
    elif name == 'mmd':
        net = UNetMMD2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], nor