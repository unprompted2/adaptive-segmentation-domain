
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
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                        activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                        lambda_mmd=params['lambda_mmd'])
    elif name == 'dat':
        net = UNetDAT2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                        activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                        lambda_dat=params['lambda_dat'], input_shape=params['input_size'])
    elif name == 'ynet':
        net = YNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                     dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                     activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                     lambda_rec=params['lambda_rec'])
    elif name == 'wnet':
        net = WNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                     dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                     activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                     lambda_rec=params['lambda_rec'], lambda_dat=params['lambda_dat'],
                     input_shape=params['input_size'])
    elif name == 'unet-ts':
        net = UNetTS2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                       dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                       activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                       lambda_w=params['lambda_w'], lambda_o=params['lambda_o'])
    else:
        net = UNetDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], l