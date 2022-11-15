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

    # dictionary that maps augmentation strings to transform objects
    mapper = {'rot90': Rotate90(),
              'flipx': Flip(prob=0.5, dim=0),
              'flipy': Flip(prob=0.5, dim=1),
              'contrast': ContrastAdjust(adj=0.1),
              'deformation': RandomDeformation(),
              'noise': AddNoise(sigma_max=0.05)}

    # build the transforms
    tf_list = []
    for key in mapper:
        if key in tfs:
            tf_list.append(mapper[key])

    # required post-processing
    if 'deformation' in tfs:
        tf_list.append(CleanDeformedLabels(coi))

    return Compose(tf_list)


def get_dataloaders(params, domain=None, domain_labels_available=1.0, supervised=False):

    input_shape = (1, *(params['input_size']))
    transform = get_transforms(params['augmentation'], coi=params['coi'])
    print_frm('Applying data augmentation! Specifically %s' % str(params['augmentation']))

    if domain is None:

        split_src = params['src']['train_val_test_split']
        split_tar = params['tar']['train_val_test_split']
        print_frm('Train data... ')
        train = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                                     (params['src']['labels'], params['tar']['labels']), len_epoch=params['len_epoch'],
                                     input_shape=input_shape, in_channels=params['in_channels'],
                                     type=params['type'], batch_size=params['train_batch_size'], transform=transform,
                                     range_split=((0, split_src[0]), (0, split_tar[0])), coi=params['coi'],
                                     range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']),
                                     partial_labels=(1, params['tar_labels_available']), seed=params['seed'])
        print_frm('Validation data...')
        val = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                                   (params['src']['labels'], params['tar']['labels']), len_epoch=params['len_epoch'],
                                   input_shape=input_shape, in_channels=para