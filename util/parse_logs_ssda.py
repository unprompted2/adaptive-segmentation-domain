
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


DPI = 100


def _symmetry(A, ord='fro'):

    A_sym = 0.5 * (A + A.T)
    A_anti = 0.5 * (A - A.T)

    s_sym = np.linalg.norm(A_sym, ord=ord)
    s_anti = np.linalg.norm(A_anti, ord=ord)

    s = (s_sym - s_anti) / (s_sym + s_anti)

    return s


def _get_filename(method, frac, src, tar):
    return method + '-' + str(frac) + '-' + src + '2' + tar + '.logs'


def _parse_file_contents(contents):
    lines = contents.split('\n')
    mIoU = -1
    total_runtime = 0
    for line in lines:
        if "Elapsed " in line:
            s = line.split(': ')[1].split(', ')
            h = int(s[0].replace(' hours', ''))
            m = int(s[1].replace(' minutes', ''))
            s = float(s[2].replace(' seconds', ''))
            total_runtime += h*3600 + m*60 + s
        if 'mIoU: ' in line:
            mIoU = float(line.split('mIoU: ')[1])
            break
    return mIoU, total_runtime


log_dir = '/home/jorisro/research/domain-adaptive-segmentation/train/semi-supervised-da/logs'

domains = ['EPFL', 'evhela', 'Kasthuri', 'MitoEM-H', 'VNC']
methods = ['no-da', 'mmd', 'dat', 'ynet', 'unet-ts']
methods_nice = {'no-da': 'No-DA', 'mmd': 'MMD', 'dat': 'DAT', 'ynet': 'Y-Net', 'unet-ts': 'UNet-TS'}
domains_train = {'EPFL': 272*0.4, 'evhela': 360*0.48, 'Kasthuri': 424*0.426, 'MitoEM-H': 16777*0.48, 'VNC': 21*0.3}
als = [0.05, 0.10, 0.20, 0.50, 1.00]

hmaps = {}
tmaps = {}
mean_to = np.zeros((len(methods), len(domains)))
mean_from = np.zeros((len(methods), len(dom