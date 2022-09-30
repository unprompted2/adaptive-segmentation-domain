
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


def _parse_file_con