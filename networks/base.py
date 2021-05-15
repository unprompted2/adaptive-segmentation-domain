

import os
import numpy as np
import torch

from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR

from neuralnets.cross_validation.base import UNet2DClassifier
from neuralnets.networks.unet import UNet2D
from neuralnets.util.losses import CrossEntropyLoss

STEP_SIZE = 5
GAMMA = 0.95


def data_from_range(rng, dataset):

    X = []
    y = []
    for i, data in enumerate(dataset.data):
        inds = np.unique((rng * len(data)).astype(int))
        X.append(data[inds])
        y.append(dataset.labels[i][inds])

    return X, y


def _crop2match(f, g):
    """
    Center crops the largest tensor of two so that its size matches the other tensor
    :param f: first tensor
    :param g: second tensor
    :return: the cropped tensors
    """

    for d in range(f.ndim):
        if f.size(d) > g.size(d):  # crop f
            diff = f.size(d) - g.size(d)
            rest = f.size(d) - g.size(d) - (diff // 2)
            f = torch.split(f, [diff // 2, g.size(d), rest], dim=d)[1]
        elif g.size(d) > f.size(d):  # crop g
            diff = g.size(d) - f.size(d)
            rest = g.size(d) - f.size(d) - (diff // 2)
            g = torch.split(g, [diff // 2, f.size(d), rest], dim=d)[1]

    return f.contiguous(), g.contiguous()


def _compute_covariance(x):

    n = x.size(0)  # batch_size

    sum_column = torch.sum(x, dim=0, keepdim=True)
    term_mul_2 = torch.mm(sum_column.t(), sum_column) / n
    d_t_d = torch.mm(x.t(), x)

    return (d_t_d - term_mul_2) / (n - 1)


def _mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2


def feature_regularization_loss(f_src, f_tar, method='coral', n_samples=None):
    """
    Compute the regularization loss between the feature representations (shape [B, C, Y, X]) of the two streams
    In case of high dimensionality, there is an option to subsample
    :param f_src: features of the source stream
    :param f_tar: features of the target stream
    :param method: regularization method ('coral' or 'mmd')
    :param optional n_samples: number of samples to be selected
    :return: regularization loss
    """

    # if the samples are not equally sized, center crop the largest one
    f_src, f_tar = _crop2match(f_src, f_tar)

    # view features to [N, D] shape
    src = f_src.view(f_src.size(0), -1)
    tar = f_tar.view(f_tar.size(0), -1)

    if n_samples is None:
        fs = src
        ft = tar
    else:
        inds = torch.randperm(src.size(1))[:n_samples]
        fs = src[:, inds.to(src.device)]
        ft = tar[:, inds.to(tar.device)]

    if method == 'coral':
        return coral(fs, ft)
    else:
        return mmd(fs, ft)


def coral(source, target):
    """
    Compute CORAL loss between two feature vectors (https://arxiv.org/abs/1607.01719)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: CORAL loss
    """
    d = source.size(1)

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    # loss = loss / (4*d*d)

    return loss


def mmd(source, target, gamma=10**3):
    """
    Compute MMD loss between two feature vectors (https://arxiv.org/abs/1605.06636)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: MMD loss
    """
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(source, target, [gamma])

    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True) / source.size(1)


class ReverseLayerF(Function):
    """
    Gradient reversal layer (https://arxiv.org/abs/1505.07818)
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None


class UNetDA2D(UNet2D):

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred = self(x_src)
        y_tar_pred = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss = loss_src + loss_tar

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)