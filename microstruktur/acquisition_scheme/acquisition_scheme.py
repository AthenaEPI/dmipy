import numpy as np
from microstruktur.signal_models.gradient_conversions import (
    g_from_b, q_from_b, b_from_q, g_from_q, b_from_g, q_from_g
)
from microstruktur.signal_models import utils
from dipy.reconst.shm import real_sym_sh_mrtrix
sh_order = 14


class AcquisitionScheme:

    def __init__(self, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta, shell_indices):

        # check_bvals_n_shell_indices_delta_Delta(bvalues, gradient_directions,
        #                                         shell_indices, delta, Delta)
        self.bvalues = bvalues
        self.gradient_directions = gradient_directions
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        self.n = self.gradient_directions
        self.delta = delta
        self.Delta = Delta
        self.tau = Delta - delta / 3.
        self.shell_indices = shell_indices

        self.shell_sh_matrices = {}
        for shell_index in np.arange(1, shell_indices.max() + 1):  # per shell
            # what bvecs in that shell
            bvecs_shell = self.n[shell_indices == shell_index]
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                sh_order, theta_, phi_)[0]


def acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta, shell_indices):
    qvalues = q_from_b(bvalues, delta, Delta)
    gradient_strengths = g_from_b(bvalues, delta, Delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta, Delta, shell_indices)


def acquisition_scheme_from_qvalues(
        qvalues, gradient_directions, delta, Delta, shell_indices):
    bvalues = b_from_q(qvalues, delta, Delta)
    gradient_strengths = g_from_q(qvalues, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta, Delta, shell_indices)


def acquisition_scheme_from_gradient_strengths(
        gradient_strengths, gradient_directions, delta, Delta, shell_indices):
    bvalues = b_from_g(gradient_strengths, delta, Delta)
    qvalues = q_from_g(gradient_strengths, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta, Delta, shell_indices)


# def check_bvals_n_shell_indices_delta_Delta(
#     bvals, n, shell_indices, delta, Delta
# ):

def check_bvals_n_shell_indices_delta_Delta(
    bvals, n, shell_indices, delta=None, Delta=None
):
    # Function that tests the validity of the input acquisition parameters.
    if len(bvals) != len(n):
        msg = "bvals and n must have the same length. "
        msg += "Currently their lengths are {} and {}.".format(
            len(bvals), len(n)
        )
        raise ValueError(msg)
    if shell_indices is not None and len(bvals) != len(shell_indices):
        msg = "bvals and shell_indices must have the same length. "
        msg += "Currently their lengths are {} and {}.".format(
            len(bvals), len(shell_indices)
        )
        raise ValueError(msg)
    if not np.all(abs(np.linalg.norm(n, axis=1) - 1.) < 0.001):
        msg = "gradient orientations n are not unit vectors. "
        raise ValueError(msg)
    if delta is not None and Delta is not None:
        if len(bvals) != len(delta) or len(bvals) != len(Delta):
            msg = "bvals, delta and Delta must have the same length. "
            msg += "Currently their lengths are {}, {} and {}.".format(
                len(bvals), len(delta), len(Delta)
            )
            raise ValueError(msg)
        if delta.ndim > 1 or Delta.ndim > 1:
            msg = "delta and Delta must be one-dimensional arrays. "
            msg += "Currently their dimensions are {} and {}.".format(
                delta.ndim, Delta.ndim
            )
            raise ValueError(msg)
        if np.min(delta) < 0 or np.min(Delta) < 0:
            msg = "delta and Delta must be zero or positive. "
            msg += "Currently their minimum values are {} and {}.".format(
                np.min(delta), np.min(Delta)
            )
            raise ValueError(msg)
    if bvals.ndim > 1:
        msg = "bvals must be a one-dimensional array. "
        msg += "Currently its dimensions is {}.".format(
            bvals.ndim
        )
        raise ValueError(msg)
    if shell_indices is not None and shell_indices.ndim > 1:
        msg = "shell_indices must be a one-dimensional array. "
        msg += "Currently its dimension is {}.".format(
            shell_indices.ndim
        )
        raise ValueError(msg)
    if n.ndim != 2 or n.shape[1] != 3:
        msg = "b-vectors n must be two dimensional array of shape [N, 3]. "
        msg += "Currently its shape is {}.".format(n.shape)
        raise ValueError(msg)
    if np.min(bvals) < 0.:
        msg = "bvals must be zero or positive. "
        msg += "Minimum value found is {}.".format(bvals.min)
        raise ValueError(msg)
