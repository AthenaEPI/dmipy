import numpy as np
from microstruktur.signal_models.gradient_conversions import (
    g_from_b, q_from_b, b_from_q, g_from_q, b_from_g, q_from_g
)
from microstruktur.signal_models import utils
from dipy.reconst.shm import real_sym_sh_mrtrix
from scipy.cluster.hierarchy import fcluster, linkage

sh_order = 14


class AcquisitionScheme:

    def __init__(self, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta):
        check_scheme_from_bvalues(bvalues, gradient_directions, delta, Delta)
        self.bvalues = bvalues
        self.gradient_directions = gradient_directions
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        self.delta = delta
        self.Delta = Delta
        self.tau = Delta - delta / 3.
        self.shell_indices, self.shell_bvalues = (
            calculate_shell_bvalues_and_indices(bvalues))
        first_indices = [
            np.argmax(self.shell_indices == ind)
            for ind in np.arange(self.shell_indices.max() + 1)]
        self.shell_delta = self.delta[first_indices]
        self.shell_Delta = self.Delta[first_indices]
        self.number_of_measurements = len(self.bvalues)
        self.b0_mask = self.shell_indices == 0
        self.number_of_b0s = np.sum(self.b0_mask)

        self.shell_sh_matrices = {}
        for shell_index in np.arange(1, self.shell_indices.max() + 1):
            shell_mask = self.shell_indices == shell_index
            self.shell_bvalues[shell_index] = np.mean(bvalues[shell_mask])
            bvecs_shell = self.gradient_directions[shell_mask]
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                sh_order, theta_, phi_)[0]


class SimpleAcquisitionSchemeRH:

    def __init__(self, bvalue, gradient_directions, qvalue=None,
                 delta=None, Delta=None, tau=None):
        self.bvalues = np.tile(bvalue, len(gradient_directions))
        if qvalue is not None:
            self.qvalues = np.tile(qvalue, len(gradient_directions))
        if delta is not None and Delta is not None:
            self.delta = np.tile(delta, len(gradient_directions))
            self.Delta = np.tile(Delta, len(gradient_directions))
        if tau is not None:
            self.tau = np.tile(tau, len(gradient_directions))
        self.gradient_directions = gradient_directions


def acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta):
    delta_, Delta_ = unify_length_reference_delta_Delta(bvalues, delta, Delta)
    qvalues = q_from_b(bvalues, delta_, Delta_)
    gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_)


def acquisition_scheme_from_qvalues(
        qvalues, gradient_directions, delta, Delta):
    delta_, Delta_ = unify_length_reference_delta_Delta(qvalues, delta, Delta)
    bvalues = b_from_q(qvalues, delta, Delta)
    gradient_strengths = g_from_q(qvalues, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_)


def acquisition_scheme_from_gradient_strengths(
        gradient_strengths, gradient_directions, delta, Delta):
    delta_, Delta_ = unify_length_reference_delta_Delta(gradient_strengths,
                                                        delta, Delta)
    bvalues = b_from_g(gradient_strengths, delta, Delta)
    qvalues = q_from_g(gradient_strengths, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_)


def unify_length_reference_delta_Delta(reference_array, delta, Delta):
    if isinstance(delta, float):
        delta_ = np.tile(delta, len(reference_array))
    else:
        delta_ = delta.copy()
    if isinstance(Delta, float):
        Delta_ = np.tile(Delta, len(reference_array))
    else:
        Delta_ = Delta.copy()
    return delta_, Delta_


def calculate_shell_bvalues_and_indices(bvalues, max_distance=50e6):
    linkage_matrix = linkage(np.c_[bvalues])
    clusters = fcluster(linkage_matrix, max_distance, criterion='distance')
    shell_indices = np.empty_like(bvalues, dtype=int)
    cluster_bvalues = np.zeros((np.max(clusters), 2))
    for ind in np.unique(clusters):
        cluster_bvalues[ind - 1] = np.mean(bvalues[clusters == ind]), ind
    shell_bvalues, ordered_cluster_indices = (
        cluster_bvalues[cluster_bvalues[:, 0].argsort()].T)
    for i, ind in enumerate(ordered_cluster_indices):
        shell_indices[clusters == ind] = i
    return shell_indices, shell_bvalues


def check_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta):
    if len(bvalues) != len(gradient_directions):
        msg = "bvalues and gradient_directions must have the same length. "
        msg += "Currently their lengths are {} and {}.".format(
            len(bvalues), len(gradient_directions)
        )
        raise ValueError(msg)
    if len(bvalues) != len(delta) or len(bvalues) != len(Delta):
        msg = "bvalues, delta and Delta must have the same length. "
        msg += "Currently their lengths are {}, {} and {}.".format(
            len(bvalues), len(delta), len(Delta)
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
    if bvalues.ndim > 1:
        msg = "bvalues must be a one-dimensional array. "
        msg += "Currently its dimensions is {}.".format(
            bvalues.ndim
        )
        raise ValueError(msg)
    if gradient_directions.ndim != 2 or gradient_directions.shape[1] != 3:
        msg = "b-vectors n must be two dimensional array of shape [N, 3]. "
        msg += "Currently its shape is {}.".format(gradient_directions.shape)
        raise ValueError(msg)
    if np.min(bvalues) < 0.:
        msg = "bvalues must be zero or positive. "
        msg += "Minimum value found is {}.".format(bvalues.min())
        raise ValueError(msg)
    if not np.all(
            abs(np.linalg.norm(gradient_directions, axis=1) - 1.) < 0.001):
        msg = "gradient orientations n are not unit vectors. "
        raise ValueError(msg)
