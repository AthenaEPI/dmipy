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
                 gradient_strengths, delta, Delta,
                 min_b_shell_distance, b0_threshold):
        check_scheme_from_bvalues(bvalues, gradient_directions, delta, Delta)
        self.min_b_shell_distance = min_b_shell_distance
        self.b0_threshold = b0_threshold
        self.bvalues = bvalues
        self.b0_mask = self.bvalues <= b0_threshold
        self.number_of_measurements = len(self.bvalues)
        self.gradient_directions = gradient_directions
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        self.delta = delta
        self.Delta = Delta
        self.tau = Delta - delta / 3.
        if self.number_of_measurements > 1:
            self.shell_indices, self.shell_bvalues = (
                calculate_shell_bvalues_and_indices(
                    bvalues, min_b_shell_distance)
            )
            self.shell_b0_mask = self.shell_bvalues <= b0_threshold
            first_indices = [
                np.argmax(self.shell_indices == ind)
                for ind in np.arange(self.shell_indices.max() + 1)]
            self.shell_qvalues = self.qvalues[first_indices]
            self.shell_gradient_strengths = (
                self.gradient_strengths[first_indices])
            self.shell_delta = self.delta[first_indices]
            self.shell_Delta = self.Delta[first_indices]
        else:
            self.shell_bvalues = self.bvalues
            self.shell_indices = np.r_[int(0)]
            if self.shell_bvalues > b0_threshold:
                self.shell_b0_mask = np.r_[False]
            else:
                self.shell_b0_mask = np.r_[True]
            self.shell_qvalues = self.qvalues
            self.shell_gradient_strengths = self.gradient_strengths
            self.shell_delta = self.delta
            self.shell_Delta = self.Delta

        self.number_of_b0s = np.sum(self.b0_mask)

        self.unique_dwi_indices = np.unique(self.shell_indices[~self.b0_mask])
        self.shell_sh_matrices = {}
        for shell_index in self.unique_dwi_indices:
            shell_mask = self.shell_indices == shell_index
            bvecs_shell = self.gradient_directions[shell_mask]
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                sh_order, theta_, phi_)[0]


class SimpleAcquisitionSchemeRH:

    def __init__(self, bvalue, gradient_directions, delta=None, Delta=None):
        Ndata = len(gradient_directions)
        self.bvalues = np.tile(bvalue, Ndata)
        self.gradient_directions = gradient_directions
        self.b0_mask = np.tile(False, Ndata)
        if delta is not None and Delta is not None:
            self.delta = np.tile(delta, Ndata)
            self.Delta = np.tile(Delta, Ndata)
            self.tau = self.Delta - self.delta / 3.0
            self.qvalues = np.tile(q_from_b(bvalue, delta, Delta), Ndata)
            self.gradient_strengths = (
                np.tile(g_from_b(bvalue, delta, Delta), Ndata)
            )


def acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta,
        min_b_shell_distance=50e6, b0_threshold=0.):
    delta_, Delta_ = unify_length_reference_delta_Delta(bvalues, delta, Delta)
    qvalues = q_from_b(bvalues, delta_, Delta_)
    gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_,
                             min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_qvalues(
        qvalues, gradient_directions, delta, Delta,
        min_b_shell_distance=50e6, b0_threshold=0.):
    delta_, Delta_ = unify_length_reference_delta_Delta(qvalues, delta, Delta)
    bvalues = b_from_q(qvalues, delta, Delta)
    gradient_strengths = g_from_q(qvalues, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_,
                             min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_gradient_strengths(
        gradient_strengths, gradient_directions, delta, Delta,
        min_b_shell_distance=50e6, b0_threshold=0.):
    delta_, Delta_ = unify_length_reference_delta_Delta(gradient_strengths,
                                                        delta, Delta)
    bvalues = b_from_g(gradient_strengths, delta, Delta)
    qvalues = q_from_g(gradient_strengths, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_,
                             min_b_shell_distance, b0_threshold)


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
