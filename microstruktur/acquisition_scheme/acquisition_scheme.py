import numpy as np
from microstruktur.signal_models.gradient_conversions import (
    g_from_b, q_from_b, b_from_q, g_from_q, b_from_g, q_from_g)
from microstruktur.signal_models import utils
from dipy.reconst.shm import real_sym_sh_mrtrix
from scipy.cluster.hierarchy import fcluster, linkage
from warnings import warn

sh_order = 14


class AcquisitionScheme:
    """
    Class that calculates and contains all information needed to simulate and
    fit data using microstructure models.
    """

    def __init__(self, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta,
                 min_b_shell_distance, b0_threshold):
        self.min_b_shell_distance = min_b_shell_distance
        self.b0_threshold = b0_threshold
        self.bvalues = bvalues
        self.b0_mask = self.bvalues <= b0_threshold
        self.number_of_b0s = np.sum(self.b0_mask)
        self.number_of_measurements = len(self.bvalues)
        self.gradient_directions = gradient_directions
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        self.delta = delta
        self.Delta = Delta
        self.tau = Delta - delta / 3.
        # if there are more then 1 measurement
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
        # if for some reason only one measurement is given (for testing)
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

        # calculates observation matrices to convert spherical harmonic
        # coefficients to the positions on the sphere for every shell
        self.unique_dwi_indices = np.unique(self.shell_indices[~self.b0_mask])
        self.shell_sh_matrices = {}
        for shell_index in self.unique_dwi_indices:
            shell_mask = self.shell_indices == shell_index
            bvecs_shell = self.gradient_directions[shell_mask]
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                sh_order, theta_, phi_)[0]
        # warning in case there are no b0 measurements
        if sum(self.b0_mask) == 0:
            msg = "No b0 measurements were detected. Check if the b0_threshold"
            msg += " option is high enough, or if there is a mistake in the "
            msg += "acquisition design."
            warn(msg)

    def print_acquisition_info(self):
        """
        prints a small summary of the acquisition scheme. Is useful to check if
        the function correctly separated the shells and if the input parameters
        were given in the right scale.
        """
        print "Acquisition scheme summary\n"
        print "total number of measurements: {}".format(
            self.number_of_measurements)
        print "number of b0 measurements: {}".format(self.number_of_b0s)
        print "number of DWI shells: {}\n".format(np.sum(~self.shell_b0_mask))
        upper_line = "shell_index |# of DWIs |bvalue [s/mm^2] "
        upper_line += "|gradient strength [mT/m] |delta [ms] |Delta[ms]"
        print upper_line
        for ind in np.arange(max(self.shell_indices) + 1):
            print "{: <12}|{: <10}|{: <16}|{: <25}|{: <11}|{: <5}".format(
                str(ind), sum(self.shell_indices == ind),
                int(self.shell_bvalues[ind] / 1e6),
                int(1e3 * self.shell_gradient_strengths[ind]),
                self.shell_delta[ind] * 1e3, self.shell_Delta[ind] * 1e3)


class SimpleAcquisitionSchemeRH:
    """
    This is a very simple class that is only used internally to create the
    rotational harmonics to be used in spherical convolution.
    """

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
        min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Creates an acquisition scheme object from bvalues, gradient directions,
    pulse duration $\delta$ and pulse separation time $\Delta$.

    Parameters
    ----------
    bvalues: 1D numpy array of shape (Ndata)
        bvalues of the acquisition in s/m^2.
        e.g., a bvalue of 1000 s/mm^2 must be entered as 1000 * 1e6 s/m^2
    gradient_directions: 2D numpy array of shape (Ndata, 3)
        gradient directions array of cartesian unit vectors.
    delta: float or 1D numpy array of shape (Ndata)
        if float, pulse duration of every measurements in seconds.
        if array, potentially varying pulse duration per measurement.
    Delta: float or 1D numpy array of shape (Ndata)
        if float, pulse separation time of every measurements in seconds.
        if array, potentially varying pulse separation time per measurement.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    AcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_ = unify_length_reference_delta_Delta(bvalues, delta, Delta)
    check_acquisition_scheme(bvalues, gradient_directions, delta_, Delta_)
    qvalues = q_from_b(bvalues, delta_, Delta_)
    gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_,
                             min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_qvalues(
        qvalues, gradient_directions, delta, Delta,
        min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Creates an acquisition scheme object from qvalues, gradient directions,
    pulse duration $\delta$ and pulse separation time $\Delta$.

    Parameters
    ----------
    qvalues: 1D numpy array of shape (Ndata)
        diffusion sensitization of the acquisition in 1/m.
        e.g. a qvalue of 10 1/mm must be entered as 10 * 1e3 1/m
    gradient_directions: 2D numpy array of shape (Ndata, 3)
        gradient directions array of cartesian unit vectors.
    delta: float or 1D numpy array of shape (Ndata)
        if float, pulse duration of every measurements in seconds.
        if array, potentially varying pulse duration per measurement.
    Delta: float or 1D numpy array of shape (Ndata)
        if float, pulse separation time of every measurements in seconds.
        if array, potentially varying pulse separation time per measurement.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    AcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_ = unify_length_reference_delta_Delta(qvalues, delta, Delta)
    check_acquisition_scheme(qvalues, gradient_directions, delta_, Delta_)
    bvalues = b_from_q(qvalues, delta, Delta)
    gradient_strengths = g_from_q(qvalues, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_,
                             min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_gradient_strengths(
        gradient_strengths, gradient_directions, delta, Delta,
        min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Creates an acquisition scheme object from gradient strengths, gradient
    directions pulse duration $\delta$ and pulse separation time $\Delta$.

    Parameters
    ----------
    gradient_strengths: 1D numpy array of shape (Ndata)
        gradient strength of the acquisition in T/m.
        e.g., a gradient strength of 300 mT/m must be entered as 300 / 1e3 T/m
    gradient_directions: 2D numpy array of shape (Ndata, 3)
        gradient directions array of cartesian unit vectors.
    delta: float or 1D numpy array of shape (Ndata)
        if float, pulse duration of every measurements in seconds.
        if array, potentially varying pulse duration per measurement.
    Delta: float or 1D numpy array of shape (Ndata)
        if float, pulse separation time of every measurements in seconds.
        if array, potentially varying pulse separation time per measurement.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    AcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_ = unify_length_reference_delta_Delta(gradient_strengths,
                                                        delta, Delta)
    check_acquisition_scheme(gradient_strengths, gradient_directions,
                             delta_, Delta_)
    bvalues = b_from_g(gradient_strengths, delta, Delta)
    qvalues = q_from_g(gradient_strengths, delta)
    return AcquisitionScheme(bvalues, gradient_directions, qvalues,
                             gradient_strengths, delta_, Delta_,
                             min_b_shell_distance, b0_threshold)


def unify_length_reference_delta_Delta(reference_array, delta, Delta):
    """
    If either delta or Delta are given as float, makes them an array the same
    size as the reference array.
    """
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
    """
    Calculates which measurements belong to different acquisition shells.
    It uses scipy's linkage clustering algorithm, which uses the max_distance
    input as a limit of including measurements in the same cluster.

    For example, if bvalues were [1, 2, 3, 4, 5] and max_distance was 1, then
    all bvalues would belong to the same cluster.
    However, if bvalues were [1, 2, 4, 5] max max_distance was 1, then this
    would result in 2 clusters.

    Parameters
    ----------
    bvalues: 1D numpy array of shape (Ndata)
        bvalues of the acquisition in s/m^2.
    max_distance: float
        maximum b-value distance for a measurement to be included in the same
        shell.

    Returns
    -------
    shell_indices: 1D numpy array of shape (Ndata)
        array of integers, starting from 0, representing to which shell a
        measurement belongs. The number itself has no meaning other than just
        being different for different shells.
    shell_bvalues: 1D numpy array of shape (Nshells)
        array of the mean bvalues for every acquisition shell.
    """
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


def check_acquisition_scheme(
        bqg_values, gradient_directions, delta, Delta):
    "function to check the validity of the input parameters."
    if bqg_values.ndim > 1:
        msg = "b/q/G input must be a one-dimensional array. "
        msg += "Currently its dimensions is {}.".format(
            bqg_values.ndim
        )
        raise ValueError(msg)
    if len(bqg_values) != len(gradient_directions):
        msg = "b/q/G input and gradient_directions must have the same length. "
        msg += "Currently their lengths are {} and {}.".format(
            len(bqg_values), len(gradient_directions)
        )
        raise ValueError(msg)
    if len(bqg_values) != len(delta) or len(bqg_values) != len(Delta):
        msg = "b/q/G input, delta and Delta must have the same length. "
        msg += "Currently their lengths are {}, {} and {}.".format(
            len(bqg_values), len(delta), len(Delta)
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
    if gradient_directions.ndim != 2 or gradient_directions.shape[1] != 3:
        msg = "gradient_directions n must be two dimensional array of shape "
        msg += "[N, 3]. Currently its shape is {}.".format(
            gradient_directions.shape)
        raise ValueError(msg)
    if np.min(bqg_values) < 0.:
        msg = "b/q/G input must be zero or positive. "
        msg += "Minimum value found is {}.".format(bqg_values.min())
        raise ValueError(msg)
    if not np.all(
            abs(np.linalg.norm(gradient_directions, axis=1) - 1.) < 0.001):
        msg = "gradient orientations n are not unit vectors. "
        raise ValueError(msg)
