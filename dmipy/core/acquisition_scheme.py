import numpy as np
from .gradient_conversions import (
    g_from_b, q_from_b, b_from_q, g_from_q, b_from_g, q_from_g)
from ..utils import utils
from ..utils.spherical_convolution import real_sym_rh_basis
from dipy.reconst.shm import real_sym_sh_mrtrix
from scipy.cluster.hierarchy import fcluster, linkage
from dipy.core.gradients import gradient_table, GradientTable
import matplotlib.pyplot as plt
from warnings import warn


__all__ = [
    'get_sh_order_from_bval',
    'DmipyAcquisitionScheme',
    'RotationalHarmonicsAcquisitionScheme',
    'SphericalMeanAcquisitionScheme',
    'acquisition_scheme_from_bvalues',
    'acquisition_scheme_from_qvalues',
    'acquisition_scheme_from_gradient_strengths',
    'acquisition_scheme_from_schemefile',
    'unify_length_reference_delta_Delta',
    'calculate_shell_bvalues_and_indices',
    'check_acquisition_scheme',
    'gtab_dipy2dmipy',
    'gtab_dmipy2dipy'
]


def get_sh_order_from_bval(bval):
    "Estimates minimum sh_order to represent data of given b-value."
    bvals = np.r_[2.02020202e+08, 7.07070707e+08, 1.21212121e+09,
                  2.52525253e+09, 3.13131313e+09, 5.35353535e+09,
                  np.inf]
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(bvals > bval)]


class DmipyAcquisitionScheme:
    """
    Class that calculates and contains all information needed to simulate and
    fit data using microstructure models.
    """

    def __init__(self, bvalues, gradient_directions, qvalues,
                 gradient_strengths, delta, Delta, TE,
                 min_b_shell_distance, b0_threshold):
        self.min_b_shell_distance = float(min_b_shell_distance)
        self.b0_threshold = float(b0_threshold)
        self.bvalues = bvalues.astype(float)
        self.b0_mask = self.bvalues <= b0_threshold
        self.number_of_b0s = np.sum(self.b0_mask)
        self.number_of_measurements = len(self.bvalues)
        self.gradient_directions = gradient_directions.astype(float)
        self.qvalues = None
        if qvalues is not None:
            self.qvalues = qvalues.astype(float)
        self.gradient_strengths = None
        if gradient_strengths is not None:
            self.gradient_strengths = gradient_strengths.astype(float)
        self.delta = None
        if delta is not None:
            self.delta = delta.astype(float)
        self.Delta = None
        if Delta is not None:
            self.Delta = Delta.astype(float)
        self.TE = None
        if TE is not None:
            self.TE = TE.astype(float)
        self.tau = None
        if self.delta is not None and self.Delta is not None:
            self.tau = Delta - delta / 3.
        # if there are more then 1 measurement
        if self.number_of_measurements > 1:
            # we check if there are multiple unique delta-Delta combinations
            if self.TE is not None:
                deltas = np.c_[self.delta, self.Delta, self.TE]
            elif self.delta is not None and self.Delta is not None:
                deltas = np.c_[self.delta, self.Delta]
            elif self.delta is None and self.Delta is not None:
                deltas = np.c_[self.Delta]
            elif self.delta is not None and self.Delta is None:
                deltas = np.c_[self.delta]
            else:
                deltas = []

            if deltas == []:
                deltas = np.c_[np.zeros(len(self.bvalues))]
            unique_deltas = np.unique(deltas, axis=0)
            self.shell_indices = np.zeros(len(bvalues), dtype=int)
            self.shell_bvalues = []
            max_index = 0
            # for every unique combination we separate shells based on bvalue
            # reason for separation is that different combinations of
            # delta and Delta can result in the same b-value, which could
            # result in wrong classification of DWIs to unique shells.
            for unique_deltas_ in unique_deltas:
                delta_mask = np.all(deltas == unique_deltas_, axis=1)
                masked_bvals = bvalues[delta_mask]
                if len(masked_bvals) > 1:
                    shell_indices_, shell_bvalues_ = (
                        calculate_shell_bvalues_and_indices(
                            masked_bvals, min_b_shell_distance)
                    )
                else:
                    shell_indices_, shell_bvalues_ = np.array(0), masked_bvals
                self.shell_indices[delta_mask] = shell_indices_ + max_index
                self.shell_bvalues.append(shell_bvalues_)
                max_index = max(self.shell_indices + 1)
            self.shell_bvalues = np.hstack(self.shell_bvalues)
            self.shell_b0_mask = self.shell_bvalues <= b0_threshold

            first_indices = [
                np.argmax(self.shell_indices == ind)
                for ind in np.arange(self.shell_indices.max() + 1)]
            self.shell_qvalues = None
            if self.qvalues is not None:
                self.shell_qvalues = self.qvalues[first_indices]
            self.shell_gradient_strengths = None
            if self.gradient_strengths is not None:
                self.shell_gradient_strengths = (
                    self.gradient_strengths[first_indices])
            self.shell_delta = None
            if self.delta is not None:
                self.shell_delta = self.delta[first_indices]
            self.shell_Delta = None
            if self.Delta is not None:
                self.shell_Delta = self.Delta[first_indices]
            self.shell_TE = None
            if self.TE is not None:
                self.shell_TE = self.TE[first_indices]
                if (len(np.unique(self.TE)) != len(np.unique(
                        self.TE[self.b0_mask]))):
                    msg = "Not every TE shell has b0 measurements.\n"
                    msg += "This is required to properly normalize the signal."
                    msg += " Make sure the TE values for b0-measurements have "
                    msg += "not defaulted to 0 for example."
                    raise ValueError(msg)
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
            self.shell_TE = TE

        # calculates observation matrices to convert spherical harmonic
        # coefficients to the positions on the sphere for every shell
        self.unique_dwi_indices = np.unique(self.shell_indices[~self.b0_mask])
        self.shell_sh_matrices = {}
        self.shell_sh_orders = np.zeros(len(self.shell_bvalues), dtype=int)
        for shell_index in self.unique_dwi_indices:
            shell_mask = self.shell_indices == shell_index
            bvecs_shell = self.gradient_directions[shell_mask]
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            self.shell_sh_orders[shell_index] = get_sh_order_from_bval(
                self.shell_bvalues[shell_index])
            self.shell_sh_matrices[shell_index] = real_sym_sh_mrtrix(
                self.shell_sh_orders[shell_index], theta_, phi_)[0]
        # warning in case there are no b0 measurements
        if sum(self.b0_mask) == 0:
            msg = "No b0 measurements were detected. Check if the b0_threshold"
            msg += " option is high enough, or if there is a mistake in the "
            msg += "acquisition design."
            warn(msg)

        self.spherical_mean_scheme = SphericalMeanAcquisitionScheme(
            self.shell_bvalues,
            self.shell_qvalues,
            self.shell_gradient_strengths,
            self.shell_Delta,
            self.shell_delta)
        if len(self.unique_dwi_indices) > 0:
            self.rotational_harmonics_scheme = (
                RotationalHarmonicsAcquisitionScheme(self)
            )

    @property
    def print_acquisition_info(self):
        """
        prints a small summary of the acquisition scheme. Is useful to check if
        the function correctly separated the shells and if the input parameters
        were given in the right scale.
        """
        print("Acquisition scheme summary\n")
        print("total number of measurements: {}".format(
            self.number_of_measurements))
        print("number of b0 measurements: {}".format(self.number_of_b0s))
        print("number of DWI shells: {}\n".format(
            np.sum(~self.shell_b0_mask)))
        upper_line = "shell_index |# of DWIs |bvalue [s/mm^2] "
        upper_line += "|gradient strength [mT/m] |delta [ms] |Delta[ms]"
        upper_line += " |TE[ms]"
        print(upper_line)
        for ind in np.arange(max(self.shell_indices) + 1):
            if (self.shell_TE is not None and
                self.shell_delta is not None and
                    self.shell_Delta is not None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        int(1e3 * self.shell_gradient_strengths[ind]),
                        self.shell_delta[ind] * 1e3,
                        self.shell_Delta[ind] * 1e3, self.shell_TE[ind] * 1e3))
            elif (self.shell_TE is None and
                  self.shell_delta is not None and
                    self.shell_Delta is not None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        int(1e3 * self.shell_gradient_strengths[ind]),
                        self.shell_delta[ind] * 1e3,
                        self.shell_Delta[ind] * 1e3, 'N/A'))
            elif (self.shell_TE is None and
                  self.shell_delta is None and
                    self.shell_Delta is not None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        'N/A', 'N/A', self.shell_Delta[ind] * 1e3, 'N/A'))
            elif (self.shell_TE is None and
                  self.shell_delta is not None and
                    self.shell_Delta is None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        'N/A', self.shell_delta[ind] * 1e3, 'N/A', 'N/A'))
            elif (self.shell_TE is None and
                  self.shell_delta is None and
                    self.shell_Delta is None):
                print(
                    "{:<12}|{:<10}|{:<16}|{:<25}|{:<11}|{:<10}|{:<5}".format(
                        str(ind), sum(self.shell_indices == ind),
                        int(self.shell_bvalues[ind] / 1e6),
                        'N/A', 'N/A', 'N/A', 'N/A'))

    def to_schemefile(self, filename):
        """
        Exports acquisition scheme information in schemefile format, which can
        be used by the Camino Monte-Carlo simulator.

        Parameters
        ----------
        filename : string,
            location at which to save the schemefile.
        """
        TE_ = self.TE
        if TE_ is None:
            TE_ = self.Delta + 2 * self.delta + 0.001
        schemefile_data = np.hstack(
            [self.gradient_directions,
             self.gradient_strengths[:, None],
             self.Delta[:, None],
             self.delta[:, None],
             TE_[:, None]])
        header = "#g_x  g_y  g_z  |G| DELTA delta TE\n"
        header += "VERSION: STEJSKALTANNER"
        np.savetxt(filename, schemefile_data,
                   header=header, comments='')

    def _rotational_harmonics_acquisition_scheme(
            self, angular_samples=10):
        """
        Calculates the acquisition scheme to return all the samples required to
        estimate the rotational harmonics of all the shells at once.

        Parameters
        ----------
        angular_samples: integer
            the number of angular samples that are sampled per shell.
        """
        thetas = np.linspace(0, np.pi / 2, angular_samples)
        r = np.ones(angular_samples)
        phis = np.zeros(angular_samples)
        angles = np.c_[r, thetas, phis]
        angles_cart = utils.sphere2cart(angles)

        Gdirs_all_shells = []
        G_all_shells = []
        delta_all_shells = []
        Delta_all_shells = []
        for G, delta, Delta in zip(self.shell_gradient_strengths,
                                   self.shell_delta, self.shell_Delta):
            Gdirs_all_shells.append(angles_cart)
            G_all_shells.append(np.tile(G, angular_samples))
            delta_all_shells.append(np.tile(delta, angular_samples))
            Delta_all_shells.append(np.tile(Delta, angular_samples))
        self.rh_acquisition_scheme = (
            acquisition_scheme_from_gradient_strengths(
                gradient_strengths=np.hstack(G_all_shells),
                gradient_directions=np.vstack(Gdirs_all_shells),
                delta=np.hstack(delta_all_shells),
                Delta=np.hstack(Delta_all_shells))
        )
        self.inverse_rh_matrix = {
            rh_order: np.linalg.pinv(real_sym_rh_basis(
                rh_order, thetas, phis
            )) for rh_order in np.arange(0, 15, 2)
        }

    def visualise_acquisition_G_Delta_rainbow(
            self,
            Delta_start=None, Delta_end=None, G_start=None, G_end=None,
            bval_isolines=np.r_[0, 250, 1000, 2500, 5000, 7500, 10000, 14000],
            alpha_shading=0.6
    ):
        """This function visualizes a q-tau acquisition scheme as a function of
        gradient strength and pulse separation (big_delta). It represents every
        measurements at its G and big_delta position regardless of b-vector,
        with a background of b-value isolines for reference. It assumes there
        is only one unique pulse length (small_delta) in the acquisition
        scheme.

        Parameters
        ----------
        Delta_start : float,
            optional minimum big_delta that is plotted in seconds
        Delta_end : float,
            optional maximum big_delta that is plotted in seconds
        G_start : float,
            optional minimum gradient strength that is plotted in T/m
        G_end : float,
            optional maximum gradient strength taht is plotted in T/m
        bval_isolines : array,
            optional array of bvalue isolines that are plotted in background
            given in s/mm^2
        alpha_shading : float between [0-1]
            optional shading of the bvalue colors in the background
        """
        Delta = self.Delta  # in seconds
        delta = self.delta  # in seconds
        G = self.gradient_strengths  # in SI units T/m

        if len(np.unique(delta)) > 1:
            msg = "This acquisition has multiple small_delta values. "
            msg += "This visualization assumes there is only one small_delta."
            raise ValueError(msg)

        if Delta_start is None:
            Delta_start = 0.005
        if Delta_end is None:
            Delta_end = Delta.max() + 0.004
        if G_start is None:
            G_start = 0.
        if G_end is None:
            G_end = G.max() + .05

        Delta_ = np.linspace(Delta_start, Delta_end, 50)
        G_ = np.linspace(G_start, G_end, 50)
        Delta_grid, G_grid = np.meshgrid(Delta_, G_)
        bvals_ = b_from_g(G_grid.ravel(), delta[0], Delta_grid.ravel()) / 1e6
        bvals_ = bvals_.reshape(G_grid.shape)

        plt.contourf(Delta_, G_, bvals_,
                     levels=bval_isolines,
                     cmap='rainbow', alpha=alpha_shading)
        cb = plt.colorbar(spacing="proportional")
        cb.ax.tick_params(labelsize=16)
        plt.scatter(Delta, G, c='k', s=25)

        plt.xlim(Delta_start, Delta_end)
        plt.ylim(G_start, G_end)
        cb.set_label('b-value ($s$/$mm^2$)', fontsize=18)
        plt.xlabel('Pulse Separation $\Delta$ [sec]', fontsize=18)
        plt.ylabel('Gradient Strength [T/m]', fontsize=18)

    def return_pruned_acquisition_scheme(self, shell_indices, data=None):
        "Returns pruned acquisition scheme and optionally also prunes data."
        booleans = []
        for index in shell_indices:
            booleans.append(self.shell_indices == index)
        mask = np.any(booleans, axis=0)

        bvals = self.bvalues[mask]
        gradient_directions = self.gradient_directions[mask]
        delta = self.delta[mask]
        Delta = self.Delta[mask]
        if self.TE is not None:
            TE = self.TE[mask]
        else:
            TE = None

        pruned_scheme = acquisition_scheme_from_bvalues(
            bvals, gradient_directions, delta, Delta, TE)
        if data is None:
            return pruned_scheme
        else:
            return pruned_scheme, data[..., mask]


class RotationalHarmonicsAcquisitionScheme:
    """
    AcquisitionScheme instance that contains the information necessary to
    calculate the rotational harmonics for a model for every acquisition shell.
    It is instantiated using a regular DmipyAcquisitionScheme and
    N_angular_samples determines how many samples are taken between mu=[0., 0.]
    and mu=[np.pi/2, 0.].

    Parameters
    ----------
    dmipy_acquisition_scheme: DmipyAcquisitionScheme instance
        An acquisition scheme that has been instantiated using dMipy.
    N_angular_samples: int
        Integer representing the number of angular samples per shell.
    """

    def __init__(self, dmipy_acquisition_scheme, N_angular_samples=10):
        self.Nsamples = N_angular_samples
        scheme = dmipy_acquisition_scheme

        thetas = np.linspace(0, np.pi / 2, N_angular_samples)
        r = np.ones(N_angular_samples)
        phis = np.zeros(N_angular_samples)
        angles = np.c_[r, thetas, phis]
        angles_cart = utils.sphere2cart(angles)

        b_all_shells = []
        Gdirs_all_shells = []
        delta_all_shells = []
        Delta_all_shells = []
        for shell_index in scheme.unique_dwi_indices:
            b = scheme.shell_bvalues[shell_index]
            b_all_shells.append(np.tile(b, N_angular_samples))
            if scheme.shell_delta is not None:
                delta = scheme.shell_delta[shell_index]
                delta_all_shells.append(np.tile(delta, N_angular_samples))
            if scheme.shell_Delta is not None:
                Delta = scheme.shell_Delta[shell_index]
                Delta_all_shells.append(np.tile(Delta, N_angular_samples))
            Gdirs_all_shells.append(angles_cart)

        self.bvalues = np.hstack(b_all_shells)
        self.gradient_directions = np.vstack(Gdirs_all_shells)
        self.delta = None
        if scheme.shell_delta is not None:
            self.delta = np.hstack(delta_all_shells)
        self.Delta = None
        if scheme.shell_Delta is not None:
            self.Delta = np.hstack(Delta_all_shells)
        if self.delta is not None and self.Delta is not None:
            self.gradient_strengths = g_from_b(
                self.bvalues,
                self.delta,
                self.Delta)
            self.qvalues = q_from_g(
                self.gradient_strengths,
                self.delta)
            self.tau = self.Delta - self.delta / 3.0
        else:
            self.gradient_strengths = self.qvalues = self.tau = None
        self.b0_mask = np.tile(False, len(self.bvalues))
        self.shell_delta = scheme.shell_delta
        self.shell_Delta = scheme.shell_Delta
        self.shell_sh_orders = (
            np.array(scheme.shell_sh_orders[scheme.unique_dwi_indices],
                     dtype=int))
        self.unique_dwi_indices = scheme.unique_dwi_indices
        self.number_of_measurements = len(self.bvalues)
        self.inverse_rh_matrix = {
            rh_order: np.linalg.pinv(real_sym_rh_basis(
                rh_order, thetas, phis
            )) for rh_order in np.arange(0, 15, 2)
        }


class SphericalMeanAcquisitionScheme:
    "Acquisition scheme for isotropic spherical mean models."

    def __init__(self, bvalues, qvalues,
                 gradient_strengths, Deltas, deltas):
        self.bvalues = bvalues
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        self.Delta = Deltas
        self.delta = deltas
        self.number_of_measurements = len(bvalues)


def acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta, TE=None,
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
    DmipyAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
        bvalues, delta, Delta, TE)
    check_acquisition_scheme(
        bvalues, gradient_directions, delta_, Delta_, TE_)
    if delta is not None and Delta is not None:
        qvalues = q_from_b(bvalues, delta_, Delta_)
        gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    else:
        qvalues = gradient_strengths = None
    return DmipyAcquisitionScheme(bvalues, gradient_directions, qvalues,
                                  gradient_strengths, delta_, Delta_, TE_,
                                  min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_qvalues(
        qvalues, gradient_directions, delta, Delta, TE=None,
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
    DmipyAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
        qvalues, delta, Delta, TE)
    check_acquisition_scheme(
        qvalues, gradient_directions, delta_, Delta_, TE_)
    bvalues = b_from_q(qvalues, delta, Delta)
    gradient_strengths = g_from_q(qvalues, delta)
    return DmipyAcquisitionScheme(bvalues, gradient_directions, qvalues,
                                  gradient_strengths, delta_, Delta_, TE_,
                                  min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_gradient_strengths(
        gradient_strengths, gradient_directions, delta, Delta, TE=None,
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
    DmipyAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(
        gradient_strengths, delta, Delta, TE)
    check_acquisition_scheme(gradient_strengths, gradient_directions,
                             delta_, Delta_, TE_)
    bvalues = b_from_g(gradient_strengths, delta, Delta)
    qvalues = q_from_g(gradient_strengths, delta)
    return DmipyAcquisitionScheme(bvalues, gradient_directions, qvalues,
                                  gradient_strengths, delta_, Delta_, TE_,
                                  min_b_shell_distance, b0_threshold)


def acquisition_scheme_from_schemefile(
        file_path, min_b_shell_distance=50e6, b0_threshold=10e6):
    r"""
    Created an acquisition scheme object from a Camino scheme file, containing
    gradient directions, strengths, pulse duration $\delta$ and pulse
    separation time $\Delta$ and TE.

    Parameters
    ----------
    file_path: string
        absolute file path to schemefile location
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    DmipyAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.
    """
    skiprows = 0
    while True:
        try:
            scheme = np.loadtxt(file_path, skiprows=skiprows)
            break
        except ValueError:
            skiprows += 1

    bvecs = scheme[:, :3]
    bvecs[np.linalg.norm(bvecs, axis=1) == 0.] = np.r_[1., 0., 0.]
    G = scheme[:, 3]
    Delta = scheme[:, 4]
    delta = scheme[:, 5]
    TE = scheme[:, 6]
    return acquisition_scheme_from_gradient_strengths(
        G, bvecs, delta, Delta, TE, min_b_shell_distance, b0_threshold)


def unify_length_reference_delta_Delta(reference_array, delta, Delta, TE):
    """
    If either delta or Delta are given as float, makes them an array the same
    size as the reference array.

    Parameters
    ----------
    reference_array : array of size (Nsamples)
        typically b-values, q-values or gradient strengths.
    delta : float or array of size (Nsamples)
        pulse duration in seconds.
    Delta : float or array of size (Nsamples)
        pulse separation in seconds.
    TE : None, float or array of size (Nsamples)
        Echo time of the acquisition in seconds.

    Returns
    -------
    delta_ : array of size (Nsamples)
        pulse duration copied to be same size as reference_array
    Delta_ : array of size (Nsamples)
        pulse separation copied to be same size as reference_array
    TE_ : None or array of size (Nsamples)
        Echo time copied to be same size as reference_array
    """
    if delta is None:
        delta_ = delta
    elif isinstance(delta, float) or isinstance(delta, int):
        delta_ = np.tile(delta, len(reference_array))
    else:
        delta_ = delta.copy()
    if Delta is None:
        Delta_ = Delta
    elif isinstance(Delta, float) or isinstance(Delta, int):
        Delta_ = np.tile(Delta, len(reference_array))
    else:
        Delta_ = Delta.copy()
    if TE is None:
        TE_ = TE
    elif isinstance(TE, float) or isinstance(TE, int):
        TE_ = np.tile(TE, len(reference_array))
    else:
        TE_ = TE.copy()
    return delta_, Delta_, TE_


def calculate_shell_bvalues_and_indices(bvalues, max_distance=20e6):
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
        bqg_values, gradient_directions, delta, Delta, TE):
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
    if delta is not None:
        if len(bqg_values) != len(delta):
            msg = "b/q/G input and delta must have the same length. "
            msg += "Currently their lengths are {} and {}.".format(
                len(bqg_values), len(delta)
            )
            raise ValueError(msg)
        if delta.ndim > 1:
            msg = "delta must be one-dimensional array. "
            msg += "Currently its dimension is {}".format(
                delta.ndim
            )
            raise ValueError(msg)
        if np.min(delta) < 0:
            msg = "delta must be zero or positive. "
            msg += "Currently its minimum value is {}.".format(
                np.min(delta)
            )
            raise ValueError(msg)
    if Delta is not None:
        if len(bqg_values) != len(Delta):
            msg = "b/q/G input and Delta must have the same length. "
            msg += "Currently their lengths are {} and {}.".format(
                len(bqg_values), len(Delta)
            )
            raise ValueError(msg)
        if Delta.ndim > 1:
            msg = "Delta must be one-dimensional array. "
            msg += "Currently its dimension is {}.".format(
                Delta.ndim
            )
            raise ValueError(msg)
        if np.min(Delta) < 0:
            msg = "Delta must be zero or positive. "
            msg += "Currently its minimum value is {}.".format(
                np.min(Delta)
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
    gradient_norms = np.linalg.norm(gradient_directions, axis=1)
    zero_norms = gradient_norms == 0.
    if not np.all(abs(gradient_norms[~zero_norms] - 1.) < 0.001):
        msg = "gradient orientations n are not unit vectors. "
        raise ValueError(msg)
    if TE is not None and len(TE) != len(bqg_values):
        msg = "If given, TE must be same length b/q/G input."
        msg += "Currently their lengths are {} and {}.".format(
            len(TE), len(gradient_directions)
        )


def gtab_dipy2dmipy(dipy_gradient_table, min_b_shell_distance=50e6,
                    b0_threshold=10e6):
    """Converts a dipy gradient_table to a dmipy acquisition_scheme.
    If no big_delta or small_delta is defined in the gradient table, then None
    is passed to the DmipyAcquisitionScheme for these fields, and no models
    can be used that need this information.

    Parameters
    ----------
    dipy_gradient_table: dipy GradientTable instance,
        object that contains bvals, bvecs, pulse separation and duration
        information.
    min_b_shell_distance : float
        minimum bvalue distance between different shells. This parameter is
        used to separate measurements into different shells, which is necessary
        for any model using spherical convolution or spherical mean.
    b0_threshold : float
        bvalue threshold for a measurement to be considered a b0 measurement.

    Returns
    -------
    DmipyAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.

    """
    if not isinstance(dipy_gradient_table, GradientTable):
        msg = "Input must be a dipy GradientTable object. "
        raise ValueError(msg)
    bvals = dipy_gradient_table.bvals * 1e6
    bvecs = dipy_gradient_table.bvecs
    delta = dipy_gradient_table.small_delta
    Delta = dipy_gradient_table.big_delta

    if delta is None or Delta is None:
        msg = "pulse_separation (big_delta) or pulse_duration (small_delta) "
        msg += "are not defined in the Dipy gtab. This means the resulting "
        msg += "DmipyAcquisitionScheme cannot be used with CompartmentModels "
        msg += "that need these."
        warn(msg)

    gtab_dmipy = acquisition_scheme_from_bvalues(
        bvalues=bvals, gradient_directions=bvecs, delta=delta, Delta=Delta,
        min_b_shell_distance=min_b_shell_distance, b0_threshold=b0_threshold)
    return gtab_dmipy


def gtab_dmipy2dipy(dmipy_gradient_table):
    """Converts a dmipy acquisition scheme to a dipy gradient_table.

    Parameters
    ----------
    DmipyAcquisitionScheme: acquisition scheme object
        contains all information of the acquisition scheme to be used in any
        microstructure model.

    Returns
    -------
    dipy_gradient_table: dipy GradientTable instance,
        object that contains bvals, bvecs, pulse separation and duration
        information.
    """
    if not isinstance(dmipy_gradient_table, DmipyAcquisitionScheme):
        msg = "Input must be a DmipyAcquisitionScheme object. "
        raise ValueError(msg)
    bvals = dmipy_gradient_table.bvalues / 1e6
    bvecs = dmipy_gradient_table.gradient_directions
    delta = dmipy_gradient_table.delta
    Delta = dmipy_gradient_table.Delta

    if len(np.unique(delta)) > 1:
        msg = "Cannot create Dipy GradientTable for Acquisition schemes with "
        msg += "multiple delta (pulse duration) values, due to current "
        msg += "limitations of Dipy GradientTables."
        raise ValueError(msg)
    elif len(np.unique(delta)) == 1:
        delta = delta[0]

    if len(np.unique(Delta)) > 1:
        msg = "Cannot create Dipy GradientTable for Acquisition schemes with "
        msg += "multiple Delta (pulse sepration) values, due to current "
        msg += "limitations of Dipy GradientTables."
        raise ValueError(msg)
    elif len(np.unique(Delta)) == 1:
        Delta = Delta[0]

    dipy_gradient_table = gradient_table(
        bvals=bvals, bvecs=bvecs, small_delta=delta, big_delta=Delta)
    return dipy_gradient_table
