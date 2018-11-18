from __future__ import division

import numpy as np
from scipy import special
from scipy.special import erf

from ..utils import utils
from ..core.constants import CONSTANTS
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import AnisotropicSignalModelProperties
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

DIFFUSIVITY_SCALING = 1e-9
DIAMETER_SCALING = 1e-6
A_SCALING = 1e-12


__all__ = [
    'C1Stick',
    'C2CylinderStejskalTannerApproximation',
    'C3CylinderCallaghanApproximation',
    'C4CylinderGaussianPhaseApproximation'
]


class C1Stick(ModelProperties, AnisotropicSignalModelProperties):
    r""" The Stick model [1]_ - a cylinder with zero radius - typically used
    for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """

    _required_acquisition_parameters = ['bvalues', 'gradient_directions']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal'
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions

        lambda_par_ = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        E_stick = _attenuation_parallel_stick(bvals, lambda_par_, n, mu)
        return E_stick

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Stick model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : array of size (Nshells)
            spherical mean of the Stick model for every acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues
        bvals_ = bvals[~acquisition_scheme.shell_b0_mask]

        lambda_par = kwargs.get('lambda_par', self.lambda_par)

        E_mean = np.ones_like(bvals)
        bval_indices_above0 = bvals > 0
        bvals_ = bvals[bval_indices_above0]
        E_mean_ = ((np.sqrt(np.pi) * erf(np.sqrt(bvals_ * lambda_par))) /
                   (2 * np.sqrt(bvals_ * lambda_par)))
        E_mean[~acquisition_scheme.shell_b0_mask] = E_mean_
        return E_mean


class C2CylinderStejskalTannerApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Stejskal-Tanner approximation of the cylinder model with finite
    radius, proposed by Soderman and Jonsson [1]_. Assumes that both the short
    gradient pulse (SGP) approximation is met and long diffusion time limit is
    reached. The perpendicular cylinder diffusion therefore only depends on the
    q-value of the acquisition.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    diameter : float,
        cylinder diameter in meters.

    Returns
    -------
    E : array, shape (N,)
        signal attenuation

    References
    ----------
    .. [1] Soderman, Olle, and Bengt Jonsson. "Restricted diffusion in
            cylindrical geometry." Journal of Magnetic Resonance, Series A
            117.1 (1995): 94-97.
    """

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'qvalues']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder'
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter

    def perpendicular_attenuation(
        self, q, diameter
    ):
        "Returns the cylinder's perpendicular signal attenuation."
        radius = diameter / 2
        # Eq. [6] in the paper
        E = ((2 * special.jn(1, 2 * np.pi * q * radius)) ** 2 /
             (2 * np.pi * q * radius) ** 2)
        return E

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        q = acquisition_scheme.qvalues

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular
        q_nonzero = q_perp > 0  # only q>0 attenuate
        E_perpendicular[q_nonzero] = self.perpendicular_attenuation(
            q_perp[q_nonzero], diameter
        )
        return E_parallel * E_perpendicular


class C3CylinderCallaghanApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Callaghan model [1]_ - a cylinder with finite radius - typically
    used for intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk. Is dependent on both q-value
    and diffusion time.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.
    diffusion_perpendicular : float,
        the intra-cylindrical, perpenicular diffusivity. By default it is set
        to a typical value for intra-axonal diffusion as 1.7e-9 m^2/s.
    number_of_roots : integer,
        number of roots to use for the Callaghan cylinder model.
    number_of_function : integer,
        number of functions to use for the Callaghan cylinder model.

    References
    ----------
    .. [1] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
            cylindrical, and spherical pores under conditions of wall
            relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
            53-59.
    """

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions', 'qvalues', 'tau']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder'
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots=20,
        number_of_functions=50,
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.diameter = diameter

        self.alpha = np.empty((number_of_roots, number_of_functions))
        self.alpha[0, 0] = 0
        if number_of_roots > 1:
            self.alpha[1:, 0] = special.jnp_zeros(0, number_of_roots - 1)
        for m in range(1, number_of_functions):
            self.alpha[:, m] = special.jnp_zeros(m, number_of_roots)

    def perpendicular_attenuation(self, q, tau, diameter):
        "Implements the finite time Callaghan model for cylinders"
        radius = diameter / 2.
        alpha = self.alpha
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)

        J = special.j1(q_argument) ** 2
        for k in range(0, self.alpha.shape[0]):
            alpha2 = alpha[k, 0] ** 2
            update = (
                4 * np.exp(-alpha2 * self.diffusion_perpendicular *
                           tau / radius ** 2) *
                q_argument_2 /
                (q_argument_2 - alpha2) ** 2 * J
            )
            res += update

        for m in range(1, self.alpha.shape[1]):
            J = special.jvp(m, q_argument, 1)
            q_argument_J = (q_argument * J) ** 2
            for k in range(self.alpha.shape[0]):
                alpha2 = self.alpha[k, m] ** 2
                update = (
                    8 * np.exp(-alpha2 * self.diffusion_perpendicular *
                               tau / radius ** 2) *
                    alpha2 / (alpha2 - m ** 2) *
                    q_argument_J /
                    (q_argument_2 - alpha2) ** 2
                )
                res += update
        return res

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular

        q_nonzero = q_perp > 0
        E_perpendicular[q_nonzero] = self.perpendicular_attenuation(
            q_perp[q_nonzero], tau[q_nonzero], diameter
        )
        return E_parallel * E_perpendicular


class C4CylinderGaussianPhaseApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Gaussian phase model [1]_ - a cylinder with finite radius -
    typically used for intra-axonal diffusion. The perpendicular diffusion is
    modelled after Van Gelderen's solution for the disk. It is dependent on
    gradient strength, pulse separation and pulse length.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.


    References
    ----------
    .. [1] Van Gelderen et al. "Evaluation of Restricted Diffusion in
            Cylinders. Phosphocreatine in Rabbit Leg Muscle"
            Journal of Magnetic Resonance Series B (1994)
    """

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions',
        'gradient_strengths', 'delta', 'Delta']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder'
    }
    _model_type = 'CompartmentModel'
    _CYLINDER_TRASCENDENTAL_ROOTS = np.sort(special.jnp_zeros(1, 100))

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant']
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self.diameter = diameter

    def perpendicular_attenuation(
        self, gradient_strength, delta, Delta, diameter
    ):
        "Calculates the cylinder's perpendicular signal attenuation."
        D = self.diffusion_perpendicular
        gamma = self.gyromagnetic_ratio
        return _attenuation_perpendicular_gaussian_phase(
            diameter, gradient_strength, delta, Delta,
            D, gamma, self._CYLINDER_TRASCENDENTAL_ROOTS)

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu)
        E_perpendicular = np.ones_like(g)
        g_perp = g * magnitude_perpendicular

        g_nonzero = g_perp > 0
        # for every unique combination get the perpendicular attenuation
        unique_deltas = np.unique([acquisition_scheme.shell_delta,
                                   acquisition_scheme.shell_Delta], axis=1)
        for delta_, Delta_ in zip(*unique_deltas):
            mask = np.all([g_nonzero, delta == delta_, Delta == Delta_],
                          axis=0)
            E_perpendicular[mask] = self.perpendicular_attenuation(
                g_perp[mask], delta_, Delta_, diameter
            )
        return E_parallel * E_perpendicular


def _attenuation_parallel_stick(bvals, lambda_par, n, mu):
    "Free gaussian diffusion for parallel cylinder direction."
    return np.exp(-bvals * lambda_par * np.dot(n, mu) ** 2)


def _attenuation_perpendicular_gaussian_phase(
        diameter, gradient_strength, delta, Delta,
        D, gamma, CYLINDER_TRASCENDENTAL_ROOTS):
    "Perpendicular Gaussian Phase signal attenuation."
    radius = diameter / 2.
    first_factor = -2 * (gradient_strength * gamma) ** 2
    alpha = CYLINDER_TRASCENDENTAL_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    summands = (
        2 * alpha2D * delta - 2 +
        2 * np.exp(-alpha2D * delta) +
        2 * np.exp(-alpha2D * Delta) -
        np.exp(-alpha2D * (Delta - delta)) -
        np.exp(-alpha2D * (Delta + delta))
    ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))

    E = np.exp(first_factor * summands.sum())
    return E


if have_numba:
    _attenuation_parallel_stick = numba.njit()(_attenuation_parallel_stick)
    _attenuation_perpendicular_gaussian_phase = numba.njit()(
        _attenuation_perpendicular_gaussian_phase)
