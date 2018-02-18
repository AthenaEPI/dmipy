# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division

import numpy as np
from scipy.special import erf

from ..utils import utils
from ..core.acquisition_scheme import SimpleAcquisitionSchemeRH
from ..core.modeling_framework import ModelProperties
from ..utils.spherical_convolution import real_sym_rh_basis
from ..utils.utils import sphere2cart
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

_samples = 10
_thetas = np.linspace(0, np.pi / 2, _samples)
_r = np.ones(_samples)
_phis = np.zeros(_samples)
_angles = np.c_[_r, _thetas, _phis]
_angles_cart = sphere2cart(_angles)

inverse_rh_matrix_kernel = {
    rh_order: np.linalg.pinv(real_sym_rh_basis(
        rh_order, _thetas, _phis
    )) for rh_order in np.arange(0, 15, 2)
}
simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(_angles_cart)

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12

__all__ = [
    'G1Ball',
    'G2Zeppelin',
    'G3RestrictedZeppelin'
]


class G1Ball(ModelProperties):
    r""" The Ball model [1]_ - an isotropic Tensor with one diffusivity.

    Parameters
    ----------
    lambda_iso : float,
        isotropic diffusivity in m^2/s.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """

    _parameter_ranges = {
        'lambda_iso': (.1, 3)
    }
    _parameter_scales = {
        'lambda_iso': DIFFUSIVITY_SCALING
    }
    _parameter_types = {
        'lambda_iso': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

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
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        E_ball = np.exp(-bvals * lambda_iso)
        return E_ball

    def rotational_harmonics_representation(self, bvalue, **kwargs):
        r"""
        The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/m^2.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        rh : array,
            rotational harmonics of stick model aligned with z-axis.
        """
        rh_order = 0
        simple_acq_scheme_rh.bvalues.fill(bvalue)
        E_kernel_sf = self(simple_acq_scheme_rh, **kwargs)
        rh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        return rh

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the model for every acquisition shell.
        """
        return self(acquisition_scheme.spherical_mean_scheme, **kwargs)


class G2Zeppelin(ModelProperties):
    r""" The Zeppelin model [1]_ - an axially symmetric Tensor - typically used
    for extra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    lambda_perp : float,
        perpendicular diffusivity in m^2/s.

    Returns
    -------
    E_zeppelin : float or array, shape(N),
        signal attenuation.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_perp': (.1, 3)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'lambda_perp': 'normal',
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

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
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        E_zeppelin = _attenuation_zeppelin(
            bvals, lambda_par, lambda_perp, n, mu)
        return E_zeppelin

    def rotational_harmonics_representation(self, bvalue, rh_order=14,
                                            **kwargs):
        r"""
        The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/m^2.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        rh : array,
            rotational harmonics of stick model aligned with z-axis.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        simple_acq_scheme_rh.bvalues.fill(bvalue)
        E_kernel_sf = self(simple_acq_scheme_rh, mu=[0., 0.],
                           lambda_par=lambda_par, lambda_perp=lambda_perp)
        rh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        return rh

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model for every acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues[
            ~acquisition_scheme.shell_b0_mask]

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)

        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)
        if lambda_par > lambda_perp:  # use [kaden et al. 2016]
            exp_bl = np.exp(-bvals * lambda_perp)
            sqrt_bl = np.sqrt(bvals * (lambda_par - lambda_perp))
            E_mean_ = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
            E_mean[~acquisition_scheme.shell_b0_mask] = E_mean_
        else:  # estimate spherical mean using rotational harmonics
            for shell_index in acquisition_scheme.unique_dwi_indices:
                rh = self.rotational_harmonics_representation(
                    bvalue=acquisition_scheme.shell_bvalues[shell_index],
                    rh_order=acquisition_scheme.shell_sh_orders[shell_index],
                    **kwargs)
                E_mean[shell_index] = rh[0] / (2 * np.sqrt(np.pi))
        return E_mean


class G3RestrictedZeppelin(ModelProperties):
    r"""
    The restricted Zeppelin model [1]_ - an axially symmetric Tensor - for
    extra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    lambda_inf : float,
        bulk diffusivity constant 10^9 m^2/s.
    A: float,
        characteristic coefficient in 10^6 m^2

    Returns
    -------
    E_zeppelin : float or array, shape(N),
        signal attenuation.

    References
    ----------
    .. [1] Burcaw, L.M., Fieremans, E., Novikov, D.S., 2015. Mesoscopic
        structure of neuronal tracts from time-dependent diffusion.
        NeuroImage 114, 18.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_inf': (.1, 3),
        'A': (0, 10)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_inf': DIFFUSIVITY_SCALING,
        'A': A_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'lambda_perp': 'normal',
        'A': 'normal'
    }
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, lambda_par=None, lambda_inf=None, A=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_inf = lambda_inf
        self.A = A

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
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)

        R1 = mu
        R2 = utils.perpendicular_vector(R1)
        R3 = np.cross(R1, R2)
        R = np.c_[R1, R2, R3]

        E_zeppelin = np.ones_like(bvals)
        for i, (bval_, n_, delta_, Delta_) in enumerate(
            zip(bvals, n, delta, Delta)
        ):
            # lambda_perp and A must be in the same unit
            restricted_term = (
                A * (np.log(Delta_ / delta_) + 3 / 2.) / (Delta_ - delta_ / 3.)
            )
            D_perp = lambda_inf + restricted_term
            D_h = np.diag(np.r_[lambda_par, D_perp, D_perp])
            D = np.dot(np.dot(R, D_h), R.T)
            E_zeppelin[i] = np.exp(-bval_ * np.dot(n_, np.dot(n_, D)))
        return E_zeppelin

    def rotational_harmonics_representation(
            self, bvalue, delta, Delta, rh_order=14, **kwargs):
        r""" The model in rotational harmonics, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/m^2.
        delta: float,
            pulse length in seconds.
        Delta: float,
            pulse separation in seconds.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        rh : array,
            rotational harmonics of the model aligned with z-axis.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        simple_acq_scheme_rh.bvalues.fill(bvalue)
        simple_acq_scheme_rh.delta.fill(delta)
        simple_acq_scheme_rh.Delta.fill(Delta)

        E_kernel_sf = self(simple_acq_scheme_rh, mu=[0., 0.],
                           lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
        rh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        return rh

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Restricted Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Restricted Zeppelin model for every
            acquisition shell.
        """
        bvals = acquisition_scheme.shell_bvalues[
            ~acquisition_scheme.shell_b0_mask]
        delta = acquisition_scheme.shell_delta[
            ~acquisition_scheme.shell_b0_mask]
        Delta = acquisition_scheme.shell_Delta[
            ~acquisition_scheme.shell_b0_mask]
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)

        restricted_term = (
            A * (np.log(Delta / delta) + 3 / 2.) / (Delta - delta / 3.)
        )
        lambda_perp = lambda_inf + restricted_term
        if lambda_par > lambda_perp.max():  # use modified [kaden et al. 2016]
            exp_bl = np.exp(-bvals * lambda_perp)
            sqrt_bl = np.sqrt(bvals * (lambda_par - lambda_perp))
            E_mean[~acquisition_scheme.shell_b0_mask] = (
                exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl))
        else:  # estimate spherical mean using rotational harmonics
            for shell_index in acquisition_scheme.unique_dwi_indices:
                rh = self.rotational_harmonics_representation(
                    bvalue=acquisition_scheme.shell_bvalues[shell_index],
                    delta=acquisition_scheme.shell_delta[shell_index],
                    Delta=acquisition_scheme.shell_Delta[shell_index],
                    rh_order=acquisition_scheme.shell_sh_orders[shell_index],
                    **kwargs)
                E_mean[shell_index] = rh[0] / (2 * np.sqrt(np.pi))
        return E_mean


def _attenuation_zeppelin(bvals, lambda_par, lambda_perp, n, mu):
    "Signal attenuation for Zeppelin model."
    mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
    magnitude_parallel = np.dot(n, mu)
    proj = np.dot(mu_perpendicular_plane, n.T)
    magnitude_perpendicular = np.sqrt(
        proj[0] ** 2 + proj[1] ** 2 + proj[2] ** 2)
    E_zeppelin = np.exp(-bvals *
                        (lambda_par * magnitude_parallel ** 2 +
                         lambda_perp * magnitude_perpendicular ** 2)
                        )
    return E_zeppelin


if have_numba:
    _attenuation_zeppelin = numba.njit()(_attenuation_zeppelin)
