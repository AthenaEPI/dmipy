# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division

import numpy as np
from scipy.special import erf

from ..utils import utils
from ..core.modeling_framework import ModelProperties
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12

__all__ = [
    'G1Ball',
    'G2Zeppelin',
    'G3TemporalZeppelin'
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

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped), 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[0],
                    E_reshaped[i])
            )
        return rh_array

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
        'mu': ([0, np.pi], [-np.pi, np.pi]),
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

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped),
                             rh_scheme.shell_sh_orders.max() // 2 + 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[sh_order],
                    E_reshaped[i])
            )
        return rh_array

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
            rh_array = self.rotational_harmonics_representation(
                acquisition_scheme, **kwargs)
            E_mean[acquisition_scheme.unique_dwi_indices] = (
                rh_array[:, 0] / (2 * np.sqrt(np.pi))
            )
        return E_mean


class G3TemporalZeppelin(ModelProperties):
    r"""
    The temporal Zeppelin model [1]_ - an axially symmetric Tensor - typically
    used to describe extra-axonal diffusion. The G3TemporalZeppelin differs
    from G2Zeppelin in that it has a time-dependent perpendicular parameter
    "A", which describe extra-axonal diffusion hindrance due to axon packing,
    and that lambda_perp is instead called lambda_inf, as it describes the
    perpendicular diffusivity when diffusion time is infinite.

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
        characteristic coefficient in 10^12 m^2

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
        'mu': ([0, np.pi], [-np.pi, np.pi]),
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
        'lambda_inf': 'normal',
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
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped),
                             rh_scheme.shell_sh_orders.max() // 2 + 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[sh_order],
                    E_reshaped[i])
            )
        return rh_array

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
            rh_array = self.rotational_harmonics_representation(
                acquisition_scheme, **kwargs)
            E_mean[acquisition_scheme.unique_dwi_indices] = (
                rh_array[:, 0] / (2 * np.sqrt(np.pi))
            )
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
