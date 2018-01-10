# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division

import numpy as np

from mipy.utils import utils
from mipy.core.acquisition_scheme import SimpleAcquisitionSchemeRH
from mipy.core.modeling_framework import MicrostructureModel
from mipy.utils.spherical_convolution import real_sym_rh_basis
from mipy.utils.utils import sphere2cart
from dipy.utils.optpkg import optional_package

numba, have_numba, _ = optional_package("numba")

samples = 10
thetas = np.linspace(0, np.pi / 2, samples)
r = np.ones(samples)
phis = np.zeros(samples)
angles = np.c_[r, thetas, phis]
angles_cart = sphere2cart(angles)

inverse_rh_matrix_kernel = {
    rh_order: np.linalg.pinv(real_sym_rh_basis(
        rh_order, thetas, phis
    )) for rh_order in np.arange(0, 15, 2)
}
simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(None, angles_cart)

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12


class G3Ball(MicrostructureModel):
    r""" The Ball model [1] - an isotropic Tensor with one diffusivity.

    Parameters
    ----------
    lambda_iso : float,
        isotropic diffusivity in 10^9 m^2/s.

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
    spherical_mean = False

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        bvals = acquisition_scheme.bvalues
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        E_ball = np.exp(-bvals * lambda_iso)
        return E_ball


class G4Zeppelin(MicrostructureModel):
    r""" The Zeppelin model [1] - an axially symmetric Tensor - for
    extra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    lambda_perp : float,
        perpendicular diffusivity in 10^9 m^2/s.

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
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

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
        E_zeppelin = attenuation_zeppelin(
            bvals, lambda_par, lambda_perp, n, mu)
        return E_zeppelin

    def rotational_harmonics_representation(self, bvalue, rh_order=14, **kwargs):
        r""" The Stick model in rotational harmonics, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernelfor spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/m^2.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.
            set to 14 to conform with order used for watson distribution.

        Returns
        -------
        rh : array,
            rotational harmonics of stick model aligned with z-axis.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        simple_acq_scheme_rh.bvalues[:] = bvalue
        E_kernel_sf = self(simple_acq_scheme_rh, mu=[0., 0.],
                           lambda_par=lambda_par, lambda_perp=lambda_perp)
        rh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        return rh


class G5RestrictedZeppelin(MicrostructureModel):
    r""" The restricted Zeppelin model [1] - an axially symmetric Tensor - for
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
    .. [1] Burcaw
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
        'lambda_perp': DIFFUSIVITY_SCALING,
        'A': A_SCALING
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, lambda_inf=None, A=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_inf = lambda_inf
        self.A = A

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

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
            self, bvalue, delta=None, Delta=None, rh_order=14, **kwargs):
        r""" The model in rotational harmonics, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernelfor spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/m^2.
        delta: float,
            delta parameter in seconds.
        Delta: float,
            Delta parameter in seconds.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.
            set to 14 to conform with order used for watson distribution.

        Returns
        -------
        rh : array,
            rotational harmonics of the model aligned with z-axis.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)
        simple_acq_scheme_rh.bvalues[:] = bvalue
        simple_acq_scheme_rh.delta[:] = delta
        simple_acq_scheme_rh.Delta[:] = Delta

        E_kernel_sf = self(simple_acq_scheme_rh, mu=[0., 0.],
                           lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
        rh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        return rh


def attenuation_zeppelin(bvals, lambda_par, lambda_perp, n, mu):
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
    attenuation_zeppelin = numba.njit()(attenuation_zeppelin)
