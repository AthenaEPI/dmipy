# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division
import numpy as np
from scipy.special import erf
from mipy.core.modeling_framework import ModelProperties

DIFFUSIVITY_SCALING = 1e-9
A_SCALING = 1e-12


class C1StickSphericalMean(ModelProperties):
    """ Spherical mean of the signal attenuation of the Stick model [1] for
    a given b-value and parallel diffusivity. Analytic expression from
    Eq. (7) in [2].

    Parameters
    ----------
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.

    References
    ----------
    .. [1] Behrens et al.
        "Characterization and propagation of uncertainty in
        diffusion-weighted MR imaging"
        Magnetic Resonance in Medicine (2003)
    .. [2] Kaden et al. "Multi-compartment microscopic diffusion imaging."
        NeuroImage 139 (2016): 346-359.
    """

    _parameter_ranges = {
        'lambda_par': (.1, 3)
    }
    _parameter_scales = {
        'lambda_par': DIFFUSIVITY_SCALING,
    }
    spherical_mean = True
    _model_type = 'other'

    def __init__(self, mu=None, lambda_par=None):
        self.lambda_par = lambda_par

    def __call__(self, acquisition_scheme, **kwargs):
        """
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E_mean : float,
            spherical mean of the Stick model.
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


class G2ZeppelinSphericalMean(ModelProperties):
    """ Spherical mean of the signal attenuation of the Zeppelin model
        for a given b-value and parallel and perpendicular diffusivity.
        Analytic expression from Eq. (8) in [1]).

        Parameters
        ----------
        lambda_par : float,
            parallel diffusivity in 10^9 m^2/s.
        lambda_perp : float,
            perpendicular diffusivity in 10^9 m^2/s.

        References
        ----------
        .. [1] Kaden et al. "Multi-compartment microscopic diffusion imaging."
            NeuroImage 139 (2016): 346-359.
        """

    _parameter_ranges = {
        'lambda_par': (.1, 3),
        'lambda_perp': (.1, 3)
    }
    _parameter_scales = {
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING
    }
    spherical_mean = True
    _model_type = 'other'

    def __init__(self, lambda_par=None, lambda_perp=None):
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, acquisition_scheme, **kwargs):
        """
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model.
        """
        bvals = acquisition_scheme.shell_bvalues
        bvals_ = bvals[~acquisition_scheme.shell_b0_mask]

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)

        E_mean = np.ones_like(bvals)
        exp_bl = np.exp(-bvals_ * lambda_perp)
        sqrt_bl = np.sqrt(bvals_ * (lambda_par - lambda_perp))
        E_mean_ = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
        E_mean[~acquisition_scheme.shell_b0_mask] = E_mean_
        return E_mean


class G3RestrictedZeppelinSphericalMean(ModelProperties):
    """ Spherical mean of the signal attenuation of the restricted Zeppelin
        model [1] for a given b-value, parallel and perpendicular diffusivity,
        and characteristic coefficient A. The function is the same as the
        zeppelin spherical mean [2] but lambda_perp is replaced with the
        restricted function.

        Parameters
        ----------
        lambda_par : float,
            parallel diffusivity in 10^9 m^2/s.
        lambda_inf : float,
            bulk diffusivity constant 10^9 m^2/s.
        A: float,
            characteristic coefficient in 10^6 m^2

        References
        ----------
        .. [1] Burcaw, L.M., Fieremans, E., Novikov, D.S., 2015. Mesoscopic
            structure of neuronal tracts from time-dependent diffusion.
            NeuroImage 114, 18.
        .. [2] Kaden et al. "Multi-compartment microscopic diffusion imaging."
            NeuroImage 139 (2016): 346-359.
        """

    _parameter_ranges = {
        'lambda_par': (.1, 3),
        'lambda_inf': (.1, 3),
        'A': (0, 10)
    }
    _parameter_scales = {
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_inf': DIFFUSIVITY_SCALING,
        'A': A_SCALING
    }
    spherical_mean = True
    _model_type = 'other'

    def __init__(self, lambda_par=None, lambda_inf=None, A=None):
        self.lambda_par = lambda_par
        self.lambda_inf = lambda_inf
        self.A = A

    def __call__(self, acquisition_scheme, **kwargs):
        """
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model.
        """
        bvals = acquisition_scheme.shell_bvalues
        delta = acquisition_scheme.shell_delta
        Delta = acquisition_scheme.shell_Delta
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_inf = kwargs.get('lambda_inf', self.lambda_inf)
        A = kwargs.get('A', self.A)

        restricted_term = (
            A * (np.log(Delta / delta) + 3 / 2.) / (Delta - delta / 3.)
        )
        lambda_perp = lambda_inf + restricted_term
        exp_bl = np.exp(-bvals * lambda_perp)
        sqrt_bl = np.sqrt(bvals * (lambda_par - lambda_perp))
        E_mean = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
        return E_mean
