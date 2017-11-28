# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division
import pkg_resources
from os.path import join

import numpy as np
from scipy import stats
from scipy import special
from dipy.reconst.shm import real_sym_sh_mrtrix

from microstruktur.utils import utils
from scipy.interpolate import bisplev
from microstruktur.core.modeling_framework import MicrostructureModel
from dipy.utils.optpkg import optional_package
numba, have_numba, _ = optional_package("numba")

SPHERICAL_INTEGRATOR = utils.SphericalIntegrator()
GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data/gradient_tables'
)
SIGNAL_MODELS_PATH = pkg_resources.resource_filename(
    'microstruktur', 'signal_models'
)
DATA_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data'
)
SPHERE_CARTESIAN = np.loadtxt(
    join(GRADIENT_TABLES_PATH, 'sphere_with_cap.txt')
)
SPHERE_SPHERICAL = utils.cart2sphere(SPHERE_CARTESIAN)
log_bingham_normalization_splinefit = np.load(
    join(DATA_PATH,
         "bingham_normalization_splinefit.npz"))['arr_0']
WATSON_SH_ORDER = 14


def get_sh_order_from_kappa(kappa):
    kappas = np.array([0.32323232, 1.29292929, 2.58585859, 4.36363636,
                       6.62626263, 9.37373737, np.inf])
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(kappas > kappa)]


class SD1Watson(MicrostructureModel):
    r""" The Watson spherical distribution model [1, 2].

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    kappa : float,
        concentration parameter of the Watson distribution.

    References
    ----------
    .. [1] Kaden et al.
           "Parametric spherical deconvolution: inferring anatomical
            connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Zhang et al.
           "NODDI: practical in vivo neurite orientation dispersion and density
            imaging of the human brain". NeuroImage (2012)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'kappa': (0, 64),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'kappa': 1.,
    }
    spherical_mean = False

    def __init__(self, mu=None, kappa=None):
        self.mu = mu
        self.kappa = kappa

    def __call__(self, n, **kwargs):
        r""" The Watson spherical distribution model [1, 2].

        Parameters
        ----------
        n : array of shape(3) or array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        Wn: float or array of shape(N),
            Probability density at orientations n, given mu and kappa.
        """
        kappa = kwargs.get('kappa', self.kappa)
        mu = kwargs.get('mu', self.mu)
        mu_cart = utils.unitsphere2cart_1d(mu)
        numerator = np.exp(kappa * np.dot(n, mu_cart) ** 2)
        denominator = 4 * np.pi * special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def spherical_harmonics_representation(self, sh_order=None, **kwargs):
        r""" The Watson spherical distribution model in spherical harmonics [1, 2].

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.
            we found 14 to be sufficient to represent concentrations of
            kappa=17.

        Returns
        -------
        watson_sh : array,
            spherical harmonics of Watson probability density.
        """
        kappa = kwargs.get('kappa', self.kappa)
        mu = kwargs.get('mu', self.mu)

        if sh_order is None:
            sh_order = get_sh_order_from_kappa(kappa)

        x_, y_, z_ = utils.unitsphere2cart_1d(mu)

        R = utils.rotation_matrix_001_to_xyz(x_, y_, z_)
        vertices_rotated = np.dot(SPHERE_CARTESIAN, R.T)
        _, theta_rotated, phi_rotated = utils.cart2sphere(vertices_rotated).T

        watson_sf = self(vertices_rotated, mu=mu, kappa=kappa)
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_rotated, phi_rotated)[0]
        sh_mat_inv = inverse_matrix(sh_mat)
        watson_sh = np.dot(sh_mat_inv, watson_sf)
        return watson_sh


class SD2Bingham(MicrostructureModel):
    r""" The Bingham spherical distribution model [1, 2, 3] using angles.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.

    References
    ----------
    .. [1] Kaden et al.
           "Parametric spherical deconvolution: inferring anatomical
            connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Sotiropoulos et al.
           "Ball and rackets: inferring fiber fanning from
            diffusion-weighted MRI". NeuroImage (2012)
    .. [3] Tariq et al.
           "Bingham--NODDI: Mapping anisotropic orientation dispersion of
            neurites using diffusion MRI". NeuroImage (2016)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'psi': (0, np.pi),
        'kappa': (0, 64),
        'beta': (0, 64)  # beta<=kappa in fact
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'psi': 1.,
        'kappa': 1.,
        'beta': 1.
    }
    spherical_mean = False

    def __init__(self, mu=None, psi=None, kappa=None, beta=None):
        self.mu = mu
        self.psi = psi
        self.kappa = kappa
        self.beta = beta

    def __call__(self, n, **kwargs):
        r""" The Watson spherical distribution model [1, 2].

        Parameters
        ----------
        n : array of shape(3) or array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        Bn: float or array of shape(N),
            Probability density at orientations n, given mu and kappa.
        """
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)

        R = utils.rotation_matrix_100_to_theta_phi_psi(mu[0], mu[1], psi)
        Bdiag = np.diag(np.r_[kappa, beta, 0])
        B = R.dot(Bdiag).dot(R.T)
        if np.ndim(n) == 1:
            numerator = np.exp(n.dot(B).dot(n))
        else:
            numerator = np.zeros(n.shape[0])
            for i, n_ in enumerate(n):
                numerator[i] = np.exp(n_.dot(B).dot(n_))

        # the denomator to normalize still needs to become a matrix argument B,
        # but function doesn't take it.
        # spherical_mean = estimate_spherical_mean_shell(numerator, n,
        #  sh_order=14)
        denominator = 4 * np.pi * self._get_normalization(kappa, beta)
        Bn = numerator / denominator
        return Bn

    def spherical_harmonics_representation(self, sh_order=14, **kwargs):
        r""" The Bingham spherical distribution model in spherical harmonics
        [1, 2].

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.
            we found 14 to be sufficient to represent concentrations of
            kappa=17.

        Returns
        -------
        bingham_sh : array,
            spherical harmonics of Watson probability density.
        """
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)

        x_, y_, z_ = utils.unitsphere2cart_1d(mu)

        R = utils.rotation_matrix_001_to_xyz(x_, y_, z_)
        vertices_rotated = np.dot(SPHERE_CARTESIAN, R.T)
        _, theta_rotated, phi_rotated = utils.cart2sphere(vertices_rotated).T

        bingham_sf = self(vertices_rotated, mu=mu, psi=psi, kappa=kappa,
                          beta=beta)

        sh_mat = real_sym_sh_mrtrix(sh_order, theta_rotated, phi_rotated)[0]
        sh_mat_inv = inverse_matrix(sh_mat)
        bingham_sh = np.dot(sh_mat_inv, bingham_sf)
        return bingham_sh

    def _get_normalization(self, kappa, beta):
        """
        The hyperconfluent function with matrix input is not available in
        python, so to normalize we estimated the bingham sphere function
        for kappa, beta in [0, 32] and estimated a 50x50 grid of its spherical
        means.

        Since the spherical mean of the bingham is similar to an exponential,
        we took its natural logarithm and fitted it to a 2D spline function.

        Below we use the fitted spline parameters in
        log_bingham_normalization_splinefit to interpolate the normalization
        for the distribution.

        code to generate the interpolation:

        from dipy.data import get_sphere, HemiSphere
        from microstruktur.signal_models.spherical_mean import (
            estimate_spherical_mean_shell)
        import numpy as np
        sphere = get_sphere()
        n = HemiSphere(sphere.x, sphere.y, sphere.z).subdivide().vertices
        R = np.eye(3)
        norm_size = 5
        numerator = np.zeros(n.shape[0])
        norm_grid = np.ones((norm_size, norm_size))
        kappa_beta_range = np.linspace(0, 32, norm_size)
        for i in np.arange(norm_size):
            for j in np.arange(i + 1):
                Bdiag = np.diag(np.r_[kappa_beta_range[i],
                                kappa_beta_range[j],
                                0])
                B = R.dot(Bdiag).dot(R.T)
                for k, n_ in enumerate(n):
                    numerator[k] = np.exp(n_.dot(B).dot(n_))
                norm_grid[i, j] = norm_grid[j, i] = (
                    estimate_spherical_mean_shell(
                        numerator, n, sh_order=12))
        log_norm_grid = np.log(norm_grid)
        kappa_grid, beta_grid = np.meshgrid(kappa_beta_range, kappa_beta_range)
        from scipy import interpolate
        tck = interpolate.bisplrep(kappa_grid.ravel(),
                                   beta_grid.ravel(),
                                   log_norm_grid.ravel(), s=0)
        np.savez("bingham_normalization_splinefit.npz", tck)

        Parameters
        ----------
        kappa : float,
            first concentration parameter of the Bingham distribution.
            defined as kappa = kappa1 - kappa3.
        beta : float,
            second concentration parameter of the Bingham distribution.
            defined as beta = kappa2 - kappa3. Bingham becomes Watson when
            beta=0.

        Returns
        -------
        bingham_normalization: float
            spherical mean / normalization of the bingham distribution
        """
        log_norm = bisplev(kappa, beta, log_bingham_normalization_splinefit)
        bingham_normalization = np.exp(log_norm)
        return bingham_normalization


class DD1GammaDistribution(MicrostructureModel):
    r"""A Gamma distribution of cylinder diameter for given alpha and beta
    parameters. NOTE: This is a distribution for axon DIAMETER and not SURFACE.
    To simulate the diffusion signal of an ensemble of gamma-distributed
    cylinders the probability still needs to be corrected for cylinder surface
    by multiplying by np.pi * radius ** 2 and renormalizing [1]. Reason being
    that diffusion signals are generated by the volume of spins inside axons
    (cylinders), which is proportional to cylinder surface and not to diameter.

    Parameters
    ----------
    alpha : float,
        shape of the gamma distribution.
    beta : float,
        scale of the gamma distrubution. Different from Bingham distribution!

    References
    ----------
    .. [1] Assaf, Yaniv, et al. "AxCaliber: a method for measuring axon
        diameter distribution from diffusion MRI." Magnetic resonance in
        medicine 59.6 (2008): 1347-1354.
    """
    _parameter_ranges = {
        'alpha': (1e-10, np.inf),
        'beta': (1e-10, np.inf)
    }
    _parameter_scales = {
        'alpha': 1.,
        'beta': 1.,
    }
    spherical_mean = False

    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, diameter, **kwargs):
        r"""
        Parameters
        ----------
        diameter : float or array, shape (N)
            cylinder (axon) diameter in meters.

        Returns
        -------
        Pgamma : float or array, shape (N)
            probability of cylinder diameter for given alpha and beta.
        """
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)
        radius = diameter / 2.
        gamma_dist = stats.gamma(alpha, scale=beta)
        Pgamma = gamma_dist.pdf(radius)
        return Pgamma


def inverse_matrix(matrix):
    return np.dot(np.linalg.inv(np.dot(matrix.T, matrix)), matrix.T)


if have_numba:
    inverse_matrix = numba.njit()(inverse_matrix)
    get_sh_order_from_kappa = numba.njit()(get_sh_order_from_kappa)
