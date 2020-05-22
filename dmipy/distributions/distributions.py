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

from dmipy.utils import utils
from scipy import interpolate
from dmipy.core.modeling_framework import ModelProperties
from dipy.utils.optpkg import optional_package
from dipy.data import get_sphere, HemiSphere
sphere = get_sphere('symmetric724')
hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)

numba, have_numba, _ = optional_package("numba")

GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'dmipy', 'data/gradient_tables'
)
SIGNAL_MODELS_PATH = pkg_resources.resource_filename(
    'dmipy', 'signal_models'
)
DATA_PATH = pkg_resources.resource_filename(
    'dmipy', 'data'
)
SPHERE_CARTESIAN = np.loadtxt(
    join(GRADIENT_TABLES_PATH, 'sphere_with_cap.txt')
)
SPHERE_SPHERICAL = utils.cart2sphere(SPHERE_CARTESIAN)
log_bingham_normalization_splinefit = np.load(
    join(DATA_PATH, "bingham_normalization_splinefit.npz"),
    encoding='bytes', allow_pickle=True)['arr_0']

inverse_sh_matrix_kernel = {
    sh_order: np.linalg.pinv(real_sym_sh_mrtrix(
        sh_order, hemisphere.theta, hemisphere.phi
    )[0]) for sh_order in np.arange(0, 15, 2)
}
BETA_SCALING = 1e-6

__all__ = [
    'get_sh_order_from_odi',
    'SD1Watson',
    'SD2Bingham',
    'DD1Gamma',
    'odi2kappa',
    'kappa2odi'
]


def get_sh_order_from_odi(odi):
    "Returns minimum sh_order to estimate spherical harmonics for given odi."
    odis = np.array([0.80606061, 0.46666667, 0.25333333,
                     0.15636364, 0.09818182, 0.06909091, 0.])
    sh_orders = np.arange(2, 15, 2)
    return sh_orders[np.argmax(odis < odi)]


class SD1Watson(ModelProperties):
    r""" The Watson spherical distribution model [1]_ [2]_.

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
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'odi': (0.02, 0.99),
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'odi': 1.,
    }
    _parameter_types = {
        'mu': 'orientation',
        'odi': 'normal'
    }
    _model_type = 'SphericalDistribution'

    def __init__(self, mu=None, odi=None):
        self.mu = mu
        self.odi = odi

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
        odi = kwargs.get('odi', self.odi)
        mu = kwargs.get('mu', self.mu)

        kappa = odi2kappa(odi)
        mu_cart = utils.unitsphere2cart_1d(mu)
        numerator = np.exp(kappa * np.dot(n, mu_cart) ** 2)
        denominator = 4 * np.pi * special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def spherical_harmonics_representation(self, sh_order=None, **kwargs):
        r""" The Watson spherical distribution model in spherical harmonics.
        The minimum order is automatically derived from numerical experiments
        to ensure fast function executation and accurate results.

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        watson_sh : array,
            spherical harmonics of Watson probability density.
        """
        odi = kwargs.get('odi', self.odi)
        mu = kwargs.get('mu', self.mu)
        if sh_order is None:
            sh_order = get_sh_order_from_odi(odi)

        watson_sf = self(hemisphere.vertices, mu=mu, odi=odi)
        sh_mat_inv = inverse_sh_matrix_kernel[sh_order]
        watson_sh = np.dot(sh_mat_inv, watson_sf)
        return watson_sh


class SD2Bingham(ModelProperties):
    r""" The Bingham spherical distribution model [1]_ [2]_ [3]_ using angles.

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
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'psi': (0, np.pi),
        'odi': (0.02, 0.99),
        'beta_fraction': (0, 1)  # beta<=kappa in fact
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'psi': 1.,
        'odi': 1.,
        'beta_fraction': 1.
    }
    _parameter_types = {
        'mu': 'orientation',
        'psi': 'circular',
        'odi': 'normal',
        'beta_fraction': 'normal'
    }
    _model_type = 'SphericalDistribution'

    def __init__(self, mu=None, psi=None, odi=None, beta_fraction=None):
        self.mu = mu
        self.psi = psi
        self.odi = odi
        self.beta_fraction = beta_fraction

    def __call__(self, n, **kwargs):
        r""" The Watson spherical distribution model.

        Parameters
        ----------
        n : array of shape(3) or array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        Bn: float or array of shape(N),
            Probability density at orientations n, given mu and kappa.
        """
        odi = kwargs.get('odi', self.odi)
        beta_fraction = kwargs.get('beta_fraction', self.beta_fraction)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)

        kappa = odi2kappa(odi)
        beta = beta_fraction * kappa

        mu_cart = utils.unitsphere2cart_1d(mu)

        R = utils.rotation_matrix_100_to_theta_phi_psi(mu[0], mu[1], psi)
        mu_beta = R.dot(np.r_[0., 1., 0.])
        numerator = _probability_bingham(kappa, beta, mu_cart, mu_beta, n)
        denominator = 4 * np.pi * self._get_normalization(kappa, beta)
        Bn = numerator / denominator
        return Bn

    def spherical_harmonics_representation(self, sh_order=None, **kwargs):
        r""" The Bingham spherical distribution model in spherical harmonics.
        The minimum order is automatically derived from numerical experiments
        to ensure fast function executation and accurate results.

        Parameters
        ----------
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.

        Returns
        -------
        bingham_sh : array,
            spherical harmonics of Bingham probability density.
        """
        odi = kwargs.get('odi', self.odi)
        beta_fraction = kwargs.get('beta_fraction', self.beta_fraction)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        if sh_order is None:
            sh_order = get_sh_order_from_odi(odi)

        bingham_sf = self(hemisphere.vertices, mu=mu, psi=psi, odi=odi,
                          beta_fraction=beta_fraction)

        sh_mat_inv = inverse_sh_matrix_kernel[sh_order]
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
        from dmipy.signal_models.spherical_mean import (
            estimate_spherical_mean_shell)
        import numpy as np
        sphere = get_sphere()
        n = HemiSphere(sphere.x, sphere.y, sphere.z).subdivide().vertices
        R = np.eye(3)
        norm_size = 50
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
        log_norm = interpolate.bisplev(kappa, beta,
                                       log_bingham_normalization_splinefit)
        bingham_normalization = np.exp(log_norm)
        return bingham_normalization


class SD3SphericalHarmonics(ModelProperties):
    r"""A real-valued spherical harmonics distribution.

    Parameters
    ----------
    sh_order: int,
        maximum spherical harmonics order.
    sh_coeff
    """

    def __init__(self, sh_order, sh_coeff=None):
        self.sh_order = sh_order
        self.N_coeff = int((sh_order + 2) * (sh_order + 1) // 2)
        if sh_coeff is not None:
            if len(sh_coeff) != self.N_coeff:
                msg = 'if given, sh_coeff length must correspond to N_coeffs '\
                      'associated with sh_order ({} vs {}).'
                raise ValueError(msg.format(len(sh_coeff, self.N_coeff)))
        self.sh_coeff = sh_coeff

        self._parameter_ranges = {'sh_coeff': [
            [None, None] for i in range(self.N_coeff)]}
        self._parameter_scales = {'sh_coeff':
                                  np.ones(self.N_coeff, dtype=float)}
        self._parameter_cardinality = {'sh_coeff': self.N_coeff}
        self._parameter_types = {'sh_coeff': 'sh_coefficients'}
        self._parameter_optimization_flags = {'sh_coeff': True}

    def __call__(self, n, **kwargs):
        r"""Returns the sphere function at cartesian orientations n given
        spherical harmonic coefficients.

        Parameters
        ----------
        n : array of shape(N x 3),
            sampled orientations of the Watson distribution.

        Returns
        -------
        SHn: array of shape(N),
            Probability density at orientations n, given sh coeffs.
        """
        # calculate SHT matrix
        _, theta, phi = utils.cart2sphere(n).T
        SHT = real_sym_sh_mrtrix(self.sh_order, theta, phi)[0]
        # transform coefficients to sphere values
        sh_coeff = kwargs.get('sh_coeff', self.sh_coeff)
        SHn = SHT.dot(sh_coeff)
        return SHn

    def spherical_harmonics_representation(self, **kwargs):
        r"""Returns the spherical harmonic coefficients themselves.
        """
        return kwargs.get('sh_coeff', self.sh_coeff)


class DD1Gamma(ModelProperties):
    r"""A Gamma distribution of cylinder diameter for given alpha and beta
    parameters. NOTE: This is a distribution for axon DIAMETER and not SURFACE.
    To simulate the diffusion signal of an ensemble of gamma-distributed
    cylinders the probability still needs to be corrected for cylinder surface
    by multiplying by np.pi * radius ** 2 and renormalizing [1]_. Reason being
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
        'alpha': (0.1, 30.),
        'beta': (1e-3, 2)
    }
    _parameter_scales = {
        'alpha': 1.,
        'beta': BETA_SCALING,
    }
    _parameter_types = {
        'alpha': 'normal',
        'beta': 'normal'
    }
    _model_type = 'SpatialDistribution'

    def __init__(self, alpha=None, beta=None, Nsteps=30,
                 normalization='standard'):
        self.alpha = alpha
        self.beta = beta
        self.Nsteps = Nsteps

        if normalization == 'standard':
            self.norm_func = self.unity
        elif normalization == 'plane':
            self.norm_func = self.length_plane
        elif normalization == 'cylinder':
            self.norm_func = self.surface_cylinder
        elif normalization == 'sphere':
            self.norm_func = self.volume_sphere
        else:
            msg = "Unknown normalization {}".format(normalization)
            raise ValueError(msg)
        self.calculate_sampling_start_and_end_points(self.norm_func)

    def length_plane(self, radius):
        "The distance normalization function for planes."
        return 2 * radius

    def surface_cylinder(self, radius):
        "The surface normalization function for cylinders."
        return np.pi * radius ** 2

    def volume_sphere(self, radius):
        "The volume normalization function for spheres."
        return (4. / 3.) * np.pi * radius ** 3

    def unity(self, radius):
        "The standard normalization for the Gamma distribution (none)."
        return np.ones(len(radius))

    def calculate_sampling_start_and_end_points(self, norm_func, gridsize=50):
        """
        For a given normalization function calculates the best start and end
        points to sample for all possible values of alpha, beta. This is done
        to make sure the function does not sample where the probability of
        basically zero.

        The function is based on first doing a dense sampling and then finding
        out which points need to be included to have sampled at least 99% of
        the area under the probability density curve.

        It sets two interpolator functions that can be called for any
        combination of alpha,beta and to return the appropriate start and end
        sampling points.

        Parameters
        ----------
        norm_func : normalization function,
            normalization of the model, depends on if it's a sphere/cylinder.
        gridsize : integer,
            value that decides how big the grid will be on which we define the
            start and end sampling points.
        """
        start_grid = np.ones([gridsize, gridsize])
        end_grid = np.ones([gridsize, gridsize])

        alpha_range = (np.array(self._parameter_ranges['alpha']) *
                       self._parameter_scales['alpha'])
        beta_range = (np.array(self._parameter_ranges['beta']) *
                      self._parameter_scales['beta'])

        alpha_linspace = np.linspace(alpha_range[0], alpha_range[1], gridsize)
        beta_linspace = np.linspace(beta_range[0], beta_range[1], gridsize)

        for i, alpha in enumerate(alpha_linspace):
            for j, beta in enumerate(beta_linspace):
                gamma_distribution = stats.gamma(alpha, scale=beta)
                outer_limit = (
                    gamma_distribution.mean() + 9 * gamma_distribution.std())
                x_grid = np.linspace(1e-8, outer_limit, 500)
                pdf = gamma_distribution.pdf(x_grid)
                pdf *= norm_func(x_grid)
                cdf = np.cumsum(pdf)
                cdf /= cdf.max()
                inverse_cdf = np.cumsum(pdf[::-1])[::-1]
                inverse_cdf /= inverse_cdf.max()
                end_grid[i, j] = x_grid[np.argmax(cdf > 0.995)]
                start_grid[i, j] = x_grid[np.argmax(inverse_cdf < 0.995)]
        start_grid = np.clip(start_grid, 1e-8, np.inf)
        end_grid = np.clip(end_grid, 1e-7, np.inf)

        alpha_grid, beta_grid = np.meshgrid(alpha_linspace, beta_linspace)

        self.start_interpolator = interpolate.bisplrep(alpha_grid.ravel(),
                                                       beta_grid.ravel(),
                                                       start_grid.T.ravel(),
                                                       kx=2, ky=2)

        self.end_interpolator = interpolate.bisplrep(alpha_grid.ravel(),
                                                     beta_grid.ravel(),
                                                     end_grid.T.ravel(),
                                                     kx=2, ky=2)

    def __call__(self, **kwargs):
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

        gamma_dist = stats.gamma(alpha, scale=beta)
        start_point = interpolate.bisplev(alpha, beta, self.start_interpolator)
        end_point = interpolate.bisplev(alpha, beta, self.end_interpolator)
        start_point = max(start_point, 1e-8)
        radii = np.linspace(start_point, end_point, self.Nsteps)
        normalization = self.norm_func(radii)
        radii_pdf = gamma_dist.pdf(radii)
        radii_pdf_area = radii_pdf * normalization
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapz(x=radii, y=radii_pdf_area)
        )
        return radii, radii_pdf_normalized


def _probability_bingham(kappa, beta, mu, mu_beta, n):
    "Non-normalized probability of the Bingham distribution."
    return np.exp(kappa * np.dot(n, mu) ** 2 +
                  beta * np.dot(n, mu_beta) ** 2)


def odi2kappa(odi):
    "Calculates concentration (kappa) from orientation dispersion index (odi)."
    return 1. / np.tan(odi * (np.pi / 2.0))


def kappa2odi(kappa):
    "Calculates orientation dispersion index (odi) from concentration (kappa)."
    return (2. / np.pi) * np.arctan(1. / kappa)


if have_numba:
    get_sh_order_from_odi = numba.njit()(get_sh_order_from_odi)
    _probability_bingham = numba.njit()(_probability_bingham)
