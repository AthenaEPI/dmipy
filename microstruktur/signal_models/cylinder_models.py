from __future__ import division
import pkg_resources
from os.path import join

import numpy as np
from scipy import special
from dipy.reconst.shm import real_sym_sh_mrtrix

from microstruktur.utils import utils
from microstruktur.core.constants import CONSTANTS
from microstruktur.utils.spherical_convolution import kernel_sh_to_rh
from ..core.acquisition_scheme import SimpleAcquisitionSchemeRH
from microstruktur.core.modeling_framework import MicrostructureModel

SPHERICAL_INTEGRATOR = utils.SphericalIntegrator()
GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data/gradient_tables'
)
SIGNAL_MODELS_PATH = pkg_resources.resource_filename(
    'microstruktur', 'signal_models'
)
SPHERE_CARTESIAN = np.loadtxt(
    join(GRADIENT_TABLES_PATH, 'sphere_with_cap.txt')
)
SPHERE_SPHERICAL = utils.cart2sphere(SPHERE_CARTESIAN)
inverse_rh_matrix_kernel = {
    rh_order: np.linalg.pinv(real_sym_sh_mrtrix(
        rh_order, SPHERE_SPHERICAL[:, 1], SPHERE_SPHERICAL[:, 2]
    )[0]) for rh_order in np.arange(0, 15, 2)
}
WATSON_SH_ORDER = 14
DIFFUSIVITY_SCALING = 1e-9
DIAMETER_SCALING = 1e-6
A_SCALING = 1e-12


class C1Stick(MicrostructureModel):
    r""" The Stick model [1] - a cylinder with zero radius - for
    intra-axonal diffusion.

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

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

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

        lambda_par_ = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        E_stick = np.exp(-bvals * lambda_par_ * np.dot(n, mu) ** 2)
        return E_stick

    def rotational_harmonics_representation(self, bvalue, rh_order=14):
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
        simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(
            bvalue, SPHERE_CARTESIAN)
        E_kernel_sf = self(simple_acq_scheme_rh, mu=np.r_[0., 0.])
        sh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        rh = kernel_sh_to_rh(sh)
        return rh


class C2CylinderSodermanApproximation(MicrostructureModel):
    r""" The Soderman model [1]_ - a cylinder with finite radius - for
    intra-axonal diffusion. Assumes that the pulse length
    is infinitely short and the diffusion time is infinitely long.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in m^2/s.
    diameter : float,
        axon (cylinder) diameter in meters.

    Returns
    -------
    E : array, shape (N,)
        signal attenuation

    References
    ----------
    .. [1]_ Soderman, Olle, and Bengt Jonsson. "Restricted diffusion in
            cylindrical geometry." Journal of Magnetic Resonance, Series A
            117.1 (1995): 94-97.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-10, 50e-6)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    spherical_mean = False

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
        radius = diameter / 2
        # Eq. [6] in the paper
        E = ((2 * special.jn(1, 2 * np.pi * q * radius)) ** 2 /
             (2 * np.pi * q * radius) ** 2)
        return E

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
        q = acquisition_scheme.qvalues

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par_ = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_parallel = np.dot(n, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = np.exp(-bvals * lambda_par_ * magnitude_parallel ** 2)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular
        q_nonzero = q_perp > 0  # only q>0 attenuate
        E_perpendicular[q_nonzero] = self.perpendicular_attenuation(
            q_perp[q_nonzero], diameter
        )
        return E_parallel * E_perpendicular

    def rotational_harmonics_representation(self, bvalue, delta, Delta,
                                            rh_order=14):
        r""" The Stick model in rotational harmonics, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernelfor spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/m^2.
        delta: float,
            pulse duration in seconds.
        Delta: float,
            pulse separation in seconds.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.
            set to 14 to conform with order used for watson distribution.

        Returns
        -------
        rh : array,
            rotational harmonics of stick model aligned with z-axis.
        """
        simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(
            bvalue, SPHERE_CARTESIAN, delta=delta, Delta=Delta)
        E_kernel_sf = self(simple_acq_scheme_rh, mu=np.r_[0., 0.])
        sh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        rh = kernel_sh_to_rh(sh)
        return rh


class C3CylinderCallaghanApproximation(MicrostructureModel):
    r""" The Callaghan model [1]_ - a cylinder with finite radius - for
    intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk.

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


    References
    ----------
    .. [1]_ Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
            cylindrical, and spherical pores under conditions of wall
            relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
            53-59.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-10, 50e-6)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    spherical_mean = False

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
        gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
        number_of_roots=10,
        number_of_functions=10,
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.diameter = diameter

        self.alpha = np.empty((number_of_roots, number_of_functions))
        self.alpha[0, 0] = 0
        if number_of_roots > 1:
            self.alpha[1:, 0] = special.jnp_zeros(0, number_of_roots - 1)
        for m in xrange(1, number_of_functions):
            self.alpha[:, m] = special.jnp_zeros(m, number_of_roots)

        self.xi = np.array(range(number_of_roots)) * np.pi
        self.zeta = np.array(range(number_of_roots)) * np.pi + np.pi / 2.0

    def perpendicular_attenuation(self, q, tau, diameter):
        """Implements the finite time Callaghan model for cylinders [1]
        """
        radius = diameter / 2.
        alpha = self.alpha
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)

        J = special.j1(q_argument) ** 2
        for k in xrange(0, self.alpha.shape[0]):
            alpha2 = alpha[k, 0] ** 2
            update = (
                4 * np.exp(-alpha2 * self.diffusion_perpendicular *
                           tau / radius ** 2) *
                q_argument_2 /
                (q_argument_2 - alpha2) ** 2 * J
            )
            res += update

        for m in xrange(1, self.alpha.shape[1]):
            J = special.jvp(m, q_argument, 1)
            q_argument_J = (q_argument * J) ** 2
            for k in xrange(self.alpha.shape[0]):
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
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par_ = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_parallel = np.dot(n, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = np.exp(-bvals * lambda_par_ * magnitude_parallel ** 2)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular

        q_nonzero = q_perp > 0
        E_perpendicular[q_nonzero] = self.perpendicular_attenuation(
            q_perp[q_nonzero], tau[q_nonzero], diameter
        )
        return E_parallel * E_perpendicular

    def rotational_harmonics_representation(
            self, bvalue, delta=None, Delta=None, rh_order=14):
        r""" The Stick model in rotational harmonics, such that Y_lm = Yl0.
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
            rotational harmonics of stick model aligned with z-axis.
        """
        simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(
            bvalue, SPHERE_CARTESIAN, delta=delta, Delta=Delta)
        E_kernel_sf = self(simple_acq_scheme_rh, mu=np.r_[0., 0.])
        sh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        rh = kernel_sh_to_rh(sh)
        return rh


class C4CylinderGaussianPhaseApproximation(MicrostructureModel):
    r""" The Gaussian phase model [1]_ - a cylinder with finite radius - for
    intra-axonal diffusion. The perpendicular diffusion is modelled
    after Van Gelderen's solution for the disk.

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
    .. [1]_ Van Gelderen et al. "Evaluation of Restricted Diffusion in
            Cylinders. Phosphocreatine in Rabbit Leg Muscle"
            Journal of Magnetic Resonance Series B (1994)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-10, 50e-6)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    spherical_mean = False
    CYLINDER_TRASCENDENTAL_ROOTS = np.sort(special.jnp_zeros(1, 1000))

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant'],
        gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.diameter = diameter

    def perpendicular_attenuation(
        self, gradient_strength, delta, Delta, diameter
    ):
        D = self.diffusion_perpendicular
        gamma = self.gyromagnetic_ratio
        radius = diameter / 2

        first_factor = -2 * (gradient_strength * gamma) ** 2
        alpha = self.CYLINDER_TRASCENDENTAL_ROOTS / radius
        alpha2 = alpha ** 2
        alpha2D = alpha2 * D

        summands = (
            2 * alpha2D * delta - 2 +
            2 * np.exp(-alpha2D * delta) +
            2 * np.exp(-alpha2D * Delta) -
            np.exp(-alpha2D * (Delta - delta)) -
            np.exp(-alpha2D * (Delta + delta))
        ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))

        E = np.exp(
            first_factor *
            summands.sum()
        )

        return E

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
        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par_ = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_parallel = np.dot(n, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = np.exp(-bvals * lambda_par_ * magnitude_parallel ** 2)
        E_perpendicular = np.ones_like(g)
        g_perp = g * magnitude_perpendicular

        # select unique delta, Delta combinations
        deltas = np.c_[delta, Delta]
        temp = np.ascontiguousarray(deltas).view(
            np.dtype((np.void, deltas.dtype.itemsize * deltas.shape[1]))
        )
        deltas_unique = np.unique(temp).view(deltas.dtype).reshape(
            -1, deltas.shape[1]
        )

        g_nonzero = g_perp > 0
        # for every unique combination get the perpendicular attenuation
        for delta_, Delta_ in deltas_unique:
            mask = np.all([g_nonzero, delta == delta_, Delta == Delta_],
                          axis=0)
            E_perpendicular[mask] = self.perpendicular_attenuation(
                g_perp[mask], delta_, Delta_, diameter
            )
        return E_parallel * E_perpendicular

    def rotational_harmonics_representation(
            self, bvalue, delta=None, Delta=None, rh_order=14):
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
            rotational harmonics of stick model aligned with z-axis.
        """
        simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(
            bvalue, SPHERE_CARTESIAN, delta=delta, Delta=Delta)
        E_kernel_sf = self(simple_acq_scheme_rh, mu=np.r_[0., 0.])
        sh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        rh = kernel_sh_to_rh(sh)
        return rh
