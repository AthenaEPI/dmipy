'''
Document Module
'''
from __future__ import division
import pkg_resources
from os.path import join
from collections import OrderedDict
from itertools import chain

import numpy as np
from scipy.special import erf
from scipy import stats
from scipy import integrate
from scipy import special
from dipy.core.geometry import cart2sphere, sphere2cart
from dipy.reconst.shm import real_sym_sh_mrtrix

from . import utils
from .free_diffusion import free_diffusion_attenuation
from . import CONSTANTS
from ..signal_models.spherical_convolution import kernel_sh_to_rh

SPHERICAL_INTEGRATOR = utils.SphericalIntegrator()
GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data/gradient_tables'
)
SPHERE_CARTESIAN = np.loadtxt(
    join(GRADIENT_TABLES_PATH, 'sphere_with_cap.txt')
)
SPHERE_SPHERICAL = np.vstack(cart2sphere(*SPHERE_CARTESIAN.T))


class MicrostrukturModel:
    @property
    def parameter_ranges(self):
        return self._parameter_ranges.copy()

    @property
    def parameter_constraints(self):
        return self._parameter_constraints()

    @property
    def parameter_cardinality(self):
        if hasattr(self, '_parameter_cardinality'):
            return self._parameter_cardinality

        self._parameter_cardinality = OrderedDict({
            k: len(np.atleast_2d(self.parameter_ranges[k]))
            for k in sorted(self.parameter_ranges)
        })
        return self._parameter_cardinality

    def parameter_vector_to_parameters(self, parameter_vector):
        parameters = {}
        current_pos = 0
        for parameter, card in self.parameter_cardinality.items():
            parameters[parameter] = parameter_vector[
                current_pos: current_pos + card
            ]
            current_pos += card
        return parameters

    def parameters_to_parameter_vector(self, **parameters):
        parameter_vector = []
        for parameter, card in self.parameter_cardinality.items():
            parameter_vector.append(parameters[parameter])
        return np.hstack(parameter_vector)

    def objective_function(
        self, parameter_vector,
        bvals=None, n=None, attenuation=None
    ):
        parameters = self.parameter_vector_to_parameters(parameter_vector)
        return np.sum((
            self(bvals, n, **parameters) - attenuation
        ) ** 2) / len(attenuation)

    @property
    def bounds_for_optimization(self):
        bounds = []
        for parameter, card in self.parameter_cardinality.items():
            range_ = self.parameter_ranges[parameter]
            if card == 1:
                bounds.append(range_)
            else:
                for i in range(card):
                    bounds.append((range_[0][i], range_[1][i]))
        return bounds


class PartialVolumeCombinedMicrostrukturModel(MicrostrukturModel):
    def __init__(
        self, models, partial_volumes=None,
        parameter_links=[], optimise_partial_volumes=False
    ):
        if partial_volumes is None:
            partial_volumes = np.repeat(1 / len(models), len(models) - 1)

        model_counts = {}
        self.model_names = []
        self.models = models
        self.partial_volumes = partial_volumes
        self.optimise_partial_volumes = optimise_partial_volumes

        for model in models:
            if model.__class__ not in model_counts:
                model_counts[model.__class__] = 1
            else:
                model_counts[model.__class__] += 1

            self.model_names.append(
                '{}_{}_'.format(
                    model.__class__.__name__,
                    model_counts[model.__class__]
                )
            )

        self._parameter_ranges = {
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_ranges.items()
        }

        self._parameter_map = {
            model_name + k: (model, k)
            for model, model_name in zip(self.models, self.model_names)
            for k in model.parameter_ranges
        }
        self.parameter_defaults = {}
        for model_name, model in zip(self.model_names, self.models):
            for parameter in model.parameter_ranges:
                self.parameter_defaults[model_name + parameter] = getattr(
                    model, parameter
                )

        self._parameter_cardinality = self.parameter_cardinality
        if self.optimise_partial_volumes:
            self.partial_volume_names = [
                'partial_volume_{}'.format(i)
                for i in range(len(self.partial_volumes))
            ]

            for i, partial_volume_name in enumerate(self.partial_volume_names):
                self._parameter_ranges[partial_volume_name] = (0, 1)
                self.parameter_defaults[partial_volume_name] = (
                    1 / (len(self.partial_volumes) + 1 - i)
                )
                self._parameter_map[partial_volume_name] = (
                    None, partial_volume_name
                )
                self._parameter_cardinality[partial_volume_name] = 1

        self._inverted_parameter_map = {
            v: k for k, v in self._parameter_map.items()
        }

        self.parameter_links = parameter_links
        for i, parameter_function in enumerate(parameter_links):
            parameter_model, parameter_name, parameter_function, arguments = \
                parameter_function

            if (
                (parameter_model, parameter_name)
                not in self._inverted_parameter_map
            ):
                raise ValueError(
                    "Parameter function {} doesn't exist".format(i)
                )

            parameter_name = self._inverted_parameter_map[
                (parameter_model, parameter_name)
            ]

            del self._parameter_ranges[parameter_name]
            del self.parameter_defaults[parameter_name]
            del self._parameter_cardinality[parameter_name]

    def add_linked_parameters_to_parameters(self, parameters):
        if len(self.parameter_links) == 0:
            return parameters
        parameters = parameters.copy()
        for parameter in self.parameter_links:
            parameter_model, parameter_name, parameter_function, arguments = \
                parameter
            parameter_name = self._inverted_parameter_map[
                (parameter_model, parameter_name)
            ]
            argument_values = []
            for argument in arguments:
                argument_name = self._inverted_parameter_map[argument]
                argument_values.append(parameters.get(
                    argument_name,
                    self.parameter_defaults[argument_name]
                ))

            parameters[parameter_name] = parameter_function(*argument_values)
        return parameters

    def __call__(self, bvals, n, **kwargs):
        values = 0
        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )

        accumulated_partial_volume = 1
        if self.optimise_partial_volumes:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = self.partial_volumes

        for model_name, model, partial_volume in zip(
            self.model_names, self.models,
            chain(partial_volumes, (None,))
        ):
            parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self._inverted_parameter_map[
                    (model, parameter)
                ]
                parameters[parameter] = kwargs.get(
                    parameter_name, self.parameter_defaults.get(parameter_name)
                )
            current_partial_volume = accumulated_partial_volume
            if partial_volume is not None:
                current_partial_volume *= partial_volume
                accumulated_partial_volume *= (1 - partial_volume)

            values = (
                values +
                current_partial_volume * model(bvals, n, **parameters)
            )
        return values


class I1Stick(MicrostrukturModel):
    r""" The Stick model [1] - a cylinder with zero radius - for
    intra-axonal diffusion.

    Parameters
    ----------
    bvals : float or array, shape(N),
        b-values in s/mm^2.
    n : array, shape(N x 3),
        b-vectors in cartesian coordinates.
    mu : array, shape(3),
        unit vector representing orientation of the Stick.
    lambda_par : float,
        parallel diffusivity in mm^2/s.


    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, np.inf)
    }

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

    def __call__(self, bvals, n, **kwargs):
        r'''
        Parameters
        ----------
        bvals : float or array, shape(N),
            b-values in s/mm^2.
        n : array, shape(N x 3),
            b-vectors in cartesian coordinates.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        x, y, z = sphere2cart(1, mu[0], mu[1])
        mu = np.r_[float(x), float(y), float(z)]
        E_stick = np.exp(-bvals * lambda_par * np.dot(n, mu) ** 2)
        return E_stick

    def rotational_harmonics_representation(self, bval, rh_order=14, **kwargs):
        r""" The Stick model in rotational harmonics, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernelfor spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/mm^2.
        lambda_par : float,
            parallel diffusivity in mm^2/s.
        sh_order : int,
            maximum spherical harmonics order to be used in the approximation.
            set to 14 to conform with order used for watson distribution.

        Returns
        -------
        rh : array,
            rotational harmonics of stick model aligned with z-axis.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)

        E_stick_sf = self(
            np.r_[bval], SPHERE_CARTESIAN,
            mu=np.r_[0., 0.], lambda_par=lambda_par
        )
        sh_mat = real_sym_sh_mrtrix(
            rh_order, SPHERE_SPHERICAL[1], SPHERE_SPHERICAL[2]
        )[0]
        sh_mat_inv = np.linalg.pinv(sh_mat)
        sh = np.dot(sh_mat_inv, E_stick_sf)
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh

    def spherical_mean(self, bval, **kwargs):
        """ Spherical mean of the signal attenuation of the Stick model for
        a given b-value and parallel diffusivity. Analytic expression from
        Eq. (7) in [1].

        Parameters
        ----------
        bval : float,
            b-value in s/mm^2.
        lambda_par : float,
            parallel diffusivity in mm^2/s.

        Returns
        -------
        E_mean : float,
            spherical mean of the Stick model.

        References
        ----------
        .. [1] Kaden et al. "Multi-compartment microscopic diffusion imaging."
           NeuroImage 139 (2016): 346-359.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        E_mean = ((np.sqrt(np.pi) * erf(np.sqrt(bval * lambda_par))) /
                  (2 * np.sqrt(bval * lambda_par)))
        return E_mean


class E3Ball(MicrostrukturModel):
    r""" The Ball model [1] - an isotropic Tensor with one diffusivity.

    Parameters
    ----------
    bvals : float or array, shape(N),
        b-values in s/mm^2.
    lambda_iso : float,
        isotropic diffusivity in mm^2/s.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """

    _parameter_ranges = {
        'lambda_iso': (0, np.inf)
    }

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

    def __call__(self, bvals, **kwargs):
        r'''
        Parameters
        ----------
        bvals : float or array, shape(N),
            b-values in s/mm^2.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''

        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        E_ball = np.exp(-bvals * lambda_iso)
        return E_ball


class E4Zeppelin(MicrostrukturModel):
    r""" The Zeppelin model [1] - an axially symmetric Tensor - for
    extra-axonal diffusion.

    Parameters
    ----------
    bvals : float or array, shape(N),
        b-values in s/mm^2.
    n : array, shape(N x 3),
        b-vectors in cartesian coordinates.
    mu : array, shape(3),
        unit vector representing orientation of the Stick.
    lambda_par : float,
        parallel diffusivity in mm^2/s.
    lambda_perp : float,
        perpendicular diffusivity in mm^2/s.

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
        'lambda_par': (0, np.inf),
        'lambda_perp': (0, np.inf)
    }

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, bvals, n, **kwargs):
        r'''
        Parameters
        ----------
        bvals : float or array, shape(N),
            b-values in s/mm^2.
        n : array, shape(N x 3),
            b-vectors in cartesian coordinates.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        x, y, z = sphere2cart(1, mu[0], mu[1])
        mu = np.r_[float(x), float(y), float(z)]

        D_h = np.diag(np.r_[lambda_par, lambda_perp, lambda_perp])
        R1 = mu
        R2 = utils.perpendicular_vector(R1)
        R3 = np.cross(R1, R2)
        R = np.c_[R1, R2, R3]
        D = np.dot(np.dot(R, D_h), R.T)

        dim_b = np.ndim(bvals)
        dim_n = np.ndim(n)

        if dim_n == 1:  # if there is only one sampled orientation
            E_zeppelin = np.exp(-bvals * np.dot(n, np.dot(n, D)))
        elif dim_b == 0 and dim_n == 2:  # one b-value on sphere
            E_zeppelin = np.zeros(n.shape[0])
            for i in range(n.shape[0]):
                E_zeppelin[i] = np.exp(-bvals * np.dot(n[i], np.dot(n[i], D)))
        elif dim_b == 1 and dim_n == 2:  # many b-values and orientations
            E_zeppelin = np.zeros(n.shape[0])
            for i in range(n.shape[0]):
                E_zeppelin[i] = np.exp(-bvals[i] * np.dot(n[i],
                                                          np.dot(n[i], D)))
        return E_zeppelin

    def rotational_harmonics_representation(self, bval, rh_order=14, **kwargs):
        r""" The Stick model in rotational harmonics, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernelfor spherical
        convolution.

        Parameters
        ----------
        bval : float,
            b-value in s/mm^2.
        lambda_par : float,
            parallel diffusivity in mm^2/s.
        lambda_perp : float,
            perpendicular diffusivity in mm^2/s.
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

        E_zeppelin_sf = self(
            bval, SPHERE_CARTESIAN,
            mu=np.r_[0., 0.], lambda_par=lambda_par, lambda_perp=lambda_perp
        )

        sh_mat = real_sym_sh_mrtrix(
            rh_order, SPHERE_SPHERICAL[1], SPHERE_SPHERICAL[2]
        )[0]
        sh_mat_inv = np.linalg.pinv(sh_mat)
        sh = np.dot(sh_mat_inv, E_zeppelin_sf)
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh

    def spherical_mean(self, bval, **kwargs):
        """ Spherical mean of the signal attenuation of the Zeppelin model
        for a given b-value and parallel and perpendicular diffusivity.
        Analytic expression from Eq. (8) in [1]).

        Parameters
        ----------
        bval : float,
            b-value in s/mm^2.
        lambda_par : float,
            parallel diffusivity in mm^2/s.
        lambda_perp : float,
            perpendicular diffusivity in mm^2/s.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model.

        References
        ----------
        .. [1] Kaden et al. "Multi-compartment microscopic diffusion imaging."
            NeuroImage 139 (2016): 346-359.
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)

        exp_bl = np.exp(-bval * lambda_perp)
        sqrt_bl = np.sqrt(bval * (lambda_par - lambda_perp))
        E_mean = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
        return E_mean


class SD3Watson(MicrostrukturModel):
    r""" The Watson spherical distribution model [1, 2].

    Parameters
    ----------
    n : array of shape(3) or array of shape(N x 3),
        sampled orientations of the Watson distribution.
    mu : array, shape(3),
        unit vector representing orientation of Watson distribution.
    kappa : float,
        concentration parameter of the Watson distribution.

    Returns
    -------
    Wn: float or array of shape(N),
        Probability density at orientations n, given mu and kappa.

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
        'kappa': (0, np.inf),
    }

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
        x, y, z = sphere2cart(1, mu[0], mu[1])
        mu_cart = np.r_[float(x), float(y), float(z)]
        numerator = np.exp(kappa * np.dot(n, mu_cart) ** 2)
        denominator = 4 * np.pi * special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def spherical_harmonics_representation(self, sh_order=14, **kwargs):
        r""" The Watson spherical distribution model in spherical harmonics [1, 2].

        Parameters
        ----------
        mu : array, shape(3),
            unit vector representing orientation of Watson distribution.
        kappa : float,
            concentration parameter of the Watson distribution.
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
        x_, y_, z_ = sphere2cart(1., mu[0], mu[1])

        R = utils.rotation_matrix_001_to_xyz(float(x_), float(y_), float(z_))
        vertices_rotated = np.dot(SPHERE_CARTESIAN, R.T)
        _, theta_rotated, phi_rotated = cart2sphere(vertices_rotated[:, 0],
                                                    vertices_rotated[:, 1],
                                                    vertices_rotated[:, 2])

        watson_sf = self(vertices_rotated, mu=mu, kappa=kappa)
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_rotated, phi_rotated)[0]
        sh_mat_inv = np.linalg.pinv(sh_mat)
        watson_sh = np.dot(sh_mat_inv, watson_sf)
        return watson_sh


class SD2Bingham(MicrostrukturModel):
    r""" The Bingham spherical distribution model [1, 2, 3] using euler angles.

    Parameters
    ----------
    n : array of shape(3) or array of shape(N x 3),
        sampled orientations of the Bingham distribution.
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.

    Returns
    -------
    Bn: float or array of shape(N),
        Probability density at orientations n, given theta, phi, psi, kappa
        and beta.

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
        'kappa': (0, np.inf),
        'beta': (0, np.inf)  # beta<=kappa in fact
    }

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
        denominator = 4 * np.pi * special.hyp1f1(0.5, 1.5, kappa)
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

        x_, y_, z_ = sphere2cart(1., mu[0], mu[1])

        R = utils.rotation_matrix_001_to_xyz(float(x_), float(y_), float(z_))
        vertices_rotated = np.dot(SPHERE_CARTESIAN, R.T)
        _, theta_rotated, phi_rotated = cart2sphere(vertices_rotated[:, 0],
                                                    vertices_rotated[:, 1],
                                                    vertices_rotated[:, 2])

        bingham_sf = self(vertices_rotated, mu=mu, psi=psi, kappa=kappa,
                          beta=beta)

        sh_mat = real_sym_sh_mrtrix(sh_order, theta_rotated, phi_rotated)[0]
        sh_mat_inv = np.linalg.pinv(sh_mat)
        bingham_sh = np.dot(sh_mat_inv, bingham_sf)
        # normalization with spherical mean as there is still the
        # normalization bug
        bingham_sh /= (bingham_sh[0] * (2 * np.sqrt(np.pi)))
        return bingham_sh


class CylindricalModelGradientEcho:
    '''
    Different Gradient Strength Echo protocols
    '''

    def __init__(
        self, Time=None, gradient_strength=None, delta=None,
        gradient_direction=None,
        perpendicular_signal_approximation_model=None,
        cylinder_direction=None,
        radius=None,
        alpha=None,
        beta=None,
        kappa=None,
        radius_max=None,
        radius_integral_steps=35,
        diffusion_constant=CONSTANTS['water_diffusion_constant'],
        gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    ):
        '''
        Everything on S.I.. To get the Gamma parameters in micrometers
        as usually done beta must be multiplied by 10e-6.
        '''
        self.Time = Time
        self.gradient_strength = gradient_strength
        self.gradient_direction = gradient_direction
        self.delta = delta
        self.Delta = Time - 2 * delta
        self.cylinder_direction = cylinder_direction
        self.radius = radius
        self.length = radius
        if radius is not None:
            self.diameter = 2 * radius
        if alpha is not None and beta is not None:
            if radius_max is None:
                gamma_dist = stats.gamma(alpha, scale=beta)
                self.radius_max = gamma_dist.mean() + 6 * gamma_dist.std()
            else:
                self.radius_max = radius_max
        self.kappa = kappa
        self.radius_integral_steps = radius_integral_steps
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.perpendicular_signal_approximation_model = \
            perpendicular_signal_approximation_model
        self.alpha = alpha
        self.beta = beta
        self.default_protocol_vars = list(locals().keys())
        self.default_protocol_vars += ['Delta', 'length', 'diameter']
        self.default_protocol_vars.remove('self')

        if self.cylinder_direction is not None:
            self.cylinder_parallel_tensor = np.outer(
                cylinder_direction,
                cylinder_direction
            )
            self.cylinder_perpendicular_plane = (
                np.eye(3) -
                self.cylinder_parallel_tensor
            )

    def unify_caliber_measures(self, kwargs):
        need_correction = sum((
            k in kwargs and kwargs[k] is not None
            for k in ('radius', 'diameter', 'length')
        ))

        if need_correction == 0:
            return kwargs
        if need_correction > 1:
            raise ValueError

        if 'diameter' in kwargs and kwargs['diameter'] is not None:
            kwargs['radius'] = kwargs['diameter'] / 2
            kwargs['length'] = kwargs['diameter'] / 2
            return kwargs
        if 'radius' in kwargs and kwargs['radius'] is not None:
            kwargs['length'] = kwargs['radius']
            kwargs['diameter'] = 2 * kwargs['radius']
        elif 'length' in kwargs and kwargs['length'] is not None:
            kwargs['radius'] = kwargs['length']
            kwargs['diameter'] = 2 * kwargs['length']
        return kwargs

    def attenuation(self, **kwargs):

        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        kwargs_aux = kwargs.copy()

        gradient_direction = kwargs['gradient_direction']
        gradient_direction = np.atleast_3d(gradient_direction)
        gradient_direction = gradient_direction / np.sqrt(
            (gradient_direction ** 2).sum(1)
        )[:, None, :]

        cylinder_direction = kwargs['cylinder_direction']
        cylinder_direction = np.atleast_3d(cylinder_direction)
        cylinder_direction = cylinder_direction / np.sqrt(
            (cylinder_direction ** 2).sum(1)
        )[:, None, :]

        cylinder_parallel_tensor = np.einsum(
            'ijk,ilk->ijl',
            cylinder_direction, cylinder_direction
        )

        # Matrix of cylinder direction * gradients
        gradient_parallel_norm = np.sqrt((np.einsum(
            'ijk, mkj -> imj',
            cylinder_parallel_tensor, gradient_direction
        ) ** 2).sum(-1))
        gradient_perpendicular_norm = np.sqrt(1 - gradient_parallel_norm ** 2)
        kwargs_aux['gradient_strength'] = (
            kwargs['gradient_strength'] *
            gradient_parallel_norm.squeeze()
        )
        parallel_attenuation = (
            np.atleast_2d(free_diffusion_attenuation(**kwargs_aux))
        )

        kwargs_aux['gradient_strength'] = (
            kwargs['gradient_strength'] *
            gradient_perpendicular_norm.squeeze()
        )
        perpendicular_attenuation = (
            # gradient_perpendicular_norm.T *
            np.atleast_2d(
                kwargs['perpendicular_signal_approximation_model'](
                    **kwargs_aux
                )
            )
        )

        # Return a matrix of gradients * cylinder direction
        return (parallel_attenuation * perpendicular_attenuation).T

    def attenuation_gamma_distributed_radii_(self, **kwargs):
        # this function does not take into account the spins in the cylinders!
        # use the other function below!
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        alpha = kwargs['alpha']
        beta = kwargs['beta']
        radius_max = kwargs['radius_max']
        attenuation = kwargs['attenuation']

        if alpha is None or beta is None or radius_max is None:
            raise ValueError('alpha, beta and radius_max must be provided')
        kwargs.setdefault('N_radii_samples', 50)

        gradient_strength = kwargs['gradient_strength']
        gamma_dist = stats.gamma(alpha, scale=beta)

        # Working in microns for better algorithm resolution
        E = integrate.odeint(
            lambda g, x: (
                gamma_dist.pdf(x * 1e-6) *
                np.abs(attenuation(radius=x * 1e-6))
            ),
            np.zeros_like(gradient_strength), [1e-10, radius_max / 1e-6]
        )[1] * 1e-6

        return E

    def attenuation_gamma_distributed_radii(self, **kwargs):
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        alpha = kwargs['alpha']
        beta = kwargs['beta']
        radius_max = kwargs['radius_max']
        attenuation = kwargs['attenuation']

        if alpha is None or beta is None or radius_max is None:
            raise ValueError('alpha, beta and radius_max must be provided')

        gradient_strength = kwargs['gradient_strength']
        gamma_dist = stats.gamma(alpha, scale=beta)

        E = np.empty(
            (kwargs['radius_integral_steps'], len(gradient_strength)),
            dtype=complex
        )

        radii = np.linspace(1e-50, radius_max, kwargs['radius_integral_steps'])
        area = np.pi * radii ** 2
        radii_pdf = gamma_dist.pdf(radii)
        radii_pdf_area = radii_pdf * area
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapz(x=radii, y=radii_pdf_area)
        )

        radius_old = kwargs['radius']

        del kwargs['radius']
        for i, radius in enumerate(radii):
            E[i] = (
                radii_pdf_normalized[i] *
                attenuation(radius=radius, **kwargs).squeeze()
            )

        kwargs['radius'] = radius_old
        E = np.trapz(E, x=radii, axis=0)
        return E

    def attenuation_watson_distributed_orientation_dblquad(self, **kwargs):
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        kappa = kwargs['kappa']
        normalization_constant = (
            4 * np.pi *
            special.hyp1f1(.5, 1.5, kappa)
        )

        mu = kwargs['cylinder_direction']
        mu /= np.linalg.norm(mu)
        attenuation = kwargs['attenuation']
        gradient_strength = kwargs['gradient_strength']

        def watson_pdf(n):
            return (
                np.exp(kappa * np.dot(mu, n) ** 2) /
                normalization_constant
            )

        kwargs_integrand = {}
        kwargs_integrand.update(kwargs)
        del kwargs_integrand['cylinder_direction']
        del kwargs_integrand['gradient_strength']
        del kwargs_integrand['diameter']
        del kwargs_integrand['length']

        def integrand_real(phi, theta, g):
            vec = np.r_[
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi)
            ]
            pdf = watson_pdf(vec)
            E = attenuation(
                gradient_strength=g,
                cylinder_direction=vec,
                **kwargs_integrand
            )
            return pdf * np.real(E) * np.sin(theta)

        E = np.empty_like(gradient_strength)
        for i, g in enumerate(gradient_strength):
            res = integrate.dblquad(
                integrand_real,
                0, 2 * np.pi,
                lambda x: 0, lambda x: np.pi,
                args=(g,), epsabs=1e-6, epsrel=1e-6
            )
            E[i] = res[0]

        kwargs['cylinder_direction'] = mu

        return E

    def attenuation_watson_distributed_orientation(self, **kwargs):
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        kappa = kwargs['kappa']
        normalization_constant = (
            4 * np.pi *
            special.hyp1f1(.5, 1.5, kappa)
        )

        mu = kwargs['cylinder_direction']
        mu /= np.linalg.norm(mu)
        attenuation = kwargs['attenuation']
        gradient_strength = kwargs['gradient_strength']
        gradient_direction = np.atleast_2d(kwargs['gradient_direction'])

        def watson_pdf(n):
            return (
                np.exp(kappa * np.dot(mu, n.T) ** 2) /
                normalization_constant
            )

        kwargs_integrand = {}
        kwargs_integrand.update(kwargs)
        del kwargs_integrand['cylinder_direction']
        del kwargs_integrand['gradient_strength']
        del kwargs_integrand['diameter']
        del kwargs_integrand['length']

        def integrand_real(vec, g):
            pdf = watson_pdf(vec)
            E = attenuation(
                gradient_strength=g,
                cylinder_direction=vec,
                **kwargs_integrand
            )
            return pdf[:, None] * E

        E = np.zeros(
            (len(gradient_strength), len(gradient_direction)),
            dtype=complex
        )
        for i, g in enumerate(gradient_strength):
            v = SPHERICAL_INTEGRATOR.integrate(
                integrand_real, args=(g,)
            )
            E[i] = v
        kwargs['cylinder_direction'] = mu

        return E


class IsotropicModelGradientEcho:
    '''
    Different Gradient Strength Echo protocols
    '''

    def __init__(
        self, Time=None, gradient_strength=None, delta=None,
        gradient_direction=None,
        signal_approximation_model=None,
        radius=None,
        alpha=None,
        beta=None,
        radius_max=20e-6,
        diffusion_constant=CONSTANTS['water_diffusion_constant'],
        gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    ):
        '''
        Everything on S.I.. To get the Gamma parameters in micrometers
        as usually done beta must be multiplied by 10e-6.
        '''
        self.Time = Time
        self.gradient_strength = gradient_strength
        self.gradient_direction = gradient_direction
        self.delta = delta
        self.Delta = Time - 2 * delta
        self.radius = radius
        self.length = radius
        self.diameter = 2 * radius
        self.radius_max = radius_max
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.signal_approximation_model = \
            signal_approximation_model
        self.alpha = alpha
        self.beta = beta
        self.default_protocol_vars = locals().keys()
        self.default_protocol_vars += ['Delta', 'length', 'diameter']
        self.default_protocol_vars.remove('self')

    def unify_caliber_measures(self, kwargs):
        need_correction = sum((
            'radius' in kwargs,
            'diameter' in kwargs,
            'length' in kwargs
        ))

        if need_correction == 0:
            return kwargs
        if need_correction > 1:
            raise ValueError

        if 'diameter' in kwargs:
            kwargs['radius'] = kwargs['diameter']
            kwargs['length'] = kwargs['diameter']
            return kwargs
        if 'radius' in kwargs:
            kwargs['length'] = kwargs['radius']
        elif 'length' in kwargs:
            kwargs['radius'] = kwargs['length']
        kwargs['diameter'] = 2 * kwargs['radius']
        return kwargs

    def attenuation(self, **kwargs):

        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        return kwargs['perpendicular_signal_approximation_model'](**kwargs)

    def attenuation_gamma_distributed_radii_(self, **kwargs):
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        alpha = kwargs['alpha']
        beta = kwargs['beta']
        radius_max = kwargs['radius_max']
        attenuation = kwargs['attenuation']

        if alpha is None or beta is None or radius_max is None:
            raise ValueError('alpha, beta and radius_max must be provided')
        kwargs.setdefault('N_radii_samples', 50)

        gradient_strength = kwargs['gradient_strength']
        gamma_dist = stats.gamma(alpha, scale=beta)

        # Working in microns for better algorithm resolution
        E = integrate.odeint(
            lambda g, x: (
                gamma_dist.pdf(x * 1e-6) *
                np.abs(attenuation(radius=x * 1e-6))
            ),
            np.zeros_like(gradient_strength), [1e-10, radius_max / 1e-6]
        )[1] * 1e-6

        return E

    def attenuation_gamma_distributed_radii(self, **kwargs):
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        alpha = kwargs['alpha']
        beta = kwargs['beta']
        radius_max = kwargs['radius_max']
        attenuation = kwargs['attenuation']

        if alpha is None or beta is None or radius_max is None:
            raise ValueError('alpha, beta and radius_max must be provided')

        gradient_strength = kwargs['gradient_strength']
        gamma_dist = stats.gamma(alpha, scale=beta)

        E = np.empty(
            (kwargs['radius_integral_steps'], len(gradient_strength)),
            dtype=complex
        )

        radii = np.linspace(1e-50, radius_max, kwargs['radius_integral_steps'])
        radius_old = kwargs['radius']
        del kwargs['radius']
        for i, radius in enumerate(radii):
            E[i] = (
                gamma_dist.pdf(radius) *
                attenuation(radius=radius, **kwargs)
            )

        kwargs['radius'] = radius_old
        E = np.trapz(E, x=radii, axis=0)
        return E

    def attenuation_watson_distributed_orientation(self, **kwargs):
        kwargs = self.unify_caliber_measures(kwargs)
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
        kwargs.setdefault('attenuation', self.attenuation)

        E = kwargs['attenuation'](**kwargs)
        return E
