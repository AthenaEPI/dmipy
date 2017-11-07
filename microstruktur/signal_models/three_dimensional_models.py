# -*- coding: utf-8 -*-
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
from scipy import special
from scipy.optimize import minimize
from dipy.reconst.shm import real_sym_sh_mrtrix

from . import utils
from . import CONSTANTS
from ..signal_models.spherical_convolution import kernel_sh_to_rh
from .spherical_mean import estimate_spherical_mean_multi_shell
from ..acquisition_scheme.acquisition_scheme import SimpleAcquisitionSchemeRH
from scipy.interpolate import bisplev
from scipy.optimize import differential_evolution
import cvxpy

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
log_bingham_normalization_splinefit = np.load(
    join(SIGNAL_MODELS_PATH,
         "bingham_normalization_splinefit.npz"))['arr_0']
WATSON_SH_ORDER = 14
DIFFUSIVITY_SCALING = 1e-9
DIAMETER_SCALING = 1e-6
A_SCALING = 1e-12


class MicrostrukturModel:
    @property
    def parameter_ranges(self):
        if not isinstance(self._parameter_ranges, OrderedDict):
            return OrderedDict([
                (k, self._parameter_ranges[k])
                for k in sorted(self._parameter_ranges)
            ])
        else:
            return self._parameter_ranges.copy()

    @property
    def parameter_scales(self):
        if not isinstance(self._parameter_scales, OrderedDict):
            return OrderedDict([
                (k, self._parameter_scales[k])
                for k in sorted(self._parameter_scales)
            ])
        else:
            return self._parameter_scales.copy()

    @property
    def parameter_cardinality(self):
        if hasattr(self, '_parameter_cardinality'):
            return self._parameter_cardinality

        self._parameter_cardinality = OrderedDict([
            (k, len(np.atleast_2d(self.parameter_ranges[k])))
            for k in self.parameter_ranges
        ])
        return self._parameter_cardinality.copy()

    def parameter_vector_to_parameters(self, parameter_vector):
        parameters = {}
        current_pos = 0
        if parameter_vector.ndim == 1:
            for parameter, card in self.parameter_cardinality.items():
                parameters[parameter] = parameter_vector[
                    current_pos: current_pos + card
                ]
                current_pos += card
        else:
            for parameter, card in self.parameter_cardinality.items():
                parameters[parameter] = parameter_vector[
                    ..., current_pos: current_pos + card
                ]
                current_pos += card
        return parameters

    def parameters_to_parameter_vector(self, **parameters):
        parameter_vector = []
        parameter_shapes = []
        for parameter, card in self.parameter_cardinality.items():
            value = np.atleast_1d(parameters[parameter])
            if card == 1 and not np.all(value.shape == np.r_[1]):
                parameter_shapes.append(value.shape)
            if card == 2 and not np.all(value.shape == np.r_[2]):
                parameter_shapes.append(value.shape[:-1])

        if len(np.unique(parameter_shapes)) > 1:
            msg = "parameter shapes are inconsistent."
            raise ValueError(msg)
        elif len(np.unique(parameter_shapes)) == 0:
            for parameter, card in self.parameter_cardinality.items():
                parameter_vector.append(parameters[parameter])
            parameter_vector = np.hstack(parameter_vector)
        elif len(np.unique(parameter_shapes)) == 1:
            for parameter, card in self.parameter_cardinality.items():
                value = np.atleast_1d(parameters[parameter])
                if card == 1 and np.all(value.shape == np.r_[1]):
                    parameter_vector.append(
                        np.tile(value[0], np.r_[parameter_shapes[0], 1])
                    )
                elif card == 2 and np.all(value.shape == np.r_[2]):
                    parameter_vector.append(
                        np.tile(value, np.r_[parameter_shapes[0], 1])
                    )
                else:
                    parameter_vector.append(parameters[parameter])
            parameter_vector = np.concatenate(parameter_vector, axis=-1)
        return parameter_vector

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

    def simulate_signal(self, acquisition_scheme, model_parameters_array):
        """ Function to simulate diffusion data using the defined
        microstructure model and acquisition parameters.

                Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.
        x0 : 1D array of size (N_parameters) or N-dimensional array the same
            size as the data.
            The model parameters of the microstructure model.
            If a 1D array is given, this is the same initial condition for
            every fitted voxel. If a higher-dimenensional array the same size
            as the data is given, then every voxel can possibly be given a
            different initial condition.

        Returns
        -------
        E_simulated: 1D array of size (N_parameters) or N-dimensional
            array the same size as x0.
            The simulated signal of the microstructure model.
        """
        Ndata = acquisition_scheme.number_of_measurements
        x0 = model_parameters_array

        x0_at_least_2d = np.atleast_2d(x0)
        x0_2d = x0_at_least_2d.reshape(-1, x0_at_least_2d.shape[-1])
        E_2d = np.empty(np.r_[x0_2d.shape[:-1], Ndata])
        for i, x0_ in enumerate(x0_2d):
            parameters = self.parameter_vector_to_parameters(x0_)
            E_2d[i] = self(acquisition_scheme, **parameters)
        E_simulated = E_2d.reshape(
            np.r_[x0_at_least_2d.shape[:-1], Ndata])

        if x0.ndim == 1:
            return np.squeeze(E_simulated)
        else:
            return E_simulated

    def fod(self, vertices, model_parameters_array):
        x0 = model_parameters_array
        x0_at_least_2d = np.atleast_2d(x0)
        x0_2d = x0_at_least_2d.reshape(-1, x0_at_least_2d.shape[-1])
        fods_2d = np.empty(np.r_[x0_2d.shape[:-1], len(vertices)])
        for i, x0_ in enumerate(x0_2d):
            parameters = self.parameter_vector_to_parameters(x0_)
            fods_2d[i] = self(vertices, quantity="FOD", **parameters)
        fods = fods_2d.reshape(
            np.r_[x0_at_least_2d.shape[:-1], len(vertices)])

        if x0.ndim == 1:
            return np.squeeze(fods)
        else:
            return fods

    def fit(self, data, acquisition_scheme, model_initial_condition_array):
        """ The data fitting function of a multi-compartment model.

        Parameters
        ----------
        data : N-dimensional array of size (N_x, N_y, ..., N_data),
            The measured DWI signal attenuation array of either a single voxel
            or an N-dimensional dataset.
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.
        x0 : 1D array of size (N_parameters) or N-dimensional array the same
            size as the data.
            The initial condition for the scipy minimize function.
            If a 1D array is given, this is the same initial condition for
            every fitted voxel. If a higher-dimenensional array the same size
            as the data is given, then every voxel can possibly be given a
            different initial condition.

        Returns
        -------
        fitted_parameters: 1D array of size (N_parameters) or N-dimensional
            array the same size as the data.
            The fitted parameters of the microstructure model.
        """
        x0 = model_initial_condition_array
        data_at_least_2d = np.atleast_2d(data)
        x0_at_least_2d = np.atleast_2d(x0)
        if x0.ndim == 1 and data.ndim > 1:
            # the same x0 will be used for every voxel in N-dimensional data.
            x0_at_least_2d = np.tile(x0, np.r_[data.shape[:-1], 1])
            data_at_least_2d = data
        if not np.all(
            x0_at_least_2d.shape[:-1] == data_at_least_2d.shape[:-1]
        ):
            # if x0 and data are both N-dimensional but have different shapes.
            msg = "data and x0 both N-dimensional but have different shapes. "
            msg += "Current shapes are {} and {}.".format(
                data_at_least_2d.shape[:-1],
                x0_at_least_2d.shape[:-1])
            raise ValueError(msg)

        data_2d = data_at_least_2d.reshape(-1, data_at_least_2d.shape[-1])
        x0_2d = x0_at_least_2d.reshape(-1, x0_at_least_2d.shape[-1])
        fitted_parameters = np.empty(x0_2d.shape, dtype=float)

        scaling = np.hstack([scale for parameter, scale in
                             self.parameter_scales.items()])
        x0_2d = x0_2d / scaling
        for idx, (voxel_data, voxel_x0) in enumerate(zip(data_2d, x0_2d)):
            if self.spherical_mean:
                voxel_data_spherical_mean = (
                    estimate_spherical_mean_multi_shell(voxel_data,
                                                        acquisition_scheme))
                res_ = minimize(self.objective_function, voxel_x0,
                                (voxel_data_spherical_mean,
                                 acquisition_scheme),
                                bounds=self.bounds_for_optimization)
            else:
                res_ = minimize(self.objective_function, voxel_x0,
                                (voxel_data, acquisition_scheme),
                                bounds=self.bounds_for_optimization)
            fitted_parameters[idx] = res_.x
        fitted_parameters *= scaling

        if data.ndim == 1:
            return np.squeeze(fitted_parameters.reshape(x0_at_least_2d.shape))
        else:
            return fitted_parameters.reshape(x0_at_least_2d.shape)

    def objective_function(self, parameter_vector, data, acquisition_scheme):
        scaling = np.hstack([scale for parameter, scale in
                             self.parameter_scales.items()])
        parameter_vector = parameter_vector * scaling
        parameters = {}
        parameters.update(
            self.parameter_vector_to_parameters(parameter_vector)
        )
        E_model = self(acquisition_scheme, **parameters)
        E_diff = E_model - data
        objective = np.sum(E_diff ** 2) / len(data)
        return objective

    def fit_brute(self, data, acquisition_scheme, ranges):
        # precompute the data simulations for the given ranges
        # for every voxel estimate the SSD
        # select and return the parmeter combination for the lowest value.
        return None

    def fit_mix(self, data, acquisition_scheme, maxiter=90):
        """
        differential_evolution
        cvxpy
        least squares

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        """
        scaling = np.hstack([scale for parameter, scale in
                             self.parameter_scales.items()])

        res_one = differential_evolution(self.objective_function,
                                         self.bounds_for_optimization,
                                         maxiter=maxiter,
                                         args=(data, acquisition_scheme))

        parameters = self.parameter_vector_to_parameters(
            res_one.x * scaling)
        phi = self(acquisition_scheme,
                   quantity="stochastic cost function", **parameters)
        x_fe = self._cvx_fit_linear_parameters(self, data, phi)
        # and now putting x_fe and x together again in parameters...
        return None

    def stochastic_objective_function(self, parameter_vector,
                                      data, acquisition_scheme):
        scaling = np.hstack([scale for parameter, scale in
                             self.parameter_scales.items()])
        parameter_vector = parameter_vector * scaling
        parameters = {}
        parameters.update(
            self.parameter_vector_to_parameters(parameter_vector)
        )

        phi_x = self(acquisition_scheme,
                     quantity="stochastic cost function", **parameters)

        phi_mp = np.dot(np.linalg.pinv(np.dot(phi_x.T, phi_x)), phi_x.T)
        f = np.dot(phi_mp, data)
        yhat = np.dot(phi_x, f)
        cost = np.dot(data - yhat, data - yhat)
        return cost

    def _cvx_fit_linear_parameters(self, data, phi):
        fe = cvxpy.Variable(phi.shape[1])
        constraints = [cvxpy.sum_entries(fe) == 1,
                       fe >= 0.011,
                       fe <= 0.89]
        obj = cvxpy.Minimize(cvxpy.sum_squares(phi * fe - data))
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()
        return np.array(fe.value).squeeze()

class PartialVolumeCombinedMicrostrukturModel(MicrostrukturModel):
    r'''
    Class for Partial Volume-Combined Microstrukture Models.
    Given a set of models :math:`m_1...m_N`, and the partial volume ratios
    math:`v_1...v_{N-1}`, the partial volume function is

    .. math::
        v_1 m_1 + (1 - v_1) v_2 m_2 + ... + (1 - v_1)...(1-v_{N-1}) m_N

    Parameters
    ----------
    models : list of N MicrostrukturModel instances,
        the models to mix.
    partial_volumes : array, shape(N - 1),
        partial volume factors.
    parameter_links : list of iterables (model, parameter name, link function,
        argument list),
        where model is a Microstruktur model, parameter name is a string
        with the name of the parameter in that model that will be linked,
        link function is a function returning the value of that parameter,
        and argument list is a list of tuples (model, parameter name) where
        those the parameters of those models will be used as arguments for the
        link function. If the model is left as None, then the parameter comes
        from the container partial volume mixing class.

    '''

    def __init__(
        self, models, partial_volumes=None,
        parameter_links=[], optimise_partial_volumes=True
    ):
        self.models = models
        self.partial_volumes = partial_volumes
        self.parameter_links = parameter_links
        self.optimise_partial_volumes = optimise_partial_volumes

        self._prepare_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._verify_model_input_requirements()

    def _prepare_parameters(self):
        self.model_names = []
        model_counts = {}

        for model in self.models:
            if model.__class__ not in model_counts:
                model_counts[model.__class__] = 1
            else:
                model_counts[model.__class__] += 1

            self.model_names.append(
                '{}_{:d}_'.format(
                    model.__class__.__name__,
                    model_counts[model.__class__]
                )
            )

        self._parameter_ranges = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_ranges.items()
        })

        self._parameter_scales = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_scales.items()
        })

        self._parameter_map = {
            model_name + k: (model, k)
            for model, model_name in zip(self.models, self.model_names)
            for k in model.parameter_ranges
        }

        self.parameter_defaults = OrderedDict()
        for model_name, model in zip(self.model_names, self.models):
            for parameter in model.parameter_ranges:
                self.parameter_defaults[model_name + parameter] = getattr(
                    model, parameter
                )

        self._inverted_parameter_map = {
            v: k for k, v in self._parameter_map.items()
        }
        self._parameter_cardinality = self.parameter_cardinality

    def _prepare_partial_volumes(self):
        if self.optimise_partial_volumes:
            self.partial_volume_names = [
                'partial_volume_{:d}'.format(i)
                for i in range(len(self.models) - 1)
            ]

            for i, partial_volume_name in enumerate(self.partial_volume_names):
                self._parameter_ranges[partial_volume_name] = (0, 1)
                self._parameter_scales[partial_volume_name] = 1.
                self.parameter_defaults[partial_volume_name] = (
                    1 / (len(self.models) - i)
                )
                self._parameter_map[partial_volume_name] = (
                    None, partial_volume_name
                )
                self._inverted_parameter_map[(None, partial_volume_name)] = \
                    partial_volume_name
                self._parameter_cardinality[partial_volume_name] = 1
        else:
            if self.partial_volumes is None:
                self.partial_volumes = np.array([
                    1 / (len(self.models) - i)
                    for i in range(len(self.models) - 1)
                ])

    def _prepare_parameter_links(self):
        for i, parameter_function in enumerate(self.parameter_links):
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
            del self._parameter_scales[parameter_name]

    def _verify_model_input_requirements(self):
        models_spherical_mean = [model.spherical_mean for model in self.models]
        if len(np.unique(models_spherical_mean)) > 1:
            msg = "Cannot mix spherical mean and non-spherical mean models. "
            msg = "Current model selection is {}".format(self.models)
            raise ValueError(msg)
        self.spherical_mean = np.all(models_spherical_mean)

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

            if len(arguments) > 0:
                argument_values = []
                for argument in arguments:
                    argument_name = self._inverted_parameter_map[argument]
                    argument_values.append(parameters.get(
                        argument_name,
                        self.parameter_defaults[argument_name]
                    ))

                parameters[parameter_name] = parameter_function(
                    *argument_values
                )
            else:
                parameters[parameter_name] = parameter_function()
        return parameters

    def __call__(self, acquisition_scheme_or_vertices,
                 quantity="signal", **kwargs):
        if quantity == "signal" or quantity == "FOD":
            values = 0
        elif quantity == "stochastic cost function":
            values = np.empty((
                acquisition_scheme_or_vertices.number_of_measurements,
                len(self.models)
            ))
            counter = 0

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
            print model_name
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
                current_partial_volume = current_partial_volume * partial_volume
                accumulated_partial_volume *= (1 - partial_volume)

            if quantity == "signal":
                values = (
                    values +
                    current_partial_volume * model(
                        acquisition_scheme_or_vertices, **parameters)
                )
            elif quantity == "FOD":
                if callable(model.fod):
                    values = (
                        values +
                        current_partial_volume * model.fod(
                            acquisition_scheme_or_vertices, **parameters)
                    )
            elif quantity == "stochastic cost function":
                values[:, counter] = model(acquisition_scheme_or_vertices,
                                           **parameters)
                counter += 1
        return values


class I1Stick(MicrostrukturModel):
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
        'lambda_par': (0, 3)
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
        mu = utils.sphere2cart(np.r_[1, mu])
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
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh


class I2CylinderSodermanApproximation(MicrostrukturModel):
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
    .. [1]_ Söderman, Olle, and Bengt Jönsson. "Restricted diffusion in
            cylindrical geometry." Journal of Magnetic Resonance, Series A
            117.1 (1995): 94-97.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, 3),
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
        mu = utils.sphere2cart(np.r_[1, mu])
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
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh


class I3CylinderCallaghanApproximation(MicrostrukturModel):
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
        'lambda_par': (0, 3),
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
        mu = utils.sphere2cart(np.r_[1, mu])
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
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh


class I4CylinderGaussianPhaseApproximation(MicrostrukturModel):
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
        'lambda_par': (0, 3),
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
        mu = utils.sphere2cart(np.r_[1, mu])
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
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh


class I1StickSphericalMean(MicrostrukturModel):
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
        'lambda_par': (0, 3)
    }
    _parameter_scales = {
        'lambda_par': DIFFUSIVITY_SCALING,
    }
    spherical_mean = True

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

    def derivative(self, bvals, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        blambda = bvals * lambda_par
        der = np.exp(-blambda) / (2 * lambda_par) *\
            (bvals * np.sqrt(np.pi) * erf(np.sqrt(blambda))) /\
            (4 * blambda ** (3 / 2.))
        return der


class E4ZeppelinSphericalMean(MicrostrukturModel):
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
        'lambda_par': (0, 3),
        'lambda_perp': (0, 3)
    }
    _parameter_scales = {
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING
    }
    spherical_mean = True

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

    def derivative(self, bvals, n, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        bllext = bvals * (lambda_par - lambda_perp)

        der_lambda_perp = bvals * np.exp(-bvals * lambda_par) *\
            (-2 * np.sqrt(bllext) - np.exp(bllext) * np.sqrt(np.pi) *
             (-1 + 2 * bllext) * erf(np.sqrt(bllext))) /\
            (4 * bllext ** (3 / 2.))

        der_lambda_par = bvals * np.exp(-bvals * lambda_par) *\
            (2 * np.sqrt(bllext) - np.exp(bllext) * np.sqrt(np.pi) *
             erf(bllext)) / (4 * bllext ** (3 / 2.))
        return der_lambda_par, der_lambda_perp


class E5RestrictedZeppelinSphericalMean(MicrostrukturModel):
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
        'lambda_par': (0, 3),
        'lambda_inf': (0, 3),
        'A': (0, 10)
    }
    _parameter_scales = {
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_inf': DIFFUSIVITY_SCALING,
        'A': A_SCALING
    }
    spherical_mean = True

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


class E2Dot(MicrostrukturModel):
    r""" The Dot model [1] - an non-diffusing compartment.

    Parameters
    ----------
    no parameters

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """

    _parameter_ranges = {
    }
    _parameter_scales = {
    }
    spherical_mean = False

    def __init__(self, dummy=None):
        self.dummy = dummy

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

        E_dot = np.ones(acquisition_scheme.number_of_measurements)
        return E_dot


class E3Ball(MicrostrukturModel):
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
        'lambda_iso': (0, 3)
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


class E4Zeppelin(MicrostrukturModel):
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
        'lambda_par': (0, 3),
        'lambda_perp': (0, 3)
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

        R1 = utils.sphere2cart(np.r_[1, mu])
        R2 = utils.perpendicular_vector(R1)
        R3 = np.cross(R1, R2)

        E_zeppelin = np.exp(-bvals * (lambda_par * np.dot(n, R1) ** 2 +
                                      lambda_perp * np.dot(n, R2) ** 2 +
                                      lambda_perp * np.dot(n, R3) ** 2))
        return E_zeppelin

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
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh


class E5RestrictedZeppelin(MicrostrukturModel):
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
        'lambda_par': (0, 3),
        'lambda_inf': (0, 3),
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
        mu = utils.sphere2cart(np.r_[1, mu])

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
            rotational harmonics of the model aligned with z-axis.
        """
        simple_acq_scheme_rh = SimpleAcquisitionSchemeRH(
            bvalue, SPHERE_CARTESIAN, delta=delta, Delta=Delta)
        E_kernel_sf = self(simple_acq_scheme_rh, mu=np.r_[0., 0.])
        sh = np.dot(inverse_rh_matrix_kernel[rh_order], E_kernel_sf)
        rh = kernel_sh_to_rh(sh, rh_order)
        return rh


class SD3Watson(MicrostrukturModel):
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
        mu_cart = utils.sphere2cart(np.r_[1., mu])
        numerator = np.exp(kappa * np.dot(n, mu_cart) ** 2)
        denominator = 4 * np.pi * special.hyp1f1(0.5, 1.5, kappa)
        Wn = numerator / denominator
        return Wn

    def spherical_harmonics_representation(self, sh_order=14, **kwargs):
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
        x_, y_, z_ = utils.sphere2cart(np.r_[1., mu])

        R = utils.rotation_matrix_001_to_xyz(x_, y_, z_)
        vertices_rotated = np.dot(SPHERE_CARTESIAN, R.T)
        _, theta_rotated, phi_rotated = utils.cart2sphere(vertices_rotated).T

        watson_sf = self(vertices_rotated, mu=mu, kappa=kappa)
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_rotated, phi_rotated)[0]
        sh_mat_inv = np.linalg.pinv(sh_mat)
        watson_sh = np.dot(sh_mat_inv, watson_sf)
        return watson_sh


class SD2Bingham(MicrostrukturModel):
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

        x_, y_, z_ = utils.sphere2cart(np.r_[1., mu])

        R = utils.rotation_matrix_001_to_xyz(x_, y_, z_)
        vertices_rotated = np.dot(SPHERE_CARTESIAN, R.T)
        _, theta_rotated, phi_rotated = utils.cart2sphere(vertices_rotated).T

        bingham_sf = self(vertices_rotated, mu=mu, psi=psi, kappa=kappa,
                          beta=beta)

        sh_mat = real_sym_sh_mrtrix(sh_order, theta_rotated, phi_rotated)[0]
        sh_mat_inv = np.linalg.pinv(sh_mat)
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


class DD1GammaDistribution(MicrostrukturModel):
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


# class CylindricalModelGradientEcho:
#     '''
#     Different Gradient Strength Echo protocols
#     '''

#     def __init__(
#         self, Time=None, gradient_strength=None, delta=None,
#         gradient_direction=None,
#         perpendicular_signal_approximation_model=None,
#         cylinder_direction=None,
#         radius=None,
#         alpha=None,
#         beta=None,
#         kappa=None,
#         radius_max=None,
#         radius_integral_steps=35,
#         diffusion_constant=CONSTANTS['water_diffusion_constant'],
#         gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
#     ):
#         '''
#         Everything on S.I.. To get the Gamma parameters in micrometers
#         as usually done beta must be multiplied by 10e-6.
#         '''
#         self.Time = Time
#         self.gradient_strength = gradient_strength
#         self.gradient_direction = gradient_direction
#         self.delta = delta
#         self.Delta = Time - 2 * delta
#         self.cylinder_direction = cylinder_direction
#         self.radius = radius
#         self.length = radius
#         if radius is not None:
#             self.diameter = 2 * radius
#         if alpha is not None and beta is not None:
#             if radius_max is None:
#                 gamma_dist = stats.gamma(alpha, scale=beta)
#                 self.radius_max = gamma_dist.mean() + 6 * gamma_dist.std()
#             else:
#                 self.radius_max = radius_max
#         self.kappa = kappa
#         self.radius_integral_steps = radius_integral_steps
#         self.diffusion_constant = diffusion_constant
#         self.gyromagnetic_ratio = gyromagnetic_ratio
#         self.perpendicular_signal_approximation_model = \
#             perpendicular_signal_approximation_model
#         self.alpha = alpha
#         self.beta = beta
#         self.default_protocol_vars = list(locals().keys())
#         self.default_protocol_vars += ['Delta', 'length', 'diameter']
#         self.default_protocol_vars.remove('self')

#         if self.cylinder_direction is not None:
#             self.cylinder_parallel_tensor = np.outer(
#                 cylinder_direction,
#                 cylinder_direction
#             )
#             self.cylinder_perpendicular_plane = (
#                 np.eye(3) -
#                 self.cylinder_parallel_tensor
#             )

#     def unify_caliber_measures(self, kwargs):
#         need_correction = sum((
#             k in kwargs and kwargs[k] is not None
#             for k in ('radius', 'diameter', 'length')
#         ))

#         if need_correction == 0:
#             return kwargs
#         if need_correction > 1:
#             raise ValueError

#         if 'diameter' in kwargs and kwargs['diameter'] is not None:
#             kwargs['radius'] = kwargs['diameter'] / 2
#             kwargs['length'] = kwargs['diameter'] / 2
#             return kwargs
#         if 'radius' in kwargs and kwargs['radius'] is not None:
#             kwargs['length'] = kwargs['radius']
#             kwargs['diameter'] = 2 * kwargs['radius']
#         elif 'length' in kwargs and kwargs['length'] is not None:
#             kwargs['radius'] = kwargs['length']
#             kwargs['diameter'] = 2 * kwargs['length']
#         return kwargs

#     def attenuation(self, **kwargs):

#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))

#         kwargs_aux = kwargs.copy()

#         gradient_direction = kwargs['gradient_direction']
#         gradient_direction = np.atleast_3d(gradient_direction)
#         gradient_direction = gradient_direction / np.sqrt(
#             (gradient_direction ** 2).sum(1)
#         )[:, None, :]

#         cylinder_direction = kwargs['cylinder_direction']
#         cylinder_direction = np.atleast_3d(cylinder_direction)
#         cylinder_direction = cylinder_direction / np.sqrt(
#             (cylinder_direction ** 2).sum(1)
#         )[:, None, :]

#         cylinder_parallel_tensor = np.einsum(
#             'ijk,ilk->ijl',
#             cylinder_direction, cylinder_direction
#         )

#         # Matrix of cylinder direction * gradients
#         gradient_parallel_norm = np.sqrt((np.einsum(
#             'ijk, mkj -> imj',
#             cylinder_parallel_tensor, gradient_direction
#         ) ** 2).sum(-1))
#         gradient_perpendicular_norm = np.sqrt(1 - gradient_parallel_norm ** 2)
#         kwargs_aux['gradient_strength'] = (
#             kwargs['gradient_strength'] *
#             gradient_parallel_norm.squeeze()
#         )
#         parallel_attenuation = (
#             np.atleast_2d(free_diffusion_attenuation(**kwargs_aux))
#         )

#         kwargs_aux['gradient_strength'] = (
#             kwargs['gradient_strength'] *
#             gradient_perpendicular_norm.squeeze()
#         )
#         perpendicular_attenuation = (
#             # gradient_perpendicular_norm.T *
#             np.atleast_2d(
#                 kwargs['perpendicular_signal_approximation_model'](
#                     **kwargs_aux
#                 )
#             )
#         )

#         # Return a matrix of gradients * cylinder direction
#         return (parallel_attenuation * perpendicular_attenuation).T

#     def attenuation_gamma_distributed_radii_(self, **kwargs):
#         # this function does not take into account the spins in the cylinders!
#         # use the other function below!
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         alpha = kwargs['alpha']
#         beta = kwargs['beta']
#         radius_max = kwargs['radius_max']
#         attenuation = kwargs['attenuation']

#         if alpha is None or beta is None or radius_max is None:
#             raise ValueError('alpha, beta and radius_max must be provided')
#         kwargs.setdefault('N_radii_samples', 50)

#         gradient_strength = kwargs['gradient_strength']
#         gamma_dist = stats.gamma(alpha, scale=beta)

#         # Working in microns for better algorithm resolution
#         E = integrate.odeint(
#             lambda g, x: (
#                 gamma_dist.pdf(x * 1e-6) *
#                 np.abs(attenuation(radius=x * 1e-6))
#             ),
#             np.zeros_like(gradient_strength), [1e-10, radius_max / 1e-6]
#         )[1] * 1e-6

#         return E

#     def attenuation_gamma_distributed_radii(self, **kwargs):
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         alpha = kwargs['alpha']
#         beta = kwargs['beta']
#         radius_max = kwargs['radius_max']
#         attenuation = kwargs['attenuation']

#         if alpha is None or beta is None or radius_max is None:
#             raise ValueError('alpha, beta and radius_max must be provided')

#         gradient_strength = kwargs['gradient_strength']
#         gamma_dist = stats.gamma(alpha, scale=beta)

#         E = np.empty(
#             (kwargs['radius_integral_steps'], len(gradient_strength)),
#             dtype=complex
#         )

#         radii = np.linspace(1e-50, radius_max, kwargs['radius_integral_steps'])
#         area = np.pi * radii ** 2
#         radii_pdf = gamma_dist.pdf(radii)
#         radii_pdf_area = radii_pdf * area
#         radii_pdf_normalized = (
#             radii_pdf_area /
#             np.trapz(x=radii, y=radii_pdf_area)
#         )

#         radius_old = kwargs['radius']

#         del kwargs['radius']
#         for i, radius in enumerate(radii):
#             E[i] = (
#                 radii_pdf_normalized[i] *
#                 attenuation(radius=radius, **kwargs).squeeze()
#             )

#         kwargs['radius'] = radius_old
#         E = np.trapz(E, x=radii, axis=0)
#         return E

#     def attenuation_watson_distributed_orientation_dblquad(self, **kwargs):
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         kappa = kwargs['kappa']
#         normalization_constant = (
#             4 * np.pi *
#             special.hyp1f1(.5, 1.5, kappa)
#         )

#         mu = kwargs['cylinder_direction']
#         mu /= np.linalg.norm(mu)
#         attenuation = kwargs['attenuation']
#         gradient_strength = kwargs['gradient_strength']

#         def watson_pdf(n):
#             return (
#                 np.exp(kappa * np.dot(mu, n) ** 2) /
#                 normalization_constant
#             )

#         kwargs_integrand = {}
#         kwargs_integrand.update(kwargs)
#         del kwargs_integrand['cylinder_direction']
#         del kwargs_integrand['gradient_strength']
#         del kwargs_integrand['diameter']
#         del kwargs_integrand['length']

#         def integrand_real(phi, theta, g):
#             vec = np.r_[
#                 np.cos(theta) * np.sin(phi),
#                 np.sin(theta) * np.sin(phi),
#                 np.cos(phi)
#             ]
#             pdf = watson_pdf(vec)
#             E = attenuation(
#                 gradient_strength=g,
#                 cylinder_direction=vec,
#                 **kwargs_integrand
#             )
#             return pdf * np.real(E) * np.sin(theta)

#         E = np.empty_like(gradient_strength)
#         for i, g in enumerate(gradient_strength):
#             res = integrate.dblquad(
#                 integrand_real,
#                 0, 2 * np.pi,
#                 lambda x: 0, lambda x: np.pi,
#                 args=(g,), epsabs=1e-6, epsrel=1e-6
#             )
#             E[i] = res[0]

#         kwargs['cylinder_direction'] = mu

#         return E

#     def attenuation_watson_distributed_orientation(self, **kwargs):
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         kappa = kwargs['kappa']
#         normalization_constant = (
#             4 * np.pi *
#             special.hyp1f1(.5, 1.5, kappa)
#         )

#         mu = kwargs['cylinder_direction']
#         mu /= np.linalg.norm(mu)
#         attenuation = kwargs['attenuation']
#         gradient_strength = kwargs['gradient_strength']
#         gradient_direction = np.atleast_2d(kwargs['gradient_direction'])

#         def watson_pdf(n):
#             return (
#                 np.exp(kappa * np.dot(mu, n.T) ** 2) /
#                 normalization_constant
#             )

#         kwargs_integrand = {}
#         kwargs_integrand.update(kwargs)
#         del kwargs_integrand['cylinder_direction']
#         del kwargs_integrand['gradient_strength']
#         del kwargs_integrand['diameter']
#         del kwargs_integrand['length']

#         def integrand_real(vec, g):
#             pdf = watson_pdf(vec)
#             E = attenuation(
#                 gradient_strength=g,
#                 cylinder_direction=vec,
#                 **kwargs_integrand
#             )
#             return pdf[:, None] * E

#         E = np.zeros(
#             (len(gradient_strength), len(gradient_direction)),
#             dtype=complex
#         )
#         for i, g in enumerate(gradient_strength):
#             v = SPHERICAL_INTEGRATOR.integrate(
#                 integrand_real, args=(g,)
#             )
#             E[i] = v
#         kwargs['cylinder_direction'] = mu

#         return E


# class IsotropicModelGradientEcho:
#     '''
#     Different Gradient Strength Echo protocols
#     '''

#     def __init__(
#         self, Time=None, gradient_strength=None, delta=None,
#         gradient_direction=None,
#         signal_approximation_model=None,
#         radius=None,
#         alpha=None,
#         beta=None,
#         radius_max=20e-6,
#         diffusion_constant=CONSTANTS['water_diffusion_constant'],
#         gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
#     ):
#         '''
#         Everything on S.I.. To get the Gamma parameters in micrometers
#         as usually done beta must be multiplied by 10e-6.
#         '''
#         self.Time = Time
#         self.gradient_strength = gradient_strength
#         self.gradient_direction = gradient_direction
#         self.delta = delta
#         self.Delta = Time - 2 * delta
#         self.radius = radius
#         self.length = radius
#         self.diameter = 2 * radius
#         self.radius_max = radius_max
#         self.diffusion_constant = diffusion_constant
#         self.gyromagnetic_ratio = gyromagnetic_ratio
#         self.signal_approximation_model = \
#             signal_approximation_model
#         self.alpha = alpha
#         self.beta = beta
#         self.default_protocol_vars = locals().keys()
#         self.default_protocol_vars += ['Delta', 'length', 'diameter']
#         self.default_protocol_vars.remove('self')

#     def unify_caliber_measures(self, kwargs):
#         need_correction = sum((
#             'radius' in kwargs,
#             'diameter' in kwargs,
#             'length' in kwargs
#         ))

#         if need_correction == 0:
#             return kwargs
#         if need_correction > 1:
#             raise ValueError

#         if 'diameter' in kwargs:
#             kwargs['radius'] = kwargs['diameter']
#             kwargs['length'] = kwargs['diameter']
#             return kwargs
#         if 'radius' in kwargs:
#             kwargs['length'] = kwargs['radius']
#         elif 'length' in kwargs:
#             kwargs['radius'] = kwargs['length']
#         kwargs['diameter'] = 2 * kwargs['radius']
#         return kwargs

#     def attenuation(self, **kwargs):

#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))

#         return kwargs['perpendicular_signal_approximation_model'](**kwargs)

#     def attenuation_gamma_distributed_radii_(self, **kwargs):
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         alpha = kwargs['alpha']
#         beta = kwargs['beta']
#         radius_max = kwargs['radius_max']
#         attenuation = kwargs['attenuation']

#         if alpha is None or beta is None or radius_max is None:
#             raise ValueError('alpha, beta and radius_max must be provided')
#         kwargs.setdefault('N_radii_samples', 50)

#         gradient_strength = kwargs['gradient_strength']
#         gamma_dist = stats.gamma(alpha, scale=beta)

#         # Working in microns for better algorithm resolution
#         E = integrate.odeint(
#             lambda g, x: (
#                 gamma_dist.pdf(x * 1e-6) *
#                 np.abs(attenuation(radius=x * 1e-6))
#             ),
#             np.zeros_like(gradient_strength), [1e-10, radius_max / 1e-6]
#         )[1] * 1e-6

#         return E

#     def attenuation_gamma_distributed_radii(self, **kwargs):
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         alpha = kwargs['alpha']
#         beta = kwargs['beta']
#         radius_max = kwargs['radius_max']
#         attenuation = kwargs['attenuation']

#         if alpha is None or beta is None or radius_max is None:
#             raise ValueError('alpha, beta and radius_max must be provided')

#         gradient_strength = kwargs['gradient_strength']
#         gamma_dist = stats.gamma(alpha, scale=beta)

#         E = np.empty(
#             (kwargs['radius_integral_steps'], len(gradient_strength)),
#             dtype=complex
#         )

#         radii = np.linspace(1e-50, radius_max, kwargs['radius_integral_steps'])
#         radius_old = kwargs['radius']
#         del kwargs['radius']
#         for i, radius in enumerate(radii):
#             E[i] = (
#                 gamma_dist.pdf(radius) *
#                 attenuation(radius=radius, **kwargs)
#             )

#         kwargs['radius'] = radius_old
#         E = np.trapz(E, x=radii, axis=0)
#         return E

#     def attenuation_watson_distributed_orientation(self, **kwargs):
#         kwargs = self.unify_caliber_measures(kwargs)
#         for k in self.default_protocol_vars:
#             kwargs.setdefault(k, getattr(self, k, None))
#         kwargs.setdefault('attenuation', self.attenuation)

#         E = kwargs['attenuation'](**kwargs)
#         return E
