# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division
import pkg_resources
from collections import OrderedDict
from itertools import chain

import numpy as np
from time import time

from microstruktur.utils.spherical_mean import (
    estimate_spherical_mean_multi_shell)
from scipy.optimize import differential_evolution, brute, minimize
from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy")
pathos, have_pathos, _ = optional_package("pathos")

if have_pathos:
    import pathos.pools as pp
    from pathos.helpers import cpu_count

GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data/gradient_tables'
)
SIGNAL_MODELS_PATH = pkg_resources.resource_filename(
    'microstruktur', 'signal_models'
)


class MicrostructureModel:
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
                        np.tile(value[0], np.r_[parameter_shapes[0], 1]))
                elif card == 1 and not np.all(value.shape == np.r_[1]):
                    parameter_vector.append(value[..., None])
                elif card == 2 and np.all(value.shape == np.r_[2]):
                    parameter_vector.append(
                        np.tile(value, np.r_[parameter_shapes[0], 1])
                    )
                else:
                    parameter_vector.append(parameters[parameter])
            parameter_vector = np.concatenate(parameter_vector, axis=-1)
        return parameter_vector

    def parameters_to_parameter_vector2(self, parameters):
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
                        np.tile(value[0], np.r_[parameter_shapes[0], 1]))
                elif card == 1 and not np.all(value.shape == np.r_[1]):
                    parameter_vector.append(value[..., None])
                elif card == 2 and np.all(value.shape == np.r_[2]):
                    parameter_vector.append(
                        np.tile(value, np.r_[parameter_shapes[0], 1])
                    )
                else:
                    parameter_vector.append(parameters[parameter])
            parameter_vector = np.concatenate(parameter_vector, axis=-1)
        return parameter_vector

    def parameter_initial_guess_to_parameter_vector(self, **parameters):
        set_parameters = {}
        for parameter, card in self.parameter_cardinality.items():
            try:
                set_parameters[parameter] = parameters[parameter]
                msg = str(parameter) + ' successfully set.'
                print (msg)
            except KeyError:
                set_parameters[parameter] = None
        return self.parameters_to_parameter_vector2(set_parameters)

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

    @property
    def opt_params_for_optimization(self):
        params = []
        for parameter, card in self.parameter_cardinality.items():
            optimize_param = self.optimized_parameters[parameter]
            if card == 1:
                params.append(optimize_param)
            else:
                for i in range(card):
                    params.append(optimize_param)
        return params

    @property
    def scales_for_optimization(self):
        return np.hstack([scale for parameter, scale in
                          self.parameter_scales.items()])

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

    def fit(self, data, parameter_initial_guess=None, mask=None,
            solver='scipy', Ns=10, maxiter=300,
            use_parallel_processing=False, number_of_processors=None):
        """ The main data fitting function of a multi-compartment model.

        Once a microstructure model is formed, this function can fit it to an
        N-dimensional dMRI data set, and returns for every voxel the fitted
        model parameters.

        No initial guess needs to be given to fit a model, but can (and should)
        be given to speed up the fitting process. The parameter_initial_guess
        input can be created using parameter_initial_guess_to_parameter_vector.

        A mask can also be given to exclude voxels from fitting (e.g. voxels
        that are outside the brain). If no mask is given then all voxels are
        included.

        An optimization approach can be chosen as either 'scipy' or 'mix'.
        - Choosing scipy will first use a brute-force optimization to find an
          initial guess for parameters without one, and will then refine the
          result using gradient-descent-based optimization.
        - Choosing mix will use the recent MIX algorithm based on separation of
          linear and non-linear parameters. MIX first uses a stochastic
          algorithm to find the non-linear parameters (non-volume fractions),
          then estimates the volume fractions while fixing the estimates of the
          non-linear parameters, and then finally refines the solution using
          a gradient-descent-based algorithm.

        The fitting process can be readily parallelized using the optional
        "pathos" package. To use it set use_parallel_processing=True. The
        algorithm will automatically use all cores in the machine, unless
        otherwise specified in number_of_processors.

        Parameters
        ----------
        data : N-dimensional array of size (N_x, N_y, ..., N_dwis),
            The measured DWI signal attenuation array of either a single voxel
            or an N-dimensional dataset.
        mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
            Optional mask of voxels to be included in the optimization.
        solver : string,
            Selection of optimization algorithm.
            - 'scipy' to use standard brute-to-fine optimization.
            - 'mix' to use Microstructure Imaging of Crossing (MIX)
              optimization.
            - [future] 'amico' to use Accelerated Microstructure Imaging via
               Convex Optimization (AMICO).
        Ns : integer,
            for brute optimization, decised how many steps are sampled for
            every parameter.
        use_parallel_processing : bool,
            whether or not to use parallel processing using pathos.
        number_of_processors : integer,
            number of processors to use for parallel processing. Defaults to
            the number of processors in the computer according to cpu_count().
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

        # estimate S0
        data_ = np.atleast_2d(data)
        S0 = np.mean(data_[..., self.scheme.b0_mask], axis=-1)

        if mask is None:
            mask = S0 > 0
        else:
            mask = np.all([mask, S0 > 0], axis=0)
        mask_pos = np.where(mask)

        N_parameters = len(self.bounds_for_optimization)
        N_voxels = np.sum(mask)

        # make starting parameters and data the same size
        if parameter_initial_guess is None:
            x0_ = np.tile(None,
                          np.r_[data_.shape[:-1], N_parameters])
        else:
            x0_ = homogenize_x0_to_data(
                data_, parameter_initial_guess)
            x0_bool = np.all(
                x0_ == None, axis=tuple(np.arange(x0_.ndim - 1)))
            x0_[..., ~x0_bool] /= self.scales_for_optimization[~x0_bool]

        if use_parallel_processing and not have_pathos:
            msg = 'Cannot use parallel processing without pathos.'
            raise ValueError(msg)
        elif use_parallel_processing and have_pathos:
            fitted_parameters_lin = [None] * N_voxels
            if number_of_processors is None:
                number_of_processors = cpu_count()
                pool = pp.ProcessPool(number_of_processors)
                print ('Using parallel processing with {} workers.').format(
                    cpu_count())
        else:
            fitted_parameters_lin = np.empty(
                np.r_[N_voxels, N_parameters], dtype=float)

        # if the models are spherical mean based then estimate the
        # spherical mean of the data.
        if self.spherical_mean:
            data_to_fit = np.zeros(
                np.r_[data_.shape[:-1],
                      self.scheme.unique_dwi_indices.max() + 1])
            for pos in zip(*mask_pos):
                data_to_fit[pos] = estimate_spherical_mean_multi_shell(
                    data_[pos], self.scheme)
        else:
            data_to_fit = data_

        start = time()
        print ('Starting fitting process')

        for idx, pos in enumerate(zip(*mask_pos)):
            voxel_E = data_to_fit[pos] / S0[pos]
            voxel_x0_vector = x0_[pos]

            if solver == 'scipy':
                fit_func = self.fit_brute2fine
                fit_args = (voxel_E, voxel_x0_vector, Ns)
            elif solver == 'mix':
                fit_func = self.fit_mix
                fit_args = (voxel_E, voxel_x0_vector, maxiter)

            if use_parallel_processing:
                fitted_parameters_lin[idx] = pool.apipe(fit_func, *fit_args)
            else:
                fitted_parameters_lin[idx] = fit_func(*fit_args)
        if use_parallel_processing:
            fitted_parameters_lin = np.array(
                [p.get() for p in fitted_parameters_lin])

        fitting_time = time() - start
        print ('Fitting complete in {} seconds.').format(fitting_time)
        print ('Average of {} seconds per voxel.').format(
            fitting_time / N_voxels)

        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = (
            fitted_parameters_lin * self.scales_for_optimization)

        if data.ndim == 1:
            return np.squeeze(fitted_parameters)
        else:
            return fitted_parameters

    def fit_brute2fine(self, data, x0_vector, Ns):
        """
        If initial x0 is given for a parameter, then the bounds are fixed for
        brute-force optimization.
        If initial x0 is given AND parameter_optimize is False for a parameter,
        then its bounds are also fixed for subsequent gradient-based
        optimization.
        """
        fit_args = (data, self.scheme)
        bounds = self.bounds_for_optimization
        bounds_brute = []
        bounds_fine = list(bounds)
        for i, x0_ in enumerate(x0_vector):
            if x0_ is None:
                bounds_brute.append(
                    slice(bounds[i][0], bounds[i][1],
                          (bounds[i][1] - bounds[i][0]) / float(Ns)))
            if x0_ is not None:
                bounds_brute.append(slice(x0_, x0_ + 1e-2, None))
            if (x0_ is not None and
                    self.opt_params_for_optimization[i] is False):
                bounds_fine[i] = np.r_[x0_, x0_]

        # brute-force optimization returns initial guess for all parameters
        x0_brute = brute(
            self.objective_function, ranges=bounds_brute, args=fit_args,
            finish=None)
        # the intial guess is used to find a local minima using gradient-based
        # optimization.
        x_final = minimize(self.objective_function, x0_brute,
                           args=fit_args, bounds=bounds_fine,
                           method='L-BFGS-B').x
        return x_final

    def objective_function(self, parameter_vector, data, acquisition_scheme):
        parameter_vector = parameter_vector * self.scales_for_optimization
        parameters = {}
        parameters.update(
            self.parameter_vector_to_parameters(parameter_vector)
        )
        E_model = self(acquisition_scheme, **parameters)
        E_diff = E_model - data
        objective = np.sum(E_diff ** 2) / len(data)
        return objective

    def fit_mix(self, data, x0_vector, maxiter=150):
        """
        differential_evolution
        cvxpy
        least squares

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Nature Scientific
               reports 6 (2016).
        """

        bounds = list(self.bounds_for_optimization)
        for i, x0_ in enumerate(x0_vector):
            if (x0_ is not None and
                    self.opt_params_for_optimization[i] is False):
                bounds[i] = np.r_[x0_, x0_]
        # step 1: Variable separation using genetic algorithm

        res_one = differential_evolution(self.stochastic_objective_function,
                                         bounds=bounds,
                                         maxiter=maxiter,
                                         args=(data, self.scheme))
        res_one_x = res_one.x
        parameters = self.parameter_vector_to_parameters(
            res_one_x * self.scales_for_optimization)

        # step 2: Estimating linear variables using cvx
        phi = self(self.scheme,
                   quantity="stochastic cost function", **parameters)
        x_fe = self._cvx_fit_linear_parameters(data, phi)

        # step 3: refine using gradient method / convert nested fractions
        x_fe_nested = np.ones(len(x_fe) - 1)
        x_fe_nested[0] = x_fe[0]
        for i in np.arange(1, len(x_fe_nested)):
            x_fe_nested[i] = x_fe[i] / x_fe[i - 1]

        res_one_x[-len(x_fe_nested):] = x_fe_nested

        res_final = minimize(self.objective_function, res_one_x,
                             (data, self.scheme),
                             bounds=bounds)
        fitted_parameters = res_final.x
        return fitted_parameters

    def stochastic_objective_function(self, parameter_vector,
                                      data, acquisition_scheme):
        parameter_vector = parameter_vector * self.scales_for_optimization
        parameters = {}
        parameters.update(
            self.parameter_vector_to_parameters(parameter_vector)
        )

        phi_x = self(acquisition_scheme,
                     quantity="stochastic cost function", **parameters)

        phi_mp = np.dot(np.linalg.pinv(np.dot(phi_x.T, phi_x)), phi_x.T)
        f = np.dot(phi_mp, data)
        yhat = np.dot(phi_x, f)
        cost = np.dot(data - yhat, data - yhat).squeeze()
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

    def R2_coefficient_of_determination(
            self, parameter_vector, data, acquisition_scheme):
        "Calculates the R-squared of the model fit."
        parameters = self.parameter_vector_to_parameters(
            parameter_vector)
        y_hat = self(acquisition_scheme, **parameters)
        y_bar = np.mean(data)
        SStot = np.sum((data - y_bar) ** 2)
        SSres = np.sum((data - y_hat) ** 2)
        R2 = 1 - SSres / SStot
        return R2


class MultiCompartmentMicrostructureModel(MicrostructureModel):
    r'''
    Class for Partial Volume-Combined Microstrukture Models.
    Given a set of models :math:`m_1...m_N`, and the partial volume ratios
    math:`v_1...v_{N-1}`, the partial volume function is

    .. math::
        v_1 m_1 + (1 - v_1) v_2 m_2 + ... + (1 - v_1)...(1-v_{N-1}) m_N

    Parameters
    ----------
    models : list of N MicrostructureModel instances,
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
        self, acquisition_scheme, models, partial_volumes=None,
        parameter_links=[], optimise_partial_volumes=True
    ):
        self.scheme = acquisition_scheme
        self.models = models
        self.partial_volumes = partial_volumes
        self.parameter_links = parameter_links
        self.optimise_partial_volumes = optimise_partial_volumes

        self._prepare_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._verify_model_input_requirements()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()

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
                self._parameter_ranges[partial_volume_name] = (0.01, .99)
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

    def _prepare_parameters_to_optimize(self):
        self.optimized_parameters = OrderedDict({
            k: True
            for k, v in self.parameter_cardinality.items()
        })

    def _verify_model_input_requirements(self):
        models_spherical_mean = [model.spherical_mean for model in self.models]
        if len(np.unique(models_spherical_mean)) > 1:
            msg = "Cannot mix spherical mean and non-spherical mean models. "
            msg = "Current model selection is {}".format(self.models)
            raise ValueError(msg)
        self.spherical_mean = np.all(models_spherical_mean)

    def _check_for_double_model_class_instances(self):
        if len(self.models) != len(np.unique(self.models)):
            msg = "Each model in the multi-compartment model must be "
            msg += "instantiated separately. For example, to make a model "
            msg += "with two sticks, the models must be given as "
            msg += "models = [stick1, stick2], not as "
            msg += "models = [stick1, stick1]."
            raise ValueError(msg)

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


def homogenize_x0_to_data(data, x0):
    if x0 is not None:
        if x0.ndim == 1:
            # the same x0 will be used for every voxel in N-dimensional data.
            x0_as_data = np.tile(x0, np.r_[data.shape[:-1], 1])
        else:
            x0_as_data = x0.copy()
    if not np.all(
        x0_as_data.shape[:-1] == data.shape[:-1]
    ):
        # if x0 and data are both N-dimensional but have different shapes.
        msg = "data and x0 both N-dimensional but have different shapes. "
        msg += "Current shapes are {} and {}.".format(
            data.shape[:-1],
            x0_as_data.shape[:-1])
        raise ValueError(msg)
    return x0_as_data
