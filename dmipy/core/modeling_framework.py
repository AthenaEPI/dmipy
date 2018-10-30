# -*- coding: utf-8 -*-
'''
Document Module
'''
from __future__ import division
import pkg_resources
from collections import OrderedDict

import numpy as np
from time import time

from ..utils.spherical_mean import (
    estimate_spherical_mean_multi_shell)
from ..utils.utils import (
    T1_tortuosity,
    parameter_equality,
    fractional_parameter)
from .fitted_modeling_framework import (
    FittedMultiCompartmentModel,
    FittedMultiCompartmentSphericalMeanModel,
    FittedMultiCompartmentSphericalHarmonicsModel)
from ..optimizers_fod.construct_observation_matrix import (
    construct_model_based_A_matrix)
from ..optimizers.brute2fine import (
    GlobalBruteOptimizer, Brute2FineOptimizer)
from ..optimizers_fod.csd_tournier import CsdTournierOptimizer
from ..optimizers_fod.csd_cvxpy import CsdCvxpyOptimizer
from ..optimizers.mix import MixOptimizer
from dipy.utils.optpkg import optional_package
from graphviz import Digraph
from uuid import uuid4
pathos, have_pathos, _ = optional_package("pathos")
numba, have_numba, _ = optional_package("numba")


if have_pathos:
    import pathos.pools as pp
    from pathos.helpers import cpu_count

GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'dmipy', 'data/gradient_tables'
)
SIGNAL_MODELS_PATH = pkg_resources.resource_filename(
    'dmipy', 'signal_models'
)

__all__ = [
    'ModelProperties',
    'MultiCompartmentModelProperties',
    'MultiCompartmentModel',
    'MultiCompartmentSphericalMeanModel',
    'MultiCompartmentSphericalHarmonicsModel',
    'homogenize_x0_to_data',
    'ReturnFixedValue'
]


class ModelProperties:
    "Contains various properties for CompartmentModels."

    S0_response = 1.

    @property
    def parameter_ranges(self):
        """Returns the optimization ranges of the model parameters.
        These ranges are given in O(1) scale so optimization algorithms
        don't suffer from large scale differences in optimization parameters.
        """
        return OrderedDict(self._parameter_ranges.copy())

    @property
    def parameter_scales(self):
        """Returns the optimization scales for the model parameters.
        The scales scale the parameter_ranges to their actual size inside
        optimization algorithms.
        """
        return OrderedDict(self._parameter_scales.copy())

    @property
    def parameter_types(self):
        """Returns the optimization scales for the model parameters.
        The scales scale the parameter_ranges to their actual size inside
        optimization algorithms.
        """
        return OrderedDict(self._parameter_types.copy())

    @property
    def parameter_names(self):
        "Returns the names of model parameters."
        return self._parameter_ranges.keys()

    @property
    def parameter_cardinality(self):
        "Returns the cardinality of model parameters"
        return OrderedDict([
            (k, len(np.atleast_2d(self.parameter_ranges[k])))
            for k in self.parameter_ranges
        ])


class MultiCompartmentModelProperties:
    "Class that contains various properties of MultiCompartmentModel instance."

    @property
    def parameter_names(self):
        "Returns the names of model parameters."
        return list(self.parameter_ranges.keys())

    def parameter_vector_to_parameters(self, parameter_vector):
        """Returns the model parameters in dictionary format according to their
        parameter_names. Takes parameter_vector as input, which is the same as
        the output of a FittedMultiCompartmentModel.fitted_parameter_vector.

        Parameters
        ----------
        parameter_vector: array of size (Ndata_x, Ndata_y, ..., Nparameters),
            array that contains the linearized model parameters for an ND-array
            of data voxels.

        Returns
        -------
        parameter: dictionary with parameter_names as parameter keys,
            contains the model parameters in dictionary format.
        """
        parameters = {}
        current_pos = 0
        if parameter_vector.ndim == 1:
            for parameter, card in self.parameter_cardinality.items():
                parameters[parameter] = parameter_vector[
                    current_pos: current_pos + card
                ]
                if card == 1:
                    parameters[parameter] = parameters[parameter][0]
                current_pos += card
        else:
            for parameter, card in self.parameter_cardinality.items():
                parameters[parameter] = parameter_vector[
                    ..., current_pos: current_pos + card
                ]
                if card == 1:
                    parameters[parameter] = parameters[parameter][..., 0]
                current_pos += card
        return parameters

    def parameters_to_parameter_vector(self, **parameters):
        """Returns the model parameters in array format. The input is a
        parameters dictionary that has parameter_names as keys. This is also
        the output of a FittedMultiCompartmentModel.fitted_parameters.

        It's possible to give an array of values for one parameter and only a
        float for others. The function will automatically assume that that the
        float parameters are constant in the data set and broadcast them
        accordingly.

        The output parameter_vector can be used in simulate_data() to generate
        data according to the given input parameters.

        Parameters
        ----------
        parameters: keyword arguments of parameter_names.
            Can be given as **parameter_dictionary that contains the model
            parameter values.

        Returns
        -------
        parameter_vector: array of size (Ndata_x, Ndata_y, ..., Nparameters),
            array that contains the linearized model parameters for an ND-array
            of data voxels.
        """
        parameter_vector = []
        parameter_shapes = []
        for parameter, card in self.parameter_cardinality.items():
            value = np.atleast_1d(parameters[parameter])
            if card == 1 and not np.all(value.shape == np.r_[1]):
                parameter_shapes.append(value.shape)
            elif card > 1 and not np.all(value.shape == np.r_[card]):
                parameter_shapes.append(value.shape[:-1])

        if len(set(parameter_shapes)) > 1:
            msg = "parameter shapes are inconsistent."
            raise ValueError(msg)
        elif len(set(parameter_shapes)) == 0:
            for parameter, card in self.parameter_cardinality.items():
                parameter_vector.append(parameters[parameter])
            parameter_vector = np.hstack(parameter_vector)
        elif len(set(parameter_shapes)) == 1:
            for parameter, card in self.parameter_cardinality.items():
                value = np.atleast_1d(parameters[parameter])
                if card == 1 and np.all(value.shape == np.r_[1]):
                    parameter_vector.append(
                        np.tile(value[0], np.r_[parameter_shapes[0], 1]))
                elif card == 1 and not np.all(value.shape == np.r_[1]):
                    parameter_vector.append(value[..., None])
                elif card > 1 and np.all(value.shape == np.r_[card]):
                    parameter_vector.append(
                        np.tile(value, np.r_[parameter_shapes[0], 1])
                    )
                else:
                    parameter_vector.append(parameters[parameter])
            parameter_vector = np.concatenate(parameter_vector, axis=-1)
        return parameter_vector

    def parameter_initial_guess_to_parameter_vector(self, **parameters):
        """Function that returns a parameter_vector while allowing for partial
        input of model parameters, setting the ones that were not given to
        'None'. Such an array can be given to the fit() function to provide an
        initial parameter guess when fitting the data to the model.

        Parameters
        ----------
        parameters: keyword arguments of parameter names,
            parameter values of only the parameters you want to give as an
            initial condition for the optimizer.

        Returns
        -------
        parameter_vector: array of size (Ndata_x, Ndata_y, ..., Nparameters),
            array that contains the linearized model parameters for an ND-array
            of data voxels, with None's for non-given parameters.
        """
        set_parameters = {}
        parameter_cardinality = self.parameter_cardinality.copy()
        for parameter, value in parameters.items():
            if parameter in self.parameter_cardinality.keys():
                set_parameters[parameter] = value
                del parameter_cardinality[parameter]
            else:
                msg = '"{}" is not a valid model parameter.'.format(parameter)
                raise ValueError(msg)
        if len(parameter_cardinality) == 0:
            print("All model parameters set.")
        else:
            for parameter, card in parameter_cardinality.items():
                set_parameters[parameter] = np.tile(np.nan, card)
        return self.parameters_to_parameter_vector(**set_parameters)

    def _prepare_parameters(self):
        """Prepares the parameter ranges, scales, cadinality and parameter
        upon instantiating the MultiCompartmentModel"""
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

        self.parameter_ranges = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_ranges.items()
        })

        self.parameter_scales = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_scales.items()
        })

        self.parameter_types = OrderedDict({
            model_name + k: v
            for model, model_name in zip(self.models, self.model_names)
            for k, v in model.parameter_types.items()
        })

        self._parameter_map = {
            model_name + k: (model, k)
            for model, model_name in zip(self.models, self.model_names)
            for k in model.parameter_ranges
        }

        self._inverted_parameter_map = {
            v: k for k, v in self._parameter_map.items()
        }

        self.parameter_cardinality = OrderedDict([
            (k, len(np.atleast_2d(self.parameter_ranges[k])))
            for k in self.parameter_ranges
        ])

    def _prepare_partial_volumes(self):
        "Prepares partial volumes upon instantiating the MultiCompartmentModel"
        if len(self.models) > 1:
            self.partial_volume_names = [
                'partial_volume_{:d}'.format(i)
                for i in range(len(self.models))
            ]

            for i, partial_volume_name in enumerate(
                    self.partial_volume_names):
                self.parameter_ranges[partial_volume_name] = (0.01, .99)
                self.parameter_scales[partial_volume_name] = 1.
                self._parameter_map[partial_volume_name] = (
                    None, partial_volume_name
                )
                self.parameter_types[partial_volume_name] = 'normal'
                self._inverted_parameter_map[(None, partial_volume_name)] = \
                    partial_volume_name
                self.parameter_cardinality[partial_volume_name] = 1
        else:
            self.partial_volume_names = []

    def _prepare_parameter_links(self):
        """Prepares parameter links if given as input to MultiCompartmentModel.
        It first checks if the parameter that will be linked exists. If so,
        then it removes it from the parameter ranges, scales and cardinality,
        so it will not be optimized (as it will be a function of other
        parameters)."""
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

            del self.parameter_ranges[parameter_name]
            del self.parameter_cardinality[parameter_name]
            del self.parameter_scales[parameter_name]
            del self.parameter_types[parameter_name]
            del self.parameter_optimization_flags[parameter_name]

    def _prepare_model_properties(self):
        """Checks that spherical mean and regular models cannot be optimized
        together, and whether the model can estimate a Fiber Orientation
        Distribution (FOD)."""
        self.fod_available = False
        for model in self.models:
            try:
                model.fod
                self.fod_available = True
            except AttributeError:
                pass

    def _check_for_double_model_class_instances(self):
        "Checks all models have unique class instances."
        if len(self.models) != len(set(self.models)):
            msg = "Each model in the multi-compartment model must be "
            msg += "instantiated separately. For example, to make a model "
            msg += "with two sticks, the models must be given as "
            msg += "models = [stick1, stick2], not as "
            msg += "models = [stick1, stick1]."
            raise ValueError(msg)

    def add_linked_parameters_to_parameters(self, parameters):
        """When making the MultiCompartmentModel function call, adds the linked
        parameter to the optimized parameters by evaluating the parameter link
        function."""
        if len(self.parameter_links) == 0:
            return parameters
        parameters = parameters.copy()
        for parameter in self.parameter_links[::-1]:
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
                        argument_name
                    ))

                parameters[parameter_name] = parameter_function(
                    *argument_values
                )
            else:
                parameters[parameter_name] = parameter_function()
        return parameters

    def _prepare_parameters_to_optimize(self):
        "Sets up which parmameters to optimize."
        self.parameter_optimization_flags = OrderedDict({
            k: True
            for k, v in self.parameter_cardinality.items()
        })

    @property
    def bounds_for_optimization(self):
        "Returns the linear parameter bounds for the model optimization."
        bounds = []
        for parameter, card in self.parameter_cardinality.items():
            range_ = self.parameter_ranges[parameter]
            if card == 1:
                bounds.append(range_)
            else:
                for i in range(card):
                    bounds.append((range_[i][0], range_[i][1]))
        return bounds

    @property
    def opt_params_for_optimization(self):
        "Returns the linear bools whether to optimize a model parameter."
        params = []
        for parameter, card in self.parameter_cardinality.items():
            optimize_param = self.parameter_optimization_flags[parameter]
            if card == 1:
                params.append(optimize_param)
            else:
                for i in range(card):
                    params.append(optimize_param)
        return params

    @property
    def scales_for_optimization(self):
        "Returns the linear parameter scales for model optimization."
        return np.hstack([scale for parameter, scale in
                          self.parameter_scales.items()])

    def _check_for_tortuosity_constraint(self):
        for link in self.parameter_links:
            if link[2] is T1_tortuosity:
                msg = "Cannot use MIX optimization when the Tortuosity "
                msg += "constraint is set in the MultiCompartmentModel. To "
                msg += "use MIX while imposing Tortuosity, set the constraint "
                msg += "in the DistributedModel step."
                raise ValueError(msg)

    def set_initial_guess_parameter(self, parameter_name, value):
        """
        Allows the user to fix an optimization parameter to a static value.
        The fixed parameter will be removed from the optimized parameters and
        added as a linked parameter.

        Parameters
        ----------
        parameter_name: string
            name of the to-be-fixed parameters, see self.parameter_names.
        value: float or list of corresponding parameter_cardinality.
            the value to fix the parameter at in SI units.
        """
        if parameter_name in self.parameter_ranges.keys():
            card = self.parameter_cardinality[parameter_name]
            if card == 1:
                if isinstance(value, int) or isinstance(value, float):
                    self.x0_parameters[parameter_name] = value
                elif isinstance(value, np.ndarray):
                    self._add_initial_guess_parameter_array(
                        parameter_name, value)
            elif card == 2:
                value = np.array(value, dtype=float)
                if value.shape[-1] != 2:
                    msg = '{} can only be fixed '.format(parameter_name)
                    msg += 'to an array or list with last dimension 2.'
                    raise ValueError(msg)
                if value.ndim == 1:
                    self.x0_parameters[parameter_name] = value
                if value.ndim > 1:
                    self._add_initial_guess_parameter_array(
                        parameter_name, value)
        else:
            msg = '{} does not exist or has already been fixed.'.format(
                parameter_name)
            raise ValueError(msg)

    def _add_initial_guess_parameter_array(
            self, parameter_name, parameter_array):
        temp_dict = self.x0_parameters.copy()
        temp_dict[parameter_name] = parameter_array
        try:
            self.parameter_initial_guess_to_parameter_vector(
                **temp_dict)
            self.x0_parameters = temp_dict
        except ValueError:
            msg = '{} does not have the same shape'.format(parameter_name)
            msg += 'as the previously fixed parameters.'
            raise ValueError(msg)

    def set_fixed_parameter(self, parameter_name, value):
        """
        Allows the user to fix an optimization parameter to a static value.
        The fixed parameter will be removed from the optimized parameters and
        added as a linked parameter.

        Parameters
        ----------
        parameter_name: string
            name of the to-be-fixed parameters, see self.parameter_names.
        value: float or list of corresponding parameter_cardinality.
            the value to fix the parameter at in SI units.
        """
        if parameter_name in self.parameter_ranges.keys():
            card = self.parameter_cardinality[parameter_name]
            if card == 1:
                if isinstance(value, int) or isinstance(value, float):
                    self._add_fixed_parameter_value(parameter_name,
                                                    float(value))
                elif isinstance(value, np.ndarray):
                    self._add_fixed_parameter_array(parameter_name, value)
            elif card == 2:
                value = np.array(value, dtype=float)
                if value.shape[-1] != 2:
                    msg = '{} can only be fixed '.format(parameter_name)
                    msg += 'to an array or list with last dimension 2.'
                    raise ValueError(msg)
                if value.ndim == 1:
                    self._add_fixed_parameter_value(parameter_name, value)
                if value.ndim > 1:
                    self._add_fixed_parameter_array(parameter_name, value)
        else:
            msg = '{} does not exist or has already been fixed.'.format(
                parameter_name)
            raise ValueError(msg)

    def _add_fixed_parameter_value(self, parameter_name, value):
        model, name = self._parameter_map[parameter_name]
        parameter_link = (model, name, ReturnFixedValue(value), [])
        self.parameter_links.append(parameter_link)
        del self.parameter_ranges[parameter_name]
        del self.parameter_cardinality[parameter_name]
        del self.parameter_scales[parameter_name]
        del self.parameter_types[parameter_name]
        del self.parameter_optimization_flags[parameter_name]

    def _add_fixed_parameter_array(self, parameter_name, parameter_array):
        temp_dict = self.x0_parameters.copy()
        temp_dict[parameter_name] = parameter_array
        try:
            self.parameter_initial_guess_to_parameter_vector(
                **temp_dict)
            self.x0_parameters = temp_dict
            self.parameter_optimization_flags[parameter_name] = False
        except ValueError:
            msg = '{} does not have the same shape'.format(parameter_name)
            msg += 'as the previously fixed parameters.'
            raise ValueError(msg)

    def set_tortuous_parameter(self, lambda_perp_parameter_name,
                               lambda_par_parameter_name,
                               volume_fraction_intra_parameter_name,
                               volume_fraction_extra_parameter_name):
        """
        Allows the user to set a tortuosity constraint on the perpendicular
        diffusivity of the extra-axonal compartment, which depends on the
        intra-axonal volume fraction and parallel diffusivity.

        The perpendicular diffusivity parameter will be removed from the
        optimized parameters and added as a linked parameter.

        Parameters
        ----------
        lambda_perp_parameter_name: string
            name of the perpendicular diffusivity parameter, see
            self.parameter_names.
        lambda_par_parameter_name: string
            name of the parallel diffusivity parameter, see
            self.parameter_names.
        volume_fraction_intra_parameter_name: string
            name of the intra-axonal volume fraction parameter, see
            self.parameter_names.
        volume_fraction_extra_parameter_name: string
            name of the extra-axonal volume fraction parameter, see
            self.parameter_names.
        """
        params = [lambda_perp_parameter_name, lambda_par_parameter_name,
                  volume_fraction_intra_parameter_name,
                  volume_fraction_extra_parameter_name]
        for param in params:
            try:
                self.parameter_cardinality[param]
            except KeyError:
                msg = ("{} does not exist or has already been fixed.").format(
                    param)
                raise ValueError(msg)

        model, name = self._parameter_map[lambda_perp_parameter_name]
        self.parameter_links.append([model, name, T1_tortuosity, [
            self._parameter_map[lambda_par_parameter_name],
            self._parameter_map[volume_fraction_intra_parameter_name],
            self._parameter_map[volume_fraction_extra_parameter_name]]
        ])
        del self.parameter_ranges[lambda_perp_parameter_name]
        del self.parameter_cardinality[lambda_perp_parameter_name]
        del self.parameter_scales[lambda_perp_parameter_name]
        del self.parameter_types[lambda_perp_parameter_name]
        del self.parameter_optimization_flags[lambda_perp_parameter_name]

    def set_equal_parameter(self, parameter_name_in, parameter_name_out):
        """
        Allows the user to set two parameters equal to each other. This is used
        for example in the NODDI model to set the parallel diffusivities of the
        Stick and Zeppelin compartment to the same value.

        The second input parameter will be removed from the optimized
        parameters and added as a linked parameter.

        Parameters
        ----------
        parameter_name_in: string
            the first parameter name, see self.parameter_names.
        parameter_name_out: string,
            the second parameter name, see self.parameter_names. This is the
            parameter that will be removed form the optimzed parameters.
        """
        params = [parameter_name_in, parameter_name_out]
        for param in params:
            try:
                self.parameter_cardinality[param]
            except KeyError:
                msg = ("{} does not exist or has already been fixed.").format(
                    param)
                raise ValueError(msg)
        model, name = self._parameter_map[parameter_name_out]
        self.parameter_links.append([model, name, parameter_equality, [
            self._parameter_map[parameter_name_in]]])
        del self.parameter_ranges[parameter_name_out]
        del self.parameter_cardinality[parameter_name_out]
        del self.parameter_scales[parameter_name_out]
        del self.parameter_types[parameter_name_out]
        del self.parameter_optimization_flags[parameter_name_out]

    def set_fractional_parameter(self,
                                 parameter1_smaller_equal_than, parameter2):
        r"""
        Allows to impose a constraint to make one parameter smaller or equal to
        another parameter. This is done by replacing parameter1 with a
        new parameter that is defined as a fraction between 0 and 1 of
        parameter2. The new parameter will be the same as the old parameter
        name with "_fraction" appended to it.

        Parameters
        ----------
        parameter1_smaller_equal_than: string
            parameter name to be made a fraction of parameter2
        parameter2: string
            the parameter that is larger or equal than parameter1
        """
        params = [parameter1_smaller_equal_than, parameter2]
        for param in params:
            try:
                self.parameter_cardinality[param]
            except KeyError:
                msg = ("{} does not exist or has already been fixed.").format(
                    param)
                raise ValueError(msg)
        # append new parameter to parameters
        new_parameter_name = parameter1_smaller_equal_than + '_fraction'

        self._add_optimization_parameter(
            new_parameter_name, [0., 1.], 1., 1, 'normal', True)
        model, name = self._parameter_map[parameter1_smaller_equal_than]
        self.parameter_links.append([model, name, fractional_parameter, [
            self._parameter_map[new_parameter_name],
            self._parameter_map[parameter2]]])

        # remove old parameter1
        del self.parameter_ranges[parameter1_smaller_equal_than]
        del self.parameter_cardinality[parameter1_smaller_equal_than]
        del self.parameter_scales[parameter1_smaller_equal_than]
        del self.parameter_types[parameter1_smaller_equal_than]
        del self.parameter_optimization_flags[parameter1_smaller_equal_than]

    def _add_optimization_parameter(
            self,
            parameter_name,
            parameter_range,
            parameter_scale,
            parameter_card,
            parameter_type,
            parameter_flag):
        """
        Creates new ordered dictionaries for model properties with the
        optimization parameter on top.
        """
        old_parameter_ranges = self.parameter_ranges
        old_parameter_scales = self.parameter_scales
        old_parameter_cardinality = self.parameter_cardinality
        old_parameter_types = self.parameter_types
        old_optimization_flags = self.parameter_optimization_flags

        self.parameter_ranges = OrderedDict({parameter_name: parameter_range})
        self.parameter_scales = OrderedDict({parameter_name: parameter_scale})
        self.parameter_cardinality = OrderedDict(
            {parameter_name: parameter_card})
        self.parameter_types = OrderedDict({parameter_name: parameter_type})
        self.parameter_optimization_flags = OrderedDict(
            {parameter_name: parameter_flag})

        for name, _ in old_parameter_ranges.items():
            self.parameter_ranges.update({name: old_parameter_ranges[name]})
            self.parameter_scales.update({name: old_parameter_scales[name]})
            self.parameter_cardinality.update(
                {name: old_parameter_cardinality[name]})
            self.parameter_types.update({name: old_parameter_types[name]})
            self.parameter_optimization_flags.update(
                {name: old_optimization_flags[name]})

        self._parameter_map.update({parameter_name: (None, 'fraction')})
        self._inverted_parameter_map.update(
            {(None, 'fraction'): parameter_name})

    def _check_model_params_with_acquisition_params(self, acquisition_scheme):
        for model in self.models:
            for parameter in model._required_acquisition_parameters:
                if getattr(acquisition_scheme, parameter) is None:
                    msg = "{} is not compatible with ".format(
                        model.__class__.__name__)
                    msg += "given acquisition scheme because it needs "
                    msg += "{} as an acquisition parameter.".format(parameter)
                    raise ValueError(msg)

    def visualize_model_setup(
            self, view=True, cleanup=True, with_parameters=False,
            im_format='png'):
        """
        Visualizes MultiCompartmentModel setup using graphviz module. It uses
        the uuid module to create a unique identifier for each model in the
        MultiCompartmentModel to make sure each node is referenced in a unique
        way.

        If cleanup is set to False it will save the PDF of the graph in the
        current working directory.

        If with_parameters is set to true, it will include all the parameters
        of each model in the graph. Note the graph will ignore any parameter
        links that may have already been imposed (e.g. parameter equality or
        fixed parameters).

        Parameters
        ----------
        view: boolean,
            Whether or not to visualize the graph in a popup screen.
        cleanup: boolean,
            Whether or not to delete the PDF file of the model setup.
        with_parameters: boolean,
            Whether or not to also visualize the parameters of each model.
        """
        dot = Digraph('Model Setup', format=im_format)
        base_model = self.__class__.__name__
        base_uuid = str(uuid4())
        dot.node(base_uuid, base_model)
        self._add_recursive_graph_node(dot, base_uuid, self, with_parameters)
        dot.render('Model Setup', view=view, cleanup=cleanup)

    def _add_recursive_graph_node(
            self, graph_model, entry_uuid, entry_model, with_parameters):
        """
        Recursive function to visualize model setup. For every model in a
        MultiCompartmentModel or a distributed model it will check if it is
        a distribution, in which case the function will call itself with the
        sub-model as input and continue until it has found the bottom of the
        model setup.

        Parameters
        ----------
        graph_model: graphviz model instance,
            Instantiated model instance to keep growing with nodes.
        entry_uuid: string,
            Entry model unique identifier from which to keep growing the graph.
        entry_model: dmipy model instance,
            Entry dmipy model from which to keep growing the graph.
        """
        for sub_model in entry_model.models:
            model_name = sub_model.__class__.__name__
            model_uuid = str(uuid4())
            graph_model.node(model_uuid, model_name)
            graph_model.edge(model_uuid, entry_uuid)
            if (sub_model._model_type == 'SphericalDistributedModel' or
                    sub_model._model_type == 'SpatialDistributedModel'):
                self._add_recursive_graph_node(
                    graph_model, model_uuid, sub_model, with_parameters)
            elif with_parameters:
                self._add_parameter_nodes(graph_model, model_uuid, sub_model)
        if hasattr(entry_model, 'distribution'):
            dist_name = entry_model.distribution.__class__.__name__
            dist_uuid = str(uuid4())
            graph_model.node(dist_uuid, dist_name)
            graph_model.edge(dist_uuid, entry_uuid)
            if with_parameters:
                self._add_parameter_nodes(
                    graph_model, dist_uuid, entry_model.distribution)

    def _add_parameter_nodes(self, graph_model, entry_uuid, entry_model):
        """
        Adds the parameters to the graph truee if with_parameters=True in the
        visualize_model_setup function.

        Parameters
        ----------
        graph_model: graphviz model instance,
            Instantiated model instance to keep growing with nodes.
        entry_uuid: string,
            Entry model unique identifier from which to keep growing the graph.
        entry_model: dmipy model instance,
            Entry dmipy model from which to keep growing the graph.
        """
        for parameter_name in entry_model.parameter_names:
            parameter_uuid = str(uuid4())
            graph_model.node(parameter_uuid, parameter_name)
            graph_model.edge(parameter_uuid, entry_uuid)

    def _check_tissue_model_acquisition_scheme(self, acquisition_scheme):
        """Tests if acquisition scheme between MC-model and tissue response
        model are the same.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        """
        for model in self.models:
            if model._model_type == 'TissueResponseModel':
                mc_scheme_params = [
                    acquisition_scheme.shell_bvalues,
                    acquisition_scheme.shell_delta,
                    acquisition_scheme.shell_Delta,
                    acquisition_scheme.shell_gradient_strengths]
                tr_scheme_params = [
                    model.acquisition_scheme.shell_bvalues,
                    model.acquisition_scheme.shell_delta,
                    model.acquisition_scheme.shell_Delta,
                    model.acquisition_scheme.shell_gradient_strengths]
                try:
                    np.testing.assert_array_almost_equal(
                        mc_scheme_params, tr_scheme_params)
                except AssertionError:
                    msg = "Acquisition scheme of MC-model and tissue response "
                    msg += "model are not the same."
                    raise ValueError(msg)


class MultiCompartmentModel(MultiCompartmentModelProperties):
    r'''
    The MultiCompartmentModel class allows to combine any number of
    CompartmentModels and DistributedModels into one combined model that can
    be used to fit and simulate dMRI data.

    Parameters
    ----------
    models : list of N CompartmentModel instances,
        the models to combine into the MultiCompartmentModel.
    parameter_links : list of iterables (model, parameter name, link function,
        argument list),
        deprecated, for testing only.
    '''

    def __init__(self, models, parameter_links=None):
        self.models = models
        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        self._prepare_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._prepare_model_properties()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()
        self._check_for_NMR_and_other_models()
        self.x0_parameters = {}

        if not have_numba:
            msg = "We highly recommend installing numba for faster function "
            msg += "execution and model fitting."
            print(msg)
        if not have_pathos:
            msg = "We highly recommend installing pathos to take advantage of "
            msg += "multicore processing."
            print(msg)

    def _check_for_NMR_and_other_models(self):
        model_types = [model._model_type for model in self.models]
        if "NMRModel" in model_types:
            if len(np.unique(model_types)) > 1:
                msg = "Cannot combine 1D-NMR and other 3D model types together"
                msg += " into a MultiCompartmentModel."
                raise ValueError(msg)

    def fit(self, acquisition_scheme, data,
            mask=None, solver='brute2fine', Ns=5, maxiter=300,
            N_sphere_samples=30, use_parallel_processing=have_pathos,
            number_of_processors=None):
        """ The main data fitting function of a MultiCompartmentModel.

        This function can fit it to an N-dimensional dMRI data set, and returns
        a FittedMultiCompartmentModel instance that contains the fitted
        parameters and other useful functions to study the results.

        No initial guess needs to be given to fit a model, but a partial or
        complete initial guess can be given if the user wants to have a
        solution that is a local minimum close to that guess. The
        parameter_initial_guess input can be created using
        parameter_initial_guess_to_parameter_vector().

        A mask can also be given to exclude voxels from fitting (e.g. voxels
        that are outside the brain). If no mask is given then all voxels are
        included.

        An optimization approach can be chosen as either 'brute2fine' or 'mix'.
        - Choosing brute2fine will first use a brute-force optimization to find
          an initial guess for parameters without one, and will then refine the
          result using gradient-descent-based optimization.

          Note that given no initial guess will make brute2fine precompute an
          global parameter grid that will be re-used for all voxels, which in
          many cases is much faster than giving voxel-varying initial condition
          that requires a grid to be estimated per voxel.

        - Choosing mix will use the recent MIX algorithm based on separation of
          linear and non-linear parameters. MIX first uses a stochastic
          algorithm to find the non-linear parameters (non-volume fractions),
          then estimates the volume fractions while fixing the estimates of the
          non-linear parameters, and then finally refines the solution using
          a gradient-descent-based algorithm.

        The fitting process can be readily parallelized using the optional
        "pathos" package. If it is installed then it will automatically use it,
        but it can be turned off by setting use_parallel_processing=False. The
        algorithm will automatically use all cores in the machine, unless
        otherwise specified in number_of_processors.

        Data with multiple TE are normalized in separate segments using the
        b0-values according that TE.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        data : N-dimensional array of size (N_x, N_y, ..., N_dwis),
            The measured DWI signal attenuation array of either a single voxel
            or an N-dimensional dataset.
        mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
            Optional mask of voxels to be included in the optimization.
        solver : string,
            Selection of optimization algorithm.
            - 'brute2fine' to use brute-force optimization.
            - 'mix' to use Microstructure Imaging of Crossing (MIX)
              optimization.
        Ns : integer,
            for brute optimization, decised how many steps are sampled for
            every parameter.
        maxiter : integer,
            for MIX optimization, how many iterations are allowed.
        N_sphere_samples : integer,
            for brute optimization, how many spherical orientations are sampled
            for 'mu'.
        use_parallel_processing : bool,
            whether or not to use parallel processing using pathos.
        number_of_processors : integer,
            number of processors to use for parallel processing. Defaults to
            the number of processors in the computer according to cpu_count().

        Returns
        -------
        FittedCompartmentModel: class instance that contains fitted parameters,
            Can be used to recover parameters themselves or other useful
            functions.
        """
        self._check_tissue_model_acquisition_scheme(acquisition_scheme)
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        # estimate S0
        self.scheme = acquisition_scheme
        data_ = np.atleast_2d(data)
        if self.scheme.TE is None or len(np.unique(self.scheme.TE)) == 1:
            S0 = np.mean(data_[..., self.scheme.b0_mask], axis=-1)
        else:  # if multiple TE are in the data
            S0 = np.ones_like(data_)
            for TE_ in self.scheme.shell_TE:
                TE_mask = self.scheme.TE == TE_
                TE_b0_mask = np.all([self.scheme.b0_mask, TE_mask], axis=0)
                S0[..., TE_mask] = np.mean(
                    data_[..., TE_b0_mask], axis=-1)[..., None]

        if mask is None:
            mask = data_[..., 0] > 0
        else:
            mask = np.all([mask, data_[..., 0] > 0], axis=0)
        mask_pos = np.where(mask)

        N_parameters = len(self.bounds_for_optimization)
        N_voxels = np.sum(mask)

        # make starting parameters and data the same size
        x0_ = self.parameter_initial_guess_to_parameter_vector(
            **self.x0_parameters)
        x0_ = homogenize_x0_to_data(
            data_, x0_)
        x0_bool = np.all(
            np.isnan(x0_), axis=tuple(np.arange(x0_.ndim - 1)))
        x0_[..., ~x0_bool] /= self.scales_for_optimization[~x0_bool]

        if use_parallel_processing and not have_pathos:
            msg = 'Cannot use parallel processing without pathos.'
            raise ValueError(msg)
        elif use_parallel_processing and have_pathos:
            fitted_parameters_lin = [None] * N_voxels
            if number_of_processors is None:
                number_of_processors = cpu_count()
            pool = pp.ProcessPool(number_of_processors)
            print('Using parallel processing with {} workers.'.format(
                number_of_processors))
        else:
            fitted_parameters_lin = np.empty(
                np.r_[N_voxels, N_parameters], dtype=float)

        start = time()
        if solver == 'brute2fine':
            global_brute = GlobalBruteOptimizer(
                self, self.scheme, x0_, Ns, N_sphere_samples)
            fit_func = Brute2FineOptimizer(self, self.scheme, Ns)
            print('Setup brute2fine optimizer in {} seconds'.format(
                time() - start))
        elif solver == 'mix':
            self._check_for_tortuosity_constraint()
            fit_func = MixOptimizer(self, self.scheme, maxiter)
            print('Setup MIX optimizer in {} seconds'.format(
                time() - start))
        else:
            msg = "Unknown solver name {}".format(solver)
            raise ValueError(msg)
        self.optimizer = fit_func

        start = time()
        for idx, pos in enumerate(zip(*mask_pos)):
            voxel_E = data_[pos] / S0[pos]
            voxel_x0_vector = x0_[pos]
            if solver == 'brute2fine':
                if global_brute.global_optimization_grid is True:
                    voxel_x0_vector = global_brute(voxel_E)
            fit_args = (voxel_E, voxel_x0_vector)

            if use_parallel_processing:
                fitted_parameters_lin[idx] = pool.apipe(fit_func, *fit_args)
            else:
                fitted_parameters_lin[idx] = fit_func(*fit_args)
        if use_parallel_processing:
            fitted_parameters_lin = np.array(
                [p.get() for p in fitted_parameters_lin])

        fitting_time = time() - start
        print('Fitting of {} voxels complete in {} seconds.'.format(
            len(fitted_parameters_lin), fitting_time))
        print('Average of {} seconds per voxel.'.format(
            fitting_time / N_voxels))

        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = (
            fitted_parameters_lin * self.scales_for_optimization)

        return FittedMultiCompartmentModel(
            self, S0, mask, fitted_parameters)

    def simulate_signal(self, acquisition_scheme, parameters_array_or_dict):
        """
        Function to simulate diffusion data for a given acquisition_scheme
        and model parameters for the MultiCompartmentModel.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy
        model_parameters_array : 1D array of size (N_parameters) or
            N-dimensional array the same size as the data.
            The model parameters of the MultiCompartmentModel model.

        Returns
        -------
        E_simulated: 1D array of size (N_parameters) or N-dimensional
            array the same size as x0.
            The simulated signal of the microstructure model.
        """
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        Ndata = acquisition_scheme.number_of_measurements
        if isinstance(parameters_array_or_dict, np.ndarray):
            x0 = parameters_array_or_dict
        elif isinstance(parameters_array_or_dict, dict):
            x0 = self.parameters_to_parameter_vector(
                **parameters_array_or_dict)

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

    def __call__(self, acquisition_scheme_or_vertices,
                 quantity="signal", **kwargs):
        """
        The MultiCompartmentModel function call for to generate signal
        attenuation for a given acquisition scheme and model parameters.

        First, the linked parameters are added to the optimized parameters.

        Then, every model in the MultiCompartmentModel is called with the right
        parameters to recover the part of the signal attenuation of that model.
        The resulting values are multiplied with the volume fractions and
        finally the combined signal attenuation is returned.

        Aside from the signal, the function call can also return the Fiber
        Orientation Distributions (FODs) when a dispersed model is used, and
        can also return the stochastic cost function for the MIX algorithm.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        quantity : string
            can be 'signal', 'FOD' or 'stochastic cost function' depending on
            the need of the model.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
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
        if len(self.models) > 1:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = [1.]

        for model_name, model, partial_volume in zip(
            self.model_names, self.models, partial_volumes
        ):
            parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self._inverted_parameter_map[
                    (model, parameter)
                ]
                parameters[parameter] = kwargs.get(
                    # , self.parameter_defaults.get(parameter_name)
                    parameter_name
                )

            if quantity == "signal":
                values = (
                    values +
                    partial_volume * model(
                        acquisition_scheme_or_vertices, **parameters)
                )
            elif quantity == "FOD":
                try:
                    values = (
                        values +
                        partial_volume * model.fod(
                            acquisition_scheme_or_vertices, **parameters)
                    )
                except AttributeError:
                    continue
            elif quantity == "stochastic cost function":
                values[:, counter] = model(acquisition_scheme_or_vertices,
                                           **parameters)
                counter += 1
        return values


class MultiCompartmentSphericalMeanModel(MultiCompartmentModelProperties):
    r'''
    The MultiCompartmentModel class allows to combine any number of
    CompartmentModels and DistributedModels into one combined model that can
    be used to fit and simulate dMRI data.

    Parameters
    ----------
    models : list of N CompartmentModel instances,
        the models to combine into the MultiCompartmentModel.
    parameter_links : list of iterables (model, parameter name, link function,
        argument list),
        deprecated, for testing only.
    '''

    def __init__(self, models, parameter_links=None):
        self.models = models
        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        self._check_for_dispersed_or_NMR_models()
        self._prepare_parameters()
        self._delete_orientation_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._prepare_model_properties()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()
        self.x0_parameters = {}

        if not have_numba:
            msg = "We highly recommend installing numba for faster function "
            msg += "execution and model fitting."
            print(msg)
        if not have_pathos:
            msg = "We highly recommend installing pathos to take advantage of "
            msg += "multicore processing."
            print(msg)

    def _check_for_dispersed_or_NMR_models(self):
        for model in self.models:
            if model._model_type is 'NMRModel':
                msg = "Cannot estimate spherical mean of 1D-NMR models."
                raise ValueError(msg)
            if model._model_type is 'SphericalDistributedModel':
                msg = "Cannot estimate spherical mean spherically distributed "
                msg += "model. Please give the input models to the distributed"
                msg += " model directly to MultiCompartmentSphericalMeanModel."
                raise ValueError(msg)

    def _delete_orientation_parameters(self):
        """
        Deletes orientation parameters from input models 'mu' since they're not
        needed in spherical mean models.
        """
        "Removes orientation parameters from input models."
        for model in self.models:
            for param_name, param_type in model.parameter_types.items():
                if param_type == 'orientation':
                    appended_param_name = self._inverted_parameter_map[
                        model, param_name]
                    del self.parameter_ranges[appended_param_name]
                    del self.parameter_scales[appended_param_name]
                    del self.parameter_cardinality[appended_param_name]
                    del self.parameter_types[appended_param_name]

    def fit(self, acquisition_scheme, data,
            mask=None, solver='brute2fine', Ns=5, maxiter=300,
            N_sphere_samples=30, use_parallel_processing=have_pathos,
            number_of_processors=None):
        """ The main data fitting function of a MultiCompartmentModel.

        This function can fit it to an N-dimensional dMRI data set, and returns
        a FittedMultiCompartmentModel instance that contains the fitted
        parameters and other useful functions to study the results.

        No initial guess needs to be given to fit a model, but a partial or
        complete initial guess can be given if the user wants to have a
        solution that is a local minimum close to that guess. The
        parameter_initial_guess input can be created using
        parameter_initial_guess_to_parameter_vector().

        A mask can also be given to exclude voxels from fitting (e.g. voxels
        that are outside the brain). If no mask is given then all voxels are
        included.

        An optimization approach can be chosen as either 'brute2fine' or 'mix'.
        - Choosing brute2fine will first use a brute-force optimization to find
          an initial guess for parameters without one, and will then refine the
          result using gradient-descent-based optimization.

          Note that given no initial guess will make brute2fine precompute an
          global parameter grid that will be re-used for all voxels, which in
          many cases is much faster than giving voxel-varying initial condition
          that requires a grid to be estimated per voxel.

        - Choosing mix will use the recent MIX algorithm based on separation of
          linear and non-linear parameters. MIX first uses a stochastic
          algorithm to find the non-linear parameters (non-volume fractions),
          then estimates the volume fractions while fixing the estimates of the
          non-linear parameters, and then finally refines the solution using
          a gradient-descent-based algorithm.

        The fitting process can be readily parallelized using the optional
        "pathos" package. If it is installed then it will automatically use it,
        but it can be turned off by setting use_parallel_processing=False. The
        algorithm will automatically use all cores in the machine, unless
        otherwise specified in number_of_processors.

        Data with multiple TE are normalized in separate segments using the
        b0-values according that TE.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        data : N-dimensional array of size (N_x, N_y, ..., N_dwis),
            The measured DWI signal attenuation array of either a single voxel
            or an N-dimensional dataset.
        mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
            Optional mask of voxels to be included in the optimization.
        solver : string,
            Selection of optimization algorithm.
            - 'brute2fine' to use brute-force optimization.
            - 'mix' to use Microstructure Imaging of Crossing (MIX)
              optimization.
        Ns : integer,
            for brute optimization, decised how many steps are sampled for
            every parameter.
        maxiter : integer,
            for MIX optimization, how many iterations are allowed.
        N_sphere_samples : integer,
            for brute optimization, how many spherical orientations are sampled
            for 'mu'.
        use_parallel_processing : bool,
            whether or not to use parallel processing using pathos.
        number_of_processors : integer,
            number of processors to use for parallel processing. Defaults to
            the number of processors in the computer according to cpu_count().

        Returns
        -------
        FittedCompartmentModel: class instance that contains fitted parameters,
            Can be used to recover parameters themselves or other useful
            functions.
        """
        self._check_tissue_model_acquisition_scheme(acquisition_scheme)
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        # estimate S0
        self.scheme = acquisition_scheme
        data_ = np.atleast_2d(data)
        if self.scheme.TE is None or len(np.unique(self.scheme.TE)) == 1:
            S0 = np.mean(data_[..., self.scheme.b0_mask], axis=-1)
        else:  # if multiple TE are in the data
            S0 = np.ones(np.r_[data_.shape[:-1],
                               len(acquisition_scheme.shell_TE)])
            for TE_ in self.scheme.shell_TE:
                TE_mask = self.scheme.shell_TE == TE_
                TE_mask_shell = self.scheme.TE == TE_
                TE_b0_mask = np.all([self.scheme.b0_mask, TE_mask_shell],
                                    axis=0)
                S0[..., TE_mask] = np.mean(
                    data_[..., TE_b0_mask], axis=-1)[..., None]

        if mask is None:
            mask = data_[..., 0] > 0
        else:
            mask = np.all([mask, data_[..., 0] > 0], axis=0)
        mask_pos = np.where(mask)

        N_parameters = len(self.bounds_for_optimization)
        N_voxels = np.sum(mask)

        # make starting parameters and data the same size
        # make starting parameters and data the same size
        x0_ = self.parameter_initial_guess_to_parameter_vector(
            **self.x0_parameters)
        x0_ = homogenize_x0_to_data(
            data_, x0_)
        x0_bool = np.all(
            np.isnan(x0_), axis=tuple(np.arange(x0_.ndim - 1)))
        x0_[..., ~x0_bool] /= self.scales_for_optimization[~x0_bool]

        if use_parallel_processing and not have_pathos:
            msg = 'Cannot use parallel processing without pathos.'
            raise ValueError(msg)
        elif use_parallel_processing and have_pathos:
            fitted_parameters_lin = [None] * N_voxels
            if number_of_processors is None:
                number_of_processors = cpu_count()
            pool = pp.ProcessPool(number_of_processors)
            print('Using parallel processing with {} workers.'.format(
                number_of_processors))
        else:
            fitted_parameters_lin = np.empty(
                np.r_[N_voxels, N_parameters], dtype=float)

        # estimate the spherical mean of the data.
        data_to_fit = np.zeros(
            np.r_[data_.shape[:-1],
                  self.scheme.unique_dwi_indices.max() + 1])
        for pos in zip(*mask_pos):
            data_to_fit[pos] = estimate_spherical_mean_multi_shell(
                data_[pos], self.scheme)

        start = time()
        if solver == 'brute2fine':
            global_brute = GlobalBruteOptimizer(
                self, self.scheme,
                x0_, Ns, N_sphere_samples)
            fit_func = Brute2FineOptimizer(self, self.scheme, Ns)
            print('Setup brute2fine optimizer in {} seconds'.format(
                time() - start))
        elif solver == 'mix':
            self._check_for_tortuosity_constraint()
            fit_func = MixOptimizer(self, self.scheme, maxiter)
            print('Setup MIX optimizer in {} seconds'.format(
                time() - start))
        else:
            msg = "Unknown solver name {}".format(solver)
            raise ValueError(msg)
        self.optimizer = fit_func

        start = time()
        for idx, pos in enumerate(zip(*mask_pos)):
            voxel_E = data_to_fit[pos] / S0[pos]
            voxel_x0_vector = x0_[pos]
            if solver == 'brute2fine':
                if global_brute.global_optimization_grid is True:
                    voxel_x0_vector = global_brute(voxel_E)
            fit_args = (voxel_E, voxel_x0_vector)

            if use_parallel_processing:
                fitted_parameters_lin[idx] = pool.apipe(fit_func, *fit_args)
            else:
                fitted_parameters_lin[idx] = fit_func(*fit_args)
        if use_parallel_processing:
            fitted_parameters_lin = np.array(
                [p.get() for p in fitted_parameters_lin])

        fitting_time = time() - start
        print('Fitting of {} voxels complete in {} seconds.'.format(
            len(fitted_parameters_lin), fitting_time))
        print('Average of {} seconds per voxel.'.format(
            fitting_time / N_voxels))

        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = (
            fitted_parameters_lin * self.scales_for_optimization)

        return FittedMultiCompartmentSphericalMeanModel(
            self, S0, mask, fitted_parameters)

    def simulate_signal(self, acquisition_scheme, parameters_array_or_dict):
        """
        Function to simulate diffusion data for a given acquisition_scheme
        and model parameters for the MultiCompartmentModel.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy
        model_parameters_array : 1D array of size (N_parameters) or
            N-dimensional array the same size as the data.
            The model parameters of the MultiCompartmentModel model.

        Returns
        -------
        E_simulated: 1D array of size (N_parameters) or N-dimensional
            array the same size as x0.
            The simulated signal of the microstructure model.
        """
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        Ndata = acquisition_scheme.shell_indices.max() + 1
        if isinstance(parameters_array_or_dict, np.ndarray):
            x0 = parameters_array_or_dict
        elif isinstance(parameters_array_or_dict, dict):
            x0 = self.parameters_to_parameter_vector(
                **parameters_array_or_dict)

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

    def __call__(self, acquisition_scheme_or_vertices,
                 quantity="signal", **kwargs):
        """
        The MultiCompartmentModel function call for to generate signal
        attenuation for a given acquisition scheme and model parameters.

        First, the linked parameters are added to the optimized parameters.

        Then, every model in the MultiCompartmentModel is called with the right
        parameters to recover the part of the signal attenuation of that model.
        The resulting values are multiplied with the volume fractions and
        finally the combined signal attenuation is returned.

        Aside from the signal, the function call can also return the Fiber
        Orientation Distributions (FODs) when a dispersed model is used, and
        can also return the stochastic cost function for the MIX algorithm.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        quantity : string
            can be 'signal', 'FOD' or 'stochastic cost function' depending on
            the need of the model.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        if quantity == "signal":
            values = 0
        elif quantity == "stochastic cost function":
            values = np.empty((
                len(acquisition_scheme_or_vertices.shell_bvalues),
                len(self.models)
            ))
            counter = 0

        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )
        if len(self.models) > 1:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = [1.]

        for model_name, model, partial_volume in zip(
            self.model_names, self.models, partial_volumes
        ):
            parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self._inverted_parameter_map[
                    (model, parameter)
                ]
                parameters[parameter] = kwargs.get(
                    # , self.parameter_defaults.get(parameter_name)
                    parameter_name
                )

            if quantity == "signal":
                values = (
                    values +
                    partial_volume * model.spherical_mean(
                        acquisition_scheme_or_vertices, **parameters)
                )
            elif quantity == "stochastic cost function":
                values[:, counter] = model.spherical_mean(
                    acquisition_scheme_or_vertices,
                    **parameters)
                counter += 1
        return values


class MultiCompartmentSphericalHarmonicsModel(MultiCompartmentModelProperties):
    r'''
    The MultiCompartmentModel class allows to combine any number of
    CompartmentModels and DistributedModels into one combined model that can
    be used to fit and simulate dMRI data.

    Parameters
    ----------
    models : list of N CompartmentModel instances,
        the models to combine into the MultiCompartmentModel.
    '''

    def __init__(self, models, sh_order=8):
        self.models = models
        self.parameter_links = []

        self._check_for_dispersed_or_NMR_models()
        self._prepare_parameters()
        self._delete_orientation_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self._prepare_model_properties()
        self._check_for_double_model_class_instances()
        self._prepare_parameters_to_optimize()
        self._add_spherical_harmonics_parameters(sh_order)
        self._check_that_one_anisotropic_kernel_is_present()
        # self._check_for_tissue_response_models()

        self.x0_parameters = {}
        self.sh_order = sh_order

        if not have_numba:
            msg = "We highly recommend installing numba for faster function "
            msg += "execution and model fitting."
            print(msg)
        if not have_pathos:
            msg = "We highly recommend installing pathos to take advantage of "
            msg += "multicore processing."
            print(msg)

    def _check_for_dispersed_or_NMR_models(self):
        for model in self.models:
            if model._model_type is 'NMRModel':
                msg = "Cannot estimate spherical mean of 1D-NMR models."
                raise ValueError(msg)
            if model._model_type is 'SphericalDistributedModel':
                msg = "Cannot estimate spherical mean spherically distributed "
                msg += "model. Please give the input models to the distributed"
                msg += " model directly to MultiCompartmentSphericalMeanModel."
                raise ValueError(msg)

    def _delete_orientation_parameters(self):
        """
        Deletes orientation parameters from input models 'mu' since they're not
        needed in spherical mean models.
        """
        "Removes orientation parameters from input models."
        for model in self.models:
            for param_name, param_type in model.parameter_types.items():
                if param_type == 'orientation':
                    appended_param_name = self._inverted_parameter_map[
                        model, param_name]
                    del self.parameter_ranges[appended_param_name]
                    del self.parameter_scales[appended_param_name]
                    del self.parameter_cardinality[appended_param_name]
                    del self.parameter_types[appended_param_name]

    def _add_spherical_harmonics_parameters(self, sh_order):
        N_coef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.parameter_ranges['sh_coeff'] = [
            [-1e3, 1e3] for i in range(N_coef)]
        self.parameter_scales['sh_coeff'] = np.ones(N_coef, dtype=float)
        self.parameter_cardinality['sh_coeff'] = N_coef
        self.parameter_types['sh_coeff'] = 'sh_coefficients'
        self.parameter_optimization_flags['sh_coeff'] = True

    def _check_if_kernel_parameters_are_fixed(self):
        "checks if only volume fraction and sh_coeff parameters are optimized."
        self.volume_fractions_fixed = True
        for name, flag in self.parameter_optimization_flags.items():
            if flag is True:
                if (not name == 'sh_coeff' and
                        not name.startswith('partial_volume')):
                    msg = 'Kernel parameter {} is not fixed.'.format(name)
                    raise ValueError(msg)
                if name.startswith('partial_volume'):
                    self.volume_fractions_fixed = False
        if (not self.volume_fractions_fixed and
                self.multiple_anisotropic_kernels):
            msg = 'Cannot have multiple anisotropic kernels without having '
            msg += 'all volume fractions fixed.'
            raise ValueError(msg)

    def _check_that_one_anisotropic_kernel_is_present(self):
        "checks if one anisotropic kernel is given."
        orientation_counter = 0
        self.multiple_anisotropic_kernels = False
        for model in self.models:
            if 'orientation' in model.parameter_types.values():
                orientation_counter += 1
        if orientation_counter == 0:
            msg = 'MultiCompartmentSphericalHarmonicsModel must at least have '
            msg += 'one anisotropic kernel input model.'
            raise ValueError(msg)
        if orientation_counter > 1:
            self.multiple_anisotropic_kernels = True

    def fit(self, acquisition_scheme, data, mask=None, solver='csd',
            lambda_lb=1e-5, unity_constraint='kernel_dependent',
            fit_S0_response=False, use_parallel_processing=have_pathos,
            number_of_processors=None, verbose=True):
        """ The main data fitting function of a
        MultiCompartmentSphericalHarmonicsModel.

        This function can fit it to an N-dimensional dMRI data set, and returns
        a FittedMultiCompartmentModel instance that contains the fitted
        parameters and other useful functions to study the results.

        A mask can also be given to exclude voxels from fitting (e.g. voxels
        that are outside the brain). If no mask is given then all voxels are
        included.

        The fitting process can be readily parallelized using the optional
        "pathos" package. If it is installed then it will automatically use it,
        but it can be turned off by setting use_parallel_processing=False. The
        algorithm will automatically use all cores in the machine, unless
        otherwise specified in number_of_processors.

        Data with multiple TE are normalized in separate segments using the
        b0-values according that TE.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        data : N-dimensional array of size (N_x, N_y, ..., N_dwis),
            The measured DWI signal attenuation array of either a single voxel
            or an N-dimensional dataset.
        mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
            Optional mask of voxels to be included in the optimization.
        solver : string,
            Can be 'csd', 'csd_tounier07' or 'csd_cvxpy', with the default
            being 'csd'. Using 'csd' will make the algorithm automatically
            use the 'tournier07' solver [1]_ if there are no volume fractions
            to fit or they are fixed. Otherwise, the slower but more general
            cvxpy solver [2]_ is used, which follows the formulation of [3]_.
        lambda_lb: positive float,
            Weight for Laplace-Beltrami regularization to impose smoothness
            into estimated FODs, follows [4]_.
        unity_constraint: String or bool,
            Whether or not to constrain the volume fractions of the FOD to
            unity. The default is set to 'kernel_dependent', meaning it will
            enforce unity if the kernel is voxel-varying or when volume
            fractions are estimated. Otherwise unity_constraint is set to
            False.
        fit_S0_response: bool,
            whether or not to fit the raw signal or signal attenuation.
            default: False, the signal is automatically divided by S0-value.
            if True, the raw signal is fitted and the S0 intensities of the
            biophysical models are used in the signal generation. This is
            useful when using tissue_response_models for example.
        use_parallel_processing : bool,
            Whether or not to use parallel processing using pathos.
        number_of_processors : integer,
            Number of processors to use for parallel processing. Defaults to
            the number of processors in the computer according to cpu_count().

        Returns
        -------
        FittedCompartmentModel: class instance that contains fitted parameters,
            Can be used to recover parameters themselves or other useful
            functions.

        References
        ----------
        .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
            "Robust determination of the fibre orientation distribution in
            diffusion MRI: non-negativity constrained super-resolved spherical
            deconvolution." Neuroimage 35.4 (2007): 1459-1472.
        .. [2] Diamond, Steven, and Stephen Boyd. "CVXPY: A Python-embedded
            modeling language for convex optimization." The Journal of Machine
            Learning Research 17.1 (2016): 2909-2913.
        .. [3] Jeurissen, Ben, et al. "Multi-tissue constrained spherical
            deconvolution for improved analysis of multi-shell diffusion MRI
            data." NeuroImage 103 (2014): 411-426.
        .. [4] Descoteaux, Maxime, et al. "Regularized, fast, and robust
            analytical Q-ball imaging." Magnetic Resonance in Medicine: An
            Official Journal of the International Society for Magnetic
            Resonance in Medicine 58.3 (2007): 497-510.
        """
        self._check_if_kernel_parameters_are_fixed()
        self._check_tissue_model_acquisition_scheme(acquisition_scheme)
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        self.voxel_varying_kernel = False
        if bool(self.x0_parameters):  # if the dictionary is not empty
            self.voxel_varying_kernel = True

        if unity_constraint == 'kernel_dependent':
            self.unity_constraint = False
            if fit_S0_response:
                self.unity_constraint = False
            elif not self.volume_fractions_fixed or self.voxel_varying_kernel:
                self.unity_constraint = True
        else:
            self.unity_constraint = unity_constraint

        self.fit_S0_response = fit_S0_response
        if self.fit_S0_response:
            S0_responses = np.r_[[model.S0_response for model in self.models]]
            self.max_S0_response = S0_responses.max()
            self.S0_responses = S0_responses / self.max_S0_response
        else:
            self.S0_responses = np.ones(len(self.models), dtype=float)
            self.max_S0_response = 1.

        # estimate S0
        self.scheme = acquisition_scheme
        data_ = np.atleast_2d(data)
        # if self.tissue_response_kernels_present:
        #     S0 = np.ones(data_.shape[:-1], dtype=float) * S0_responses.max()
        if self.scheme.TE is None or len(np.unique(self.scheme.TE)) == 1:
            S0 = np.mean(data_[..., self.scheme.b0_mask], axis=-1)
        else:  # if multiple TE are in the data
            S0 = np.ones_like(data_)
            for TE_ in self.scheme.shell_TE:
                TE_mask = self.scheme.TE == TE_
                TE_b0_mask = np.all([self.scheme.b0_mask, TE_mask], axis=0)
                S0[..., TE_mask] = np.mean(
                    data_[..., TE_b0_mask], axis=-1)[..., None]

        if mask is None:
            mask = data_[..., 0] > 0
        else:
            mask = np.all([mask, data_[..., 0] > 0], axis=0)
        mask_pos = np.where(mask)

        N_parameters = len(self.bounds_for_optimization)
        N_voxels = np.sum(mask)

        # make starting parameters and data the same size
        x0_ = self.parameter_initial_guess_to_parameter_vector(
            **self.x0_parameters)
        x0_ = homogenize_x0_to_data(
            data_, x0_)

        start = time()
        if solver == 'csd':
            if self.volume_fractions_fixed:
                fit_func = CsdTournierOptimizer(
                    acquisition_scheme, self, x0_, self.sh_order,
                    unity_constraint=self.unity_constraint,
                    lambda_lb=lambda_lb)
                if use_parallel_processing:
                    msg = 'Parallel processing turned off for tournier07'
                    msg += ' optimizer because it does not improve fitting '
                    msg += 'speed.'
                    if verbose:
                        print(msg)
                    use_parallel_processing = False
                if verbose:
                    print(
                        'Setup Tournier07 FOD optimizer in {} seconds'.format(
                            time() - start))
            else:
                fit_func = CsdCvxpyOptimizer(
                    acquisition_scheme, self, x0_, self.sh_order,
                    unity_constraint=self.unity_constraint,
                    lambda_lb=lambda_lb)
                if verbose:
                    print('Setup CVXPY FOD optimizer in {} seconds'.format(
                        time() - start))
        elif solver == 'csd_tournier07':
            fit_func = CsdTournierOptimizer(
                acquisition_scheme, self, x0_, self.sh_order,
                unity_constraint=self.unity_constraint, lambda_lb=lambda_lb)
            if use_parallel_processing:
                msg = 'Parallel processing turned off for tournier07 optimizer'
                msg += ' because it does not improve fitting speed.'
                if verbose:
                    print(msg)
                use_parallel_processing = False
            if verbose:
                print('Setup Tournier07 FOD optimizer in {} seconds'.format(
                    time() - start))
        elif solver == 'csd_cvxpy':
            fit_func = CsdCvxpyOptimizer(
                acquisition_scheme, self, x0_, self.sh_order,
                unity_constraint=self.unity_constraint, lambda_lb=lambda_lb)
            if verbose:
                print('Setup CVXPY FOD optimizer in {} seconds'.format(
                    time() - start))
        else:
            msg = "Unknown solver name {}".format(solver)
            raise ValueError(msg)

        self.optimizer = fit_func

        if use_parallel_processing and not have_pathos:
            msg = 'Cannot use parallel processing without pathos.'
            raise ValueError(msg)
        elif use_parallel_processing and have_pathos:
            fitted_parameters_lin = [None] * N_voxels
            if number_of_processors is None:
                number_of_processors = cpu_count()
            pool = pp.ProcessPool(number_of_processors)
            if verbose:
                print('Using parallel processing with {} workers.'.format(
                    number_of_processors))
        else:
            fitted_parameters_lin = np.empty(
                np.r_[N_voxels, N_parameters], dtype=float)

        start = time()
        for idx, pos in enumerate(zip(*mask_pos)):
            if fit_S0_response:
                data_to_fit = data_[pos] / self.max_S0_response
            else:
                data_to_fit = data_[pos] / S0[pos]
            voxel_x0_vector = x0_[pos]
            fit_args = (data_to_fit, voxel_x0_vector)

            if use_parallel_processing:
                fitted_parameters_lin[idx] = pool.apipe(fit_func, *fit_args)
            else:
                fitted_parameters_lin[idx] = fit_func(*fit_args)
        if use_parallel_processing:
            fitted_parameters_lin = np.array(
                [p.get() for p in fitted_parameters_lin])

        fitting_time = time() - start
        if verbose:
            print('Fitting of {} voxels complete in {} seconds.'.format(
                len(fitted_parameters_lin), fitting_time))
            print('Average of {} seconds per voxel.'.format(
                fitting_time / N_voxels))
        fitted_parameters = np.zeros_like(x0_, dtype=float)
        fitted_parameters[mask_pos] = fitted_parameters_lin

        return FittedMultiCompartmentSphericalHarmonicsModel(
            self, S0, mask, fitted_parameters)

    def simulate_signal(self, acquisition_scheme, parameters_array_or_dict):
        """
        Function to simulate diffusion data for a given acquisition_scheme
        and model parameters for the MultiCompartmentModel.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy
        model_parameters_array : 1D array of size (N_parameters) or
            N-dimensional array the same size as the data.
            The model parameters of the MultiCompartmentModel model.

        Returns
        -------
        E_simulated: 1D array of size (N_parameters) or N-dimensional
            array the same size as x0.
            The simulated signal of the microstructure model.
        """
        self._check_model_params_with_acquisition_params(acquisition_scheme)

        Ndata = acquisition_scheme.number_of_measurements
        if isinstance(parameters_array_or_dict, np.ndarray):
            x0 = parameters_array_or_dict
        elif isinstance(parameters_array_or_dict, dict):
            x0 = self.parameters_to_parameter_vector(
                **parameters_array_or_dict)

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

    def __call__(self, acquisition_scheme, **kwargs):
        """
        The MultiCompartmentModel function call for to generate signal
        attenuation for a given acquisition scheme and model parameters.

        First, the linked parameters are added to the optimized parameters.

        Then, every model in the MultiCompartmentModel is called with the right
        parameters to recover the part of the signal attenuation of that model.
        The resulting values are multiplied with the volume fractions and
        finally the combined signal attenuation is returned.

        Aside from the signal, the function call can also return the Fiber
        Orientation Distributions (FODs) when a dispersed model is used, and
        can also return the stochastic cost function for the MIX algorithm.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        quantity : string
            can be 'signal', 'FOD' or 'stochastic cost function' depending on
            the need of the model.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )
        self.S0_responses = kwargs.get('S0_responses', self.S0_responses)
        self.fit_S0_response = kwargs.get(
            'fit_S0_response', self.fit_S0_response)

        A = self._construct_convolution_kernel(
            self.parameters_to_parameter_vector(**kwargs))

        # if vf fixed then just multiply with sh_coeff
        if self.volume_fractions_fixed:
            E = np.dot(A, kwargs['sh_coeff'])
        else:
            sh_coeff = np.zeros(self.optimizer.Ncoef_total)
            for i, name in enumerate(self.partial_volume_names):
                sh_coeff[self.optimizer.vf_indices[i]] = (
                    kwargs[name] / (2 * np.sqrt(np.pi)))
            sh_coeff[self.optimizer.sh_start:
                     self.optimizer.Ncoef + self.optimizer.sh_start] = kwargs[
                'sh_coeff']
            E = np.dot(A, sh_coeff)
        return E

    def _construct_convolution_kernel(self, x0_vector):
        """
        Helper function that constructs the convolution kernel for the given
        multi-compartment model and the initial condition x0_vector.

        First the parameter vector is converted to a dictionary with the
        corresponding parameter names. Then, the linked parameters are added to
        the given ones. Finally, the rotational harmonics of the model is
        passed to the construct_model_based_A_matrix, which constructs the
        kernel for an arbitrary PGSE-acquisition scheme.

        For multiple models with fixed volume fractions, the A-matrices
        are combined to have a combined convolution kernel.

        For multiple models without fixed volume fractions, the convolution
        kernels for anisotropic and isotropic models are concatenated, with
        the isotropic kernels always having a spherical harmonics order of 0.

        Parameters
        ----------
        x0_vector: array of size (N_parameters),
            Contains the fixed parameters of the convolution kernel.

        Returns
        -------
        kernel: array of size (N_coef, N_data),
            Observation matrix that maps the FOD spherical harmonics
            coefficients to the DWI signal values.
        """
        parameters_dict = self.parameter_vector_to_parameters(
            x0_vector)
        parameters_dict = self.add_linked_parameters_to_parameters(
            parameters_dict)

        if self.volume_fractions_fixed:
            if len(self.models) > 1:
                partial_volumes = [
                    parameters_dict[p] for p in self.partial_volume_names
                ]
            else:
                partial_volumes = [1.]
            kernel = 0.
            for model, partial_volume, S0 in zip(self.models,
                                                 partial_volumes,
                                                 self.S0_responses):
                parameters = {}
                for parameter in model.parameter_ranges:
                    parameter_name = self._inverted_parameter_map[
                        (model, parameter)
                    ]
                    parameters[parameter] = parameters_dict.get(
                        parameter_name
                    )
                model_rh = (
                    model.rotational_harmonics_representation(
                        self.scheme, **parameters))
                A_matrix = construct_model_based_A_matrix(
                    self.scheme, model_rh, self.sh_order)
                A_matrix = self._divide_by_positive_dirac_rh(A_matrix)
                kernel += S0 * partial_volume * A_matrix
        else:
            kernel = []
            for model, S0 in zip(self.models, self.S0_responses):
                parameters = {}
                for parameter in model.parameter_ranges:
                    parameter_name = self._inverted_parameter_map[
                        (model, parameter)
                    ]
                    parameters[parameter] = parameters_dict.get(
                        parameter_name
                    )
                model_rh = (
                    model.rotational_harmonics_representation(
                        self.scheme, **parameters))
                if 'orientation' in model.parameter_types.values():
                    A_matrix = construct_model_based_A_matrix(
                        self.scheme, model_rh, self.sh_order)
                    A_matrix = self._divide_by_positive_dirac_rh(A_matrix)
                    kernel.append(S0 * A_matrix)
                else:
                    kernel.append(S0 * construct_model_based_A_matrix(
                        self.scheme, model_rh, 0))

            kernel = np.hstack(kernel)
        return kernel

    def _divide_by_positive_dirac_rh(self, A_matrix):
        import cvxpy
        from dipy.reconst.shm import gen_dirac, sph_harm_ind_list
        from dipy.data import get_sphere, HemiSphere
        from dipy.reconst.shm import real_sym_sh_mrtrix

        sphere = get_sphere('symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        L_positivity = real_sym_sh_mrtrix(
            self.sh_order, hemisphere.theta, hemisphere.phi)[0]

        m, ll = sph_harm_ind_list(self.sh_order)
        dirac = gen_dirac(m, ll, 0, 0)
        dirac_data = np.dot(A_matrix, dirac)

        sh = cvxpy.Variable(A_matrix.shape[1])
        constraints = [L_positivity * sh >= 0.]
        cost = cvxpy.sum_squares(A_matrix * sh - dirac_data)
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        problem.solve()

        positive_dirac = sh.value[m == 0]
        mult = positive_dirac / dirac[m == 0]

        for i, ll_ in enumerate(ll):
            A_matrix[:, i] /= mult[ll_ // 2]

        return A_matrix


def homogenize_x0_to_data(data, x0):
    """
    Function that checks if data and initial guess x0 are of the same size.
    If x0 is 1D, it will be tiled to be the same size as data.
    """
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


class ReturnFixedValue:
    "Parameter fixing class for parameter links."

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value
