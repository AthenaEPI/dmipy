from mipy.distributions.distributions import (
    SD1Watson, SD2Bingham, DD1GammaDistribution)
from collections import OrderedDict
from itertools import chain
from mipy.utils.spherical_convolution import sh_convolution
from mipy.utils.utils import T1_tortuosity, parameter_equality
import copy
import numpy as np


class DistributedModel:
    spherical_mean = False

    def _check_for_double_model_class_instances(self):
        if len(self.models) != len(np.unique(self.models)):
            msg = "Each model in the multi-compartment model must be "
            msg += "instantiated separately. For example, to make a model "
            msg += "with two sticks, the models must be given as "
            msg += "models = [stick1, stick2], not as "
            msg += "models = [stick1, stick1]."
            raise ValueError(msg)

    def _check_for_dispersable_models(self):
        for i, model in enumerate(self.models):
            try:
                callable(model.rotational_harmonics_representation)
            except AttributeError:
                msg = "Cannot disperse models input {}. ".format(i)
                msg += "It has no rotational_harmonics_representation."
                raise AttributeError(msg)

    def _check_for_distributable_models(self):
        for i, model in enumerate(self.models):
            if self.target_parameter not in model._parameter_ranges:
                msg = "Cannot distribute models input {}. ".format(i)
                msg += "It has no {} as parameter.".format(
                    self.target_parameter)
                raise AttributeError(msg)

    def _check_for_same_model_type(self):
        model_types = [model._model_type for model in self.models]
        if len(np.unique(model_types)) > 1:
            msg = "Cannot mix models of different types. "
            msg += "Current input model types are {}.".format(
                model_types)
            raise AttributeError(msg)
        self.normalization = model_types[0]

    def _prepare_parameters(self, models_and_distribution):
        self.model_names = []
        model_counts = {}

        for model in models_and_distribution:
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
            for model, model_name in zip(models_and_distribution, self.model_names)
            for k, v in model.parameter_ranges.items()
        })

        self.parameter_scales = OrderedDict({
            model_name + k: v
            for model, model_name in zip(models_and_distribution, self.model_names)
            for k, v in model.parameter_scales.items()
        })

        self._parameter_map = {
            model_name + k: (model, k)
            for model, model_name in zip(models_and_distribution, self.model_names)
            for k in model.parameter_ranges
        }

        self._inverted_parameter_map = {
            v: k for k, v in self._parameter_map.items()
        }

    def _delete_target_parameter_from_parameters(self):
        for model in self.models:
            parameter_name = self._inverted_parameter_map[
                (model, self.target_parameter)
            ]
            del self.parameter_ranges[parameter_name]
            del self.parameter_scales[parameter_name]

    def _prepare_partial_volumes(self):
        if len(self.models) > 1:
            self.partial_volume_names = [
                'partial_volume_{:d}'.format(i)
                for i in range(len(self.models) - 1)
            ]

            for i, partial_volume_name in enumerate(self.partial_volume_names):
                self.parameter_ranges[partial_volume_name] = (0.01, .99)
                self.parameter_scales[partial_volume_name] = 1.
                self._parameter_map[partial_volume_name] = (
                    None, partial_volume_name
                )
                self._inverted_parameter_map[(None, partial_volume_name)] = \
                    partial_volume_name

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

            del self.parameter_ranges[parameter_name]
            del self.parameter_scales[parameter_name]

    def add_linked_parameters_to_parameters(self, parameters):
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

    def set_fixed_parameter(self, parameter_name, value):
        if parameter_name in self.parameter_ranges.keys():
            model, name = self._parameter_map[parameter_name]
            parameter_link = (model, name, ReturnFixedValue(value), [])
            self.parameter_links.append(parameter_link)
            del self.parameter_ranges[parameter_name]
            del self.parameter_scales[parameter_name]
        else:
            print ('{} does not exist or has already been fixed.').format(
                parameter_name)

    def set_tortuous_parameter(self, lambda_perp,
                               lambda_par,
                               volume_fraction_intra):
        params = [lambda_perp, lambda_par, volume_fraction_intra]
        for param in params:
            try:
                self.parameter_ranges[param]
            except KeyError:
                print("{} does not exist or has already been fixed.").format(
                    param)
                return None

        model, name = self._parameter_map[lambda_perp]
        self.parameter_links.append([model, name, T1_tortuosity, [
            self._parameter_map[lambda_par],
            self._parameter_map[volume_fraction_intra]]
        ])
        del self.parameter_ranges[lambda_perp]
        del self.parameter_scales[lambda_perp]

    def set_equal_parameter(self, parameter_name_in, parameter_name_out):
        params = [parameter_name_in, parameter_name_out]
        for param in params:
            try:
                self.parameter_ranges[param]
            except KeyError:
                print("{} does not exist or has already been fixed.").format(
                    param)
                return None
        model, name = self._parameter_map[parameter_name_out]
        self.parameter_links.append([model, name, parameter_equality, [
            self._parameter_map[parameter_name_in]]])
        del self.parameter_ranges[parameter_name_out]
        del self.parameter_scales[parameter_name_out]

    def copy(self):
        return copy.copy(self)

    @property
    def parameter_names(self):
        return self.parameter_ranges.keys()

    def __call__(self, acquisition_scheme, **kwargs):
        if (isinstance(self.distribution, SD1Watson) or
                isinstance(self.distribution, SD2Bingham)):
            return self.sh_convolved_model(acquisition_scheme, **kwargs)
        elif isinstance(self.distribution, DD1GammaDistribution):
            return self.integrated_model(acquisition_scheme, **kwargs)
        else:
            msg = "Unknown distribution."
            raise ValueError(msg)

    def sh_convolved_model(self, acquisition_scheme, **kwargs):
        values = 0.
        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )

        distribution_parameters = {}
        for parameter in self.distribution.parameter_ranges:
            parameter_name = self._inverted_parameter_map[
                (self.distribution, parameter)
            ]
            distribution_parameters[parameter] = kwargs.get(
                parameter_name)
        sh_distribution = self.distribution.spherical_harmonics_representation(
            **distribution_parameters)

        if len(self.models) > 1:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = []

        remaining_volume_fraction = 1.
        for model_name, model, partial_volume in zip(
            self.model_names, self.models,
            chain(partial_volumes, [None])
        ):
            parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self._inverted_parameter_map[
                    (model, parameter)
                ]
                parameters[parameter] = kwargs.get(
                    parameter_name
                )

            shell_indices = acquisition_scheme.shell_indices
            unique_dwi_indices = acquisition_scheme.unique_dwi_indices
            E = np.ones(acquisition_scheme.number_of_measurements)
            for shell_index in unique_dwi_indices:  # per shell
                shell_mask = shell_indices == shell_index
                sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
                # rotational harmonics of stick
                rh_stick = model.rotational_harmonics_representation(
                    bvalue=acquisition_scheme.shell_bvalues[shell_index],
                    rh_order=acquisition_scheme.shell_sh_orders[shell_index],
                    **parameters)
                # convolving micro-environment with watson distribution
                E_dispersed_sh = sh_convolution(sh_distribution, rh_stick)
                # recover signal values from watson-convolved spherical harmonics
                E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                       E_dispersed_sh)

            if partial_volume is not None:
                volume_fraction = remaining_volume_fraction * partial_volume
                remaining_volume_fraction = remaining_volume_fraction - volume_fraction
            else:
                volume_fraction = remaining_volume_fraction

            values = values + volume_fraction * E
        return values

    def integrated_model(self, acquisition_scheme, **kwargs):
        values = 0.
        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )

        distribution_parameters = {}
        for parameter in self.distribution.parameter_ranges:
            parameter_name = self._inverted_parameter_map[
                (self.distribution, parameter)
            ]
            distribution_parameters[parameter] = kwargs.get(
                parameter_name)
        radii, P_radii = self.distribution(**distribution_parameters)

        if len(self.models) > 1:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = []

        remaining_volume_fraction = 1.
        for model_name, model, partial_volume in zip(
            self.model_names, self.models,
            chain(partial_volumes, [None])
        ):
            parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self._inverted_parameter_map[
                    (model, parameter)
                ]
                parameters[parameter] = kwargs.get(
                    parameter_name
                )
            E = np.empty(
                (len(radii),
                 acquisition_scheme.number_of_measurements))
            for i, radius in enumerate(radii):
                parameters['diameter'] = radius * 2
                E[i] = (
                    P_radii[i] *
                    model(acquisition_scheme, **parameters)
                )
            E = np.trapz(E, x=radii, axis=0)

            if partial_volume is not None:
                volume_fraction = remaining_volume_fraction * partial_volume
                remaining_volume_fraction = remaining_volume_fraction - volume_fraction
            else:
                volume_fraction = remaining_volume_fraction

            values = values + volume_fraction * E
        return values

    def fod(self, vertices, **kwargs):
        distribution_parameters = {}
        for parameter in self.distribution.parameter_ranges:
            parameter_name = self._inverted_parameter_map[
                (self.distribution, parameter)
            ]
            distribution_parameters[parameter] = kwargs.get(
                parameter_name)
        return self.distribution(vertices, **distribution_parameters)


class SD1WatsonDistributed(DistributedModel):

    def __init__(self, models, parameter_links=[]):
        self.models = models
        self.target_parameter = "mu"
        self._check_for_double_model_class_instances()
        self._check_for_dispersable_models()

        self.parameter_links = parameter_links
        self.distribution = SD1Watson()

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()


class SD2BinghamDistributed(DistributedModel):

    def __init__(self, models, parameter_links=[]):
        self.models = models
        self.target_parameter = "mu"
        self._check_for_double_model_class_instances()
        self._check_for_dispersable_models()

        self.parameter_links = parameter_links
        self.distribution = SD2Bingham()

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()


class DD1GammaDistributed(DistributedModel):

    def __init__(self, models, parameter_links=[]):
        self.models = models
        self.target_parameter = "diameter"
        self._check_for_double_model_class_instances()
        self._check_for_distributable_models()
        self._check_for_same_model_type()

        self.parameter_links = parameter_links
        self.distribution = DD1GammaDistribution(
            normalization=self.normalization)

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()


class ReturnFixedValue:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value
