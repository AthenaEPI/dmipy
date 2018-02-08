from . import distributions
from collections import OrderedDict
from itertools import chain
from ..utils.spherical_convolution import sh_convolution
from ..utils.utils import T1_tortuosity, parameter_equality
import copy
import numpy as np

__all__ = [
    'DistributedModel',
    'SD1WatsonDistributed',
    'SD2BinghamDistributed',
    'DD1GammaDistributed',
    'ReturnFixedValue'
]


class DistributedModel:
    "Contains various properties of distributed models."
    _spherical_mean = False

    def _check_for_double_model_class_instances(self):
        "Checks if there are no models with the same class instantiation."
        if len(self.models) != len(set(self.models)):
            msg = "Each model in the multi-compartment model must be "
            msg += "instantiated separately. For example, to make a model "
            msg += "with two sticks, the models must be given as "
            msg += "models = [stick1, stick2], not as "
            msg += "models = [stick1, stick1]."
            raise ValueError(msg)

    def _check_for_dispersable_models(self):
        """
        Checks if a model can be dispersed, i.e. has a rotational harmonics
        representation.
        """
        for i, model in enumerate(self.models):
            try:
                callable(model.rotational_harmonics_representation)
            except AttributeError:
                msg = "Cannot disperse models input {}. ".format(i)
                msg += "It has no rotational_harmonics_representation."
                raise AttributeError(msg)

    def _check_for_distributable_models(self):
        "Checks if the to-be distributed parameter is in the input models."
        for i, model in enumerate(self.models):
            if self.target_parameter not in model._parameter_ranges:
                msg = "Cannot distribute models input {}. ".format(i)
                msg += "It has no {} as parameter.".format(
                    self.target_parameter)
                raise AttributeError(msg)

    def _check_for_same_parameter_type(self):
        """
        For gamma distribution, checks if sphere/cylinder models are not
        mixed.
        """
        parameter_types = [model._parameter_types[self.target_parameter]
                           for model in self.models]
        if len(np.unique(parameter_types)) > 1:
            msg = "Cannot mix models with different parameter types. "
            msg += "Current input parameter types are {}.".format(
                parameter_types)
            raise AttributeError(msg)

    def _prepare_parameters(self, models_and_distribution):
        """
        Prepares the DistributedModel parameter properties for both the
        distribution and the distributed models.
        """
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
            for model, model_name in zip(models_and_distribution,
                                         self.model_names)
            for k, v in model.parameter_ranges.items()
        })

        self.parameter_scales = OrderedDict({
            model_name + k: v
            for model, model_name in zip(models_and_distribution,
                                         self.model_names)
            for k, v in model.parameter_scales.items()
        })

        self.parameter_types = OrderedDict({
            model_name + k: v
            for model, model_name in zip(models_and_distribution,
                                         self.model_names)
            for k, v in model.parameter_types.items()
        })

        self._parameter_map = {
            model_name + k: (model, k)
            for model, model_name in zip(models_and_distribution,
                                         self.model_names)
            for k in model.parameter_ranges
        }

        self._inverted_parameter_map = {
            v: k for k, v in self._parameter_map.items()
        }

        self.parameter_cardinality = OrderedDict([
            (k, len(np.atleast_2d(self.parameter_ranges[k])))
            for k in self.parameter_ranges
        ])

    def _delete_target_parameter_from_parameters(self):
        "Removes the to-be-distributed parameter from the parameter list."
        for model in self.models:
            parameter_name = self._inverted_parameter_map[
                (model, self.target_parameter)
            ]
            del self.parameter_ranges[parameter_name]
            del self.parameter_scales[parameter_name]
            del self.parameter_cardinality[parameter_name]
            del self.parameter_types[parameter_name]

    def _prepare_partial_volumes(self):
        "Prepares the partial volumes for the DistributedModel."
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
        "Prepares the parameter links, if they are given."
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
            del self.parameter_cardinality[parameter_name]
            del self.parameter_types[parameter_name]

    def add_linked_parameters_to_parameters(self, parameters):
        """
        Adds the linked parameters to the optimized parameters.

        Parameters
        ----------
        parameters: dictionary of model parameters,
            contains the optimized parameters.

        Returns
        -------
        parameters: dictionary of model parameters,
            contains the optimzed and linked parameters.
        """
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
            model, name = self._parameter_map[parameter_name]
            parameter_link = (model, name, ReturnFixedValue(value), [])
            self.parameter_links.append(parameter_link)
            del self.parameter_ranges[parameter_name]
            del self.parameter_scales[parameter_name]
            del self.parameter_cardinality[parameter_name]
            del self.parameter_types[parameter_name]
        else:
            print('{} does not exist or has already been fixed.').format(
                parameter_name)

    def set_tortuous_parameter(self, lambda_perp,
                               lambda_par,
                               volume_fraction_intra):
        """
        Allows the user to set a tortuosity constraint on the perpendicular
        diffusivity of the extra-axonal compartment, which depends on the
        intra-axonal volume fraction and parallel diffusivity.

        The perpendicular diffusivity parameter will be removed from the
        optimized parameters and added as a linked parameter.

        Parameters
        ----------
        lambda_perp: string
            name of the perpendicular diffusivity parameter, see
            self.parameter_names.
        lambda_par: string
            name of the parallel diffusivity parameter, see
            self.parameter_names.
        volume_fraction_intra: string
            name of the intra-axonal volume fraction parameter, see
            self.parameter_names.
        """
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
        del self.parameter_cardinality[lambda_perp]
        del self.parameter_types[lambda_perp]

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
        del self.parameter_cardinality[parameter_name_out]
        del self.parameter_types[parameter_name_out]

    def copy(self):
        """
        Retuns a different instantiation of the DistributedModel with the same
        configuration, which can be used together with the original in a
        MultiCompartmentModel. For example, to do NODDI with multiple
        orientations.
        """
        return copy.copy(self)

    @property
    def parameter_names(self):
        "Retuns the DistributedModel parameter names."
        return self.parameter_ranges.keys()

    def __call__(self, acquisition_scheme, **kwargs):
        """
        The DistributedModel function call. If the model is spherical
        distribution it will return a spherical harmonics-convolved model.
        If it is a spatial distribution then it returns an integrated model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        if (isinstance(self.distribution, distributions.SD1Watson) or
                isinstance(self.distribution, distributions.SD2Bingham)):
            return self.sh_convolved_model(acquisition_scheme, **kwargs)
        elif isinstance(self.distribution, distributions.DD1Gamma):
            return self.integrated_model(acquisition_scheme, **kwargs)
        else:
            msg = "Unknown distribution."
            raise ValueError(msg)

    def sh_convolved_model(self, acquisition_scheme, **kwargs):
        """
        The spherical harmonics convolved function call for spherically
        distributions like Watson and Bingham.

        First, the linked parameters are added to the optimized parameters, and
        the spherical harmonics of the spherical distribution are recovered.
        The volume fractions are also converted from nested to regular ones.

        Then, for every model in the DistributedModel, and for every
        acquisition shell, the rotational harmonics of the model are recovered
        and convolved with the distribution spherical harmonics. The resulting
        values are multiplied with the volume fractions and finally the
        dispersed signal attenuation is returned.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
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
                    qvalue=acquisition_scheme.shell_qvalues[shell_index],
                    gradient_strength=(
                        acquisition_scheme.shell_gradient_strengths[
                            shell_index]),
                    delta=acquisition_scheme.shell_delta[shell_index],
                    Delta=acquisition_scheme.shell_Delta[shell_index],
                    rh_order=acquisition_scheme.shell_sh_orders[shell_index],
                    **parameters)
                # convolving micro-environment with watson distribution
                E_dispersed_sh = sh_convolution(sh_distribution, rh_stick)
                # recover signal values from convolved spherical harmonics
                E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                       E_dispersed_sh)

            if partial_volume is not None:
                volume_fraction = remaining_volume_fraction * partial_volume
                remaining_volume_fraction = (
                    remaining_volume_fraction - volume_fraction)
            else:
                volume_fraction = remaining_volume_fraction

            values = values + volume_fraction * E
        return values

    def integrated_model(self, acquisition_scheme, **kwargs):
        """
        The spatially integrated function call for spatial distributions like
        Gamma. Currently, the model assumed we are distributing diameters.

        First, the linked parameters are added to the optimized parameters, and
        the probability that a water particle exists inside a cylinder of a
        certain size in the distribution is sampled for a range of diameters.
        The volume fractions are also converted from nested to regular ones
        (although typically not more than 1 model is used for a Gamma
        distribution).

        Then, for every model in the DistributedModel, the signal attenuations
        of the model are are recovered for the sampled diameters and multiplied
        and integrated over their probabilities. The resulting
        values are multiplied with the volume fractions and finally the
        integrated signal attenuation is returned.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
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
                remaining_volume_fraction = (
                    remaining_volume_fraction - volume_fraction)
            else:
                volume_fraction = remaining_volume_fraction

            values = values + volume_fraction * E
        return values

    def fod(self, vertices, **kwargs):
        """
        Returns the Fiber Orientation Distribution (FOD) of a dispersed model.

        Parameters
        ----------
        vertices: array of size (N_samples, 3)
            cartesian unit vectors at which to sample the FOD
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        distribution_parameters = {}
        for parameter in self.distribution.parameter_ranges:
            parameter_name = self._inverted_parameter_map[
                (self.distribution, parameter)
            ]
            distribution_parameters[parameter] = kwargs.get(
                parameter_name)
        return self.distribution(vertices, **distribution_parameters)


class SD1WatsonDistributed(DistributedModel):
    """
    The DistributedModel instantiation for a Watson-dispersed model. Multiple
    models can be dispersed at the same time (like a Stick and Zeppelin for
    NODDI for example). The parameter 'mu' of the models will be removed and
    replaced by the 'mu' and 'odi' of the Watson distribution.

    After instantiation the WatsonDistributed model can be treated exactly the
    same as a CompartmentModel as an input for a MultiCompartmentModel.

    Parameters
    ----------
    models: list of length 1 or more,
        list of models to be Watson-dispersed.
    parameters_links: list of length 1 or more,
        deprecated for testing use only.
    """
    _model_type = 'SphericalDistributedModel'

    def __init__(self, models, parameter_links=None):
        self.models = models
        self.target_parameter = "mu"
        self._check_for_double_model_class_instances()
        self._check_for_dispersable_models()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []
        self.distribution = distributions.SD1Watson()

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()


class SD2BinghamDistributed(DistributedModel):
    """
    The DistributedModel instantiation for a Bingham-dispersed model. Multiple
    models can be dispersed at the same time (like a Stick and Zeppelin for
    NODDI for example). The parameter 'mu' of the models will be removed and
    replaced by the 'mu', 'odi', 'beta_fraction' and 'psi' of the Bingham
    distribution.

    After instantiation the BinghamDistributed model can be treated exactly the
    same as a CompartmentModel as an input for a MultiCompartmentModel.

    Parameters
    ----------
    models: list of length 1 or more,
        list of models to be Watson-dispersed.
    parameters_links: list of length 1 or more,
        deprecated for testing use only.
    """
    _model_type = 'SphericalDistributedModel'

    def __init__(self, models, parameter_links=None):
        self.models = models
        self.target_parameter = "mu"
        self._check_for_double_model_class_instances()
        self._check_for_dispersable_models()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []
        self.distribution = distributions.SD2Bingham()

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()


class DD1GammaDistributed(DistributedModel):
    """
    The DistributedModel instantiation for a Gamma-distributed model for
    cylinder or sphere models. Multiple models can be distributed at the same
    time (but such multi-cylinder-distribution models have never been used as
    far as we know). The parameter 'diameter' of the models will be removed and
    replaced by the 'alpha' and 'beta', of the Gamma distribution.

    After instantiation the GammaDistributed model can be treated exactly the
    same as a CompartmentModel as an input for a MultiCompartmentModel.

    Parameters
    ----------
    models: list of length 1 or more,
        list of models to be Watson-dispersed.
    parameters_links: list of length 1 or more,
        deprecated for testing use only.
    """
    _model_type = 'SpatialDistributedModel'

    def __init__(self, models, parameter_links=None,
                 target_parameter='diameter'):
        self.models = models
        self.target_parameter = target_parameter
        self._check_for_double_model_class_instances()
        self._check_for_distributable_models()
        self._check_for_same_parameter_type()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        self.normalization = models[0]._parameter_types[target_parameter]
        self.distribution = distributions.DD1Gamma(
            normalization=self.normalization)

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()


class ReturnFixedValue:
    "Parameter fixing class for parameter links."

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value
