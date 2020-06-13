from . import distributions
from collections import OrderedDict
from itertools import chain
from ..utils.spherical_convolution import sh_convolution
from ..utils.utils import T1_tortuosity, parameter_equality
from ..core.signal_model_properties import AnisotropicSignalModelProperties
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
            if self.target_parameter not in model.parameter_ranges:
                msg = "Cannot distribute models input {}. ".format(i)
                msg += "It has no {} as parameter.".format(
                    self.target_parameter)
                raise AttributeError(msg)

    def _check_for_same_parameter_type(self):
        """
        For gamma distribution, checks if sphere/cylinder models are not
        mixed.
        """
        parameter_types = [model.parameter_types[self.target_parameter]
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

    def _delete_orientation_from_parameters(self):
        "Removes orientation parameters from input models."
        orientation_counter = 0
        for model in self.models:
            for param_name, param_type in model.parameter_types.items():
                if param_type == 'orientation':
                    appended_param_name = self._inverted_parameter_map[
                        model, param_name]
                    del self.parameter_ranges[appended_param_name]
                    del self.parameter_scales[appended_param_name]
                    del self.parameter_cardinality[appended_param_name]
                    del self.parameter_types[appended_param_name]
                    orientation_counter += 1
        if orientation_counter == 0:
            msg = 'At least one input model must have an orientation '
            msg += 'parameter.'
            raise ValueError(msg)

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
                self.parameter_types[partial_volume_name] = 'normal'
                self._inverted_parameter_map[(None, partial_volume_name)] = \
                    partial_volume_name
                self.parameter_cardinality[partial_volume_name] = 1

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
            card = self.parameter_cardinality[parameter_name]
            if card == 1:
                if isinstance(value, int):
                    value = float(value)
                if not isinstance(value, float):
                    msg = '{} can only be fixed to a float value.'.format(
                        parameter_name)
                    raise ValueError(msg)
            elif card == 2:
                value = np.array(value, dtype=float)
                if value.shape != (2,):
                    msg = '{} can only be fixed '.format(parameter_name)
                    msg += 'to an array or list of length 2.'
                    raise ValueError(msg)
            model, name = self._parameter_map[parameter_name]
            parameter_link = (model, name, ReturnFixedValue(value), [])
            self.parameter_links.append(parameter_link)
            del self.parameter_ranges[parameter_name]
            del self.parameter_scales[parameter_name]
            del self.parameter_cardinality[parameter_name]
            del self.parameter_types[parameter_name]
        else:
            msg = '{} does not exist or has already been fixed.'.format(
                parameter_name)
            raise ValueError(msg)

    def set_tortuous_parameter(self, lambda_perp,
                               lambda_par,
                               volume_fraction_intra,
                               S0_intra=None,
                               S0_extra=None):
        """
        Allows the user to set a tortuosity constraint on the perpendicular
        diffusivity of the extra-axonal compartment, which depends on the
        intra-axonal volume fraction and parallel diffusivity.

        The perpendicular diffusivity parameter will be removed from the
        optimized parameters and added as a linked parameter.

        To employ the multi-tissue correction of tortuosity it is sufficient to
        pass the S0_intra and S0_extra parameters.


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
        S0_intra: float,
            S0 response of the tissue associated to the intra-cellular
            compartment. Default: 1 .
        S0_extra: float,
            S0 response of the tissue associated to the extra-cellular
            compartment. Default: 1.
        """
        params = [lambda_perp, lambda_par, volume_fraction_intra]
        for param in params:
            try:
                self.parameter_ranges[param]
            except KeyError:
                print("{} does not exist or has already been fixed.").format(
                    param)
                return None

        if S0_intra is None and S0_extra is None:
            S0_intra = 1.
            S0_extra = 1.
        elif S0_intra is not None and S0_extra is not None:
            if self.S0_tissue_responses is None:
                msg = ('The multi compartment model does not have an S0 for '
                       'each compartment. It is necessary in order to use the '
                       'tortuosity constraint with multi-tissue correction.')
                raise ValueError(msg)
            if S0_intra not in self.S0_tissue_responses:
                msg = ('The specified S0_intra does not correspond to any S0 '
                       'in the multi-compartment model.')
                raise ValueError(msg)
            if S0_extra not in self.S0_tissue_responses:
                msg = ('The specified S0_extra does not correspond to any S0 '
                       'in the multi-compartment model.')
                raise ValueError(msg)
            S0_intra = float(S0_intra)
            S0_extra = float(S0_extra)
        else:
            raise ValueError('Only one S0 has been specified. Both S0_intra '
                             'and S0_extra must be passed.')

        tortuosity = T1_tortuosity(S0_intra, S0_extra)
        model, name = self._parameter_map[lambda_perp]
        self.parameter_links.append([model, name, tortuosity, [
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

    def _set_required_acquisition_parameters(self):
        self._required_acquisition_parameters = []
        for model in self.models:
            self._required_acquisition_parameters += (
                model._required_acquisition_parameters)
        self._required_acquisition_parameters = np.unique(
            self._required_acquisition_parameters).tolist()

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
        if hasattr(self, 'distribution'):
            if (isinstance(self.distribution, distributions.SD1Watson) or
                    isinstance(self.distribution, distributions.SD2Bingham)):
                return self.sh_convolved_model(acquisition_scheme, **kwargs)
            elif isinstance(self.distribution, distributions.DD1Gamma):
                return self.integrated_model(acquisition_scheme, **kwargs)
        else:
            return self.bundle_model(acquisition_scheme, **kwargs)

    def bundle_model(self, acquisition_scheme, **kwargs):
        """
        Simple bundle model that does not apply any distribution. It can be
        considered a sub-multi-compartment model that allows for tortuosity
        constraints.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.
        """
        kwargs = self.add_linked_parameters_to_parameters(
            kwargs
        )

        if len(self.models) > 1:
            partial_volumes = [
                kwargs[p] for p in self.partial_volume_names
            ]
        else:
            partial_volumes = []

        remaining_volume_fraction = 1.
        E = 0.
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

            if partial_volume is not None:
                volume_fraction = remaining_volume_fraction * partial_volume
                remaining_volume_fraction = (
                    remaining_volume_fraction - volume_fraction)
            else:
                volume_fraction = remaining_volume_fraction
            E += volume_fraction * model(acquisition_scheme, **parameters)
        return E

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
        rh_models = 0.
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

            if partial_volume is not None:
                volume_fraction = remaining_volume_fraction * partial_volume
                remaining_volume_fraction = (
                    remaining_volume_fraction - volume_fraction)
            else:
                volume_fraction = remaining_volume_fraction

            rh_model = model.rotational_harmonics_representation(
                acquisition_scheme, **parameters)
            rh_models = rh_models + volume_fraction * rh_model

        E = np.ones(acquisition_scheme.number_of_measurements)
        for i, shell_index in enumerate(acquisition_scheme.unique_dwi_indices):
            shell_mask = acquisition_scheme.shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            sh_order = int(acquisition_scheme.shell_sh_orders[shell_index])
            E_dispersed_sh = sh_convolution(
                sh_distribution, rh_models[i, :sh_order // 2 + 1])
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

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
                parameters[self.target_parameter] = radius * 2
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


class BundleModel(DistributedModel, AnisotropicSignalModelProperties):
    """
    The DistributedModel instantiation for a simple, non-distributed bundle
    model. This allows to join models in a secondary layer that allows for
    tortuosity constraints.

    Parameters
    ----------
    models: list of length 1 or more,
        list of models to be Watson-dispersed.
    parameters_links: list of length 1 or more,
        deprecated for testing use only.
    """
    _model_type = 'BundleModel'

    def __init__(self, models, parameter_links=None):
        self.models = models
        self._set_required_acquisition_parameters()
        self._check_for_double_model_class_instances()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        _models_and_distribution = list(self.models)
        self._prepare_parameters(_models_and_distribution)
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        self.mu_params = []
        for param in self.parameter_names:
            if param.endswith('mu'):
                self.mu_params.append(param)

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        rh_scheme.rotational_harmonics_scheme = rh_scheme
        for param in self.mu_params:
            kwargs.update({param: [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        rh_array = np.zeros((len(E_reshaped), max_sh_order // 2 + 1))

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[sh_order],
                    E_reshaped[i])
            )
        return rh_array


class SD1WatsonDistributed(DistributedModel, AnisotropicSignalModelProperties):
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
        self._set_required_acquisition_parameters()
        self._check_for_double_model_class_instances()
        self._check_for_dispersable_models()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []
        self.distribution = distributions.SD1Watson()

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_orientation_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        for param in self.parameter_names:
            if param.endswith('mu'):
                self.mu_param = param

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        rh_scheme.rotational_harmonics_scheme = rh_scheme
        kwargs.update({self.mu_param: [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        rh_array = np.zeros((len(E_reshaped), max_sh_order // 2 + 1))

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[sh_order],
                    E_reshaped[i])
            )
        return rh_array


class SD2BinghamDistributed(
        DistributedModel, AnisotropicSignalModelProperties):
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
        self._set_required_acquisition_parameters()
        self._check_for_double_model_class_instances()
        self._check_for_dispersable_models()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []
        self.distribution = distributions.SD2Bingham()

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_orientation_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        for param in self.parameter_names:
            if param.endswith('mu'):
                self.mu_param = param


class DD1GammaDistributed(DistributedModel, AnisotropicSignalModelProperties):
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
        self._set_required_acquisition_parameters()
        self.target_parameter = target_parameter
        self._check_for_double_model_class_instances()
        self._check_for_distributable_models()
        self._check_for_same_parameter_type()

        self.parameter_links = parameter_links
        if parameter_links is None:
            self.parameter_links = []

        self.normalization = models[0].parameter_types[target_parameter]
        self.distribution = distributions.DD1Gamma(
            normalization=self.normalization)

        _models_and_distribution = list(self.models)
        _models_and_distribution.append(self.distribution)
        self._prepare_parameters(_models_and_distribution)
        self._delete_target_parameter_from_parameters()
        self._prepare_partial_volumes()
        self._prepare_parameter_links()
        for param in self.parameter_names:
            if param.endswith('mu'):
                self.mu_param = param

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({self.mu_param: [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        rh_array = np.zeros((len(E_reshaped), max_sh_order // 2 + 1))

        for i, (shell_index, sh_order) in enumerate(
                rh_scheme.shell_sh_orders.items()):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[sh_order],
                    E_reshaped[i])
            )
        return rh_array

    def set_diameter_constrained_parameter_beta(
            self, diameter_min, diameter_max):
        # append new parameter to parameters
        for parameter in self.parameter_names:
            if parameter.endswith('beta'):
                beta_param = parameter
            if parameter.endswith('alpha'):
                alpha_param = parameter
        new_parameter_name = beta_param + '_fraction'
        parameters = [alpha_param, new_parameter_name]
        self.parameter_ranges.update({new_parameter_name: [0., 1.]})
        self.parameter_scales.update({new_parameter_name: 1.})
        self.parameter_cardinality.update({new_parameter_name: 1})
        self.parameter_types.update({new_parameter_name: 'normal'})

        self._parameter_map.update({new_parameter_name: (None, 'fraction')})
        self._inverted_parameter_map.update(
            {(None, 'fraction'): new_parameter_name})

        # add parmeter link to fractional parameter
        opt_function = ReturnConstrainedBeta(diameter_min, diameter_max)
        model, name = self._parameter_map[beta_param]
        self.parameter_links.append([model, name, opt_function,
                                     [self._parameter_map[param]
                                      for param in parameters]])
        del self.parameter_ranges[beta_param]
        del self.parameter_scales[beta_param]
        del self.parameter_cardinality[beta_param]
        del self.parameter_types[beta_param]


class ReturnConstrainedBeta:
    "Optimization parameter class to constrain gamma distribution mean."

    def __init__(self, diameter_min, diameter_max):
        self.diameter_min = diameter_min
        self.diameter_max = diameter_max
        self.diameter_range = diameter_max - diameter_min

    def __call__(self, alpha, beta_fraction):
        "Diameter = 2 * alpha * beta"
        beta_max = self.diameter_max / (2.0 * alpha)
        beta_min = self.diameter_min / (2.0 * alpha)
        beta_range = beta_max - beta_min
        beta = beta_min + beta_fraction * beta_range
        return beta


class ReturnFixedValue:
    "Parameter fixing class for parameter links."

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value
