import numpy as np
from scipy.optimize import differential_evolution, minimize, fmin_cobyla

__all__ = [
    'MixOptimizer',
    'nested_to_normalized_fractions',
    'cobyla_positivity_constraint',
    'cobyla_unity_constraint'
]


class MixOptimizer:
    """ The stochastic Microstructure In Crossings (MIX) optimizer [1] uses a
    three-step process to fit the parameters of a multi-compartment (MC) model
    to data. The key innovation is that they separate linear from non-linear
    parameters in the fitting process, meaning the linear volume fractions
    and non-linear other ones(e.g. diffusivities) are optimized at different
    stages in the process.

    In the first step [1] describes using a genetic algorithm to estimate the
    non-linear parameters of an MC model. For this we use scipy's
    differential_evolution algorithm.

    In the second step [1] describes using CVX to estimate the linear volume
    fractions of an MC model. For this we use scipy's COBYLA algorithm since
    it allows us to impose the parameter constraints we need for volume
    fractions; namely that they are positive and sum up to one.

    The third and last step in [1] is a refining step to find a local minimum
    given the solutions of step one and two. For this we use scipy's
    gradient-based L-BFGS-B algorithm with nested volume fractions.

    The final result takes a model's parameter_ranges into account, only
    yielding parameters within their allowed optimization domain.

    Parameters
    ----------
    model : MultiCompartmentModel instance,
        A multicompartment model that has been instantiated using dMipy.
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    maxiter : integer
        The maximum allowed iterations for the differential evolution algorithm

    References
    ----------
    .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
           White Matter Fibers from diffusion MRI." Nature Scientific
           reports 6 (2016).

    """

    def __init__(
            self, model, acquisition_scheme, maxiter=150, signal_based=False):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.maxiter = maxiter
        self.Nmodels = len(self.model.models)
        self.signal_based = signal_based

    def __call__(self, data, x0_vector=np.array([np.nan]), S0=1.):
        """ The fitting function of the MIX algorithm. Fits the data in three
        distinct steps, first fitting non-linear parameters using
        differential_evolution, then linear parameters using COBYLA, and
        finally refining using L-BFGS-B. See main documentation for details.

        Parameters
        ----------
        data : Array of size (Ndata)
            The normalized dMRI signal attenuation to be fitted.
        x0_vector : Array of size (Nparameters)
            Possible initial guess vector for parameter initiation.

        Returns
        -------
        fitted_parameters : Array of size (Nparameters),
            The fitted MC model parameters using MIX.

        """
        # if there is only one model then MIX only uses DE.
        bounds = list(self.model.bounds_for_optimization)
        if self.Nmodels == 1:
            bounds_de = bounds
            opt_flag_array = self.model.opt_params_for_optimization
            if x0_vector is not None:
                if not np.all(np.isnan(x0_vector)):
                    bounds_de = []
                    for i, x0_ in enumerate(x0_vector):
                        if np.isnan(x0_):
                            bounds_de.append(bounds[i])

            # step 1: stochastic optimization on non-linear parameters.
            res_de = differential_evolution(
                self.stochastic_objective_function,
                bounds=bounds_de,
                maxiter=self.maxiter,
                args=(data, self.acquisition_scheme, x0_vector, S0),
                polish=True).x
            if np.all(np.isnan(x0_vector)):
                fitted_parameters = res_de
                return fitted_parameters
            else:
                x0_bool_array = ~np.isnan(x0_vector)
                fitted_parameters = np.ones(len(x0_bool_array))
                fitted_parameters[~x0_bool_array] = res_de
                fitted_parameters[x0_bool_array] = x0_vector[
                    x0_bool_array]
                return fitted_parameters
        # if there is more than 1 model then we do the 3 steps.
        if self.Nmodels > 1:
            bounds_de = bounds
            bounds_minimize = bounds[:-1]  # nested volume fractions
            opt_flag_array = self.model.opt_params_for_optimization
            if x0_vector is not None:
                if not np.all(np.isnan(x0_vector)):
                    bounds_de = []
                    bounds_minimize = []
                    for i, x0_ in enumerate(x0_vector[:-self.Nmodels]):
                        if np.isnan(x0_):
                            bounds_de.append(bounds[i])
                    for i, x0_ in enumerate(x0_vector):
                        if opt_flag_array[i] is True:
                            bounds_minimize.append(bounds[i])
                    bounds_minimize = bounds_minimize[:-1]
                else:
                    bounds_de = bounds_de[:-self.Nmodels]
            # step 1: stochastic optimization on non-linear parameters.
            res_de = differential_evolution(self.stochastic_objective_function,
                                            bounds=bounds_de,
                                            maxiter=self.maxiter,
                                            args=(data,
                                                  self.acquisition_scheme,
                                                  x0_vector,
                                                  S0))
            optimized_parameter_vector = res_de.x
            if np.all(np.isnan(x0_vector)):
                parameter_vector = np.r_[optimized_parameter_vector,
                                         np.ones(self.Nmodels)]
            elif np.all(~np.isnan(x0_vector[-self.Nmodels:])):
                x0_bool_array = ~np.isnan(x0_vector)
                parameter_vector = np.ones(len(x0_bool_array))
                parameter_vector[~x0_bool_array] = optimized_parameter_vector
                parameter_vector[x0_bool_array] = x0_vector[x0_bool_array]
                return parameter_vector
            else:
                x0_bool_array = ~np.isnan(x0_vector)
                parameter_vector = np.ones(len(x0_bool_array))
                x0_bool_n0_vf = x0_bool_array[:-self.Nmodels]
                parameter_vector_no_vf = np.empty(
                    len(x0_bool_n0_vf), dtype=float)
                parameter_vector_no_vf[~x0_bool_n0_vf] = (
                    optimized_parameter_vector)
                parameter_vector_no_vf[x0_bool_n0_vf] = x0_vector[
                    x0_bool_array]
                parameter_vector[:-self.Nmodels] = parameter_vector_no_vf
            parameters = self.model.parameter_vector_to_parameters(
                parameter_vector * self.model.scales_for_optimization)

            # step 2: Estimating linear variables using COBYLA
            phi = self.model(self.acquisition_scheme,
                             quantity="stochastic cost function", **parameters)
            if self.signal_based:
                phi /= S0
            try:
                phi_inv = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
                vf_x0 = np.dot(phi_inv, data)
                vf_x0 /= np.sum(np.clip(vf_x0, 0, np.inf))
                vf = fmin_cobyla(self.cobyla_cost_function, x0=vf_x0,
                                 cons=[cobyla_positivity_constraint,
                                       cobyla_unity_constraint],
                                 args=(phi, data),
                                 maxfun=2000)
            except np.linalg.linalg.LinAlgError:
                # happens when models have the same signal attenuations.
                vf = np.ones(self.Nmodels) / float(self.Nmodels)
            vf_nested = np.ones(len(vf) - 1)
            vf_nested[0] = vf[0]
            for i in np.arange(1, len(vf_nested)):
                vf_nested[i] = vf[i] / vf[i - 1]

            # Convert to nested volume fractions
            x0_refine = np.r_[optimized_parameter_vector, vf_nested]
            # step 3: refine using gradient method
            x_fine_nested = minimize(self.objective_function, x0_refine,
                                     (data,
                                      self.acquisition_scheme,
                                      x0_vector, S0),
                                     bounds=bounds_minimize).x
            nested_fractions = x_fine_nested[-(self.Nmodels - 1):]
            normalized_fractions = nested_to_normalized_fractions(
                nested_fractions)
            minimize_fitted_parameters = np.r_[
                x_fine_nested[:-(self.Nmodels - 1)], normalized_fractions]

            if not np.all(self.model.opt_params_for_optimization):
                x0_bool_array = ~np.isnan(x0_vector)
                fitted_parameters = np.ones(len(x0_bool_array))
                fitted_parameters[~x0_bool_array] = minimize_fitted_parameters
                fitted_parameters[x0_bool_array] = x0_vector[x0_bool_array]
            else:
                fitted_parameters = minimize_fitted_parameters
            return fitted_parameters

    def stochastic_objective_function(self, optimized_parameter_vector,
                                      data, acquisition_scheme, x0_params, S0):
        """Objective function for stochastic non-linear parameter estimation
        using differential_evolution
        """
        x0_bool_array = ~np.isnan(x0_params)
        if self.Nmodels == 1:
            # add fixed parameters if given.
            if np.all(np.isnan(x0_params)):
                parameter_vector = optimized_parameter_vector
            else:
                parameter_vector = np.empty(len(x0_bool_array))
                parameter_vector[~x0_bool_array] = optimized_parameter_vector
                parameter_vector[x0_bool_array] = x0_params[x0_bool_array]

            parameter_vector = (
                parameter_vector * self.model.scales_for_optimization)
            parameters = self.model.parameter_vector_to_parameters(
                parameter_vector)
            E_hat = self.model(acquisition_scheme, **parameters)
        elif self.Nmodels > 1:
            if np.all(np.isnan(x0_params)):
                parameter_vector = np.r_[optimized_parameter_vector,
                                         np.ones(self.Nmodels)]
            else:
                parameter_vector = np.ones(len(x0_bool_array))
                x0_bool_n0_vf = x0_bool_array[:-self.Nmodels]
                parameter_vector_no_vf = np.empty(
                    len(x0_bool_n0_vf), dtype=float)
                parameter_vector_no_vf[~x0_bool_n0_vf] = (
                    optimized_parameter_vector)
                parameter_vector_no_vf[x0_bool_n0_vf] = x0_params[
                    :-self.Nmodels][x0_bool_n0_vf]
                parameter_vector[:-self.Nmodels] = parameter_vector_no_vf
            parameter_vector = (
                parameter_vector * self.model.scales_for_optimization)
            parameters = self.model.parameter_vector_to_parameters(
                parameter_vector)
            phi_x = self.model(acquisition_scheme,
                               quantity="stochastic cost function",
                               **parameters)
            if np.all(~np.isnan(x0_params[-self.Nmodels:])):
                # if initial guess is given for volume fractions
                vf = x0_params[-self.Nmodels:]
            else:
                A = np.dot(phi_x.T, phi_x)
                try:
                    phi_inv = np.dot(np.linalg.inv(A), phi_x.T)
                    vf = np.dot(phi_inv, data)
                except np.linalg.linalg.LinAlgError:
                    # happens when models have the same signal attenuations.
                    vf = np.ones(self.Nmodels) / float(self.Nmodels)
            E_hat = np.dot(phi_x, vf)
        if self.signal_based:
            E_hat /= S0
        objective = np.dot(data - E_hat, data - E_hat).squeeze()
        return objective * 1e5

    def cobyla_cost_function(self, fractions, phi, data):
        "Objective function of linear parameter estimation using COBYLA."
        E_hat = np.dot(phi, fractions)
        diff = data - E_hat
        objective = np.dot(diff, diff)
        return objective * 1e5

    def objective_function(
            self, optimized_parameter_vector, data, acquisition_scheme,
            x0_params, S0):
        "Objective function of final refining step using L-BFGS-B"
        nested_fractions = optimized_parameter_vector[-(self.Nmodels - 1):]
        normalized_fractions = nested_to_normalized_fractions(
            nested_fractions)
        optimized_parameter_vector = np.r_[
            optimized_parameter_vector[:-(self.Nmodels - 1)],
            normalized_fractions]
        if not np.all(self.model.opt_params_for_optimization):
            x0_bool_array = ~np.isnan(x0_params)
            parameter_vector = np.ones(len(x0_bool_array))
            parameter_vector[~x0_bool_array] = (
                optimized_parameter_vector)
            parameter_vector[x0_bool_array] = x0_params[x0_bool_array]
        else:
            parameter_vector = optimized_parameter_vector

        parameter_vector = (
            parameter_vector * self.model.scales_for_optimization)
        parameters = {}
        parameters.update(
            self.model.parameter_vector_to_parameters(parameter_vector)
        )
        E_model = self.model(acquisition_scheme, **parameters)
        if self.signal_based:
            E_model /= S0
        E_diff = E_model - data
        objective = np.dot(E_diff, E_diff) / len(data)
        return objective * 1e5


def nested_to_normalized_fractions(nested_fractions):
    "Function to convert nested to normalized volume fractions."
    N = len(nested_fractions)
    normalized_fractions = np.zeros(N + 1)
    remaining_fraction = 1.
    for i in range(N):
        normalized_fractions[i] = remaining_fraction * nested_fractions[i]
        remaining_fraction -= normalized_fractions[i]
    normalized_fractions[-1] = remaining_fraction
    return normalized_fractions


def cobyla_positivity_constraint(volume_fractions, *args):
    "COBYLA positivity constraint on volume fractions"
    return volume_fractions - 0.001


def cobyla_unity_constraint(volume_fractions, *args):
    "COBYLA unity constraint on volume fractions"
    return np.sum(volume_fractions) - 1
