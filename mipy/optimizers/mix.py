import numpy as np
from scipy.optimize import differential_evolution, minimize, fmin_cobyla


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
    acquisition_scheme : MipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    maxiter : integer
        The maximum allowed iterations for the differential evolution algorithm

    References
    ----------
    .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
           White Matter Fibers from diffusion MRI." Nature Scientific
           reports 6 (2016).

    """

    def __init__(self, model, acquisition_scheme, maxiter=150):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.maxiter = maxiter
        self.Nmodels = len(self.model.models)

    def __call__(self, data, x0_vector):
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
        bounds = list(self.model.bounds_for_optimization)
        for i, x0_ in enumerate(x0_vector):
            if (x0_ is not None and
                    self.model.opt_params_for_optimization[i] is False):
                bounds[i] = np.r_[x0_, x0_ + 1e-6]

        # step 1: Variable separation using differential evolution algorithm
        bounds_de = list(bounds)
        if self.Nmodels > 1:
            bounds_de = bounds_de[:-self.Nmodels]

        res_de = differential_evolution(self.stochastic_objective_function,
                                        bounds=bounds_de,
                                        maxiter=self.maxiter,
                                        args=(data, self.acquisition_scheme))
        res_de_x = res_de.x
        if self.Nmodels > 1:
            res_de_x = np.r_[res_de_x, np.ones(self.Nmodels)]
        parameters = self.model.parameter_vector_to_parameters(
            res_de_x * self.model.scales_for_optimization)

        # step 2: Estimating linear variables using COBYLA (if there are any)
        if self.Nmodels > 1:
            phi = self.model(self.acquisition_scheme,
                             quantity="stochastic cost function", **parameters)
            phi_inv = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
            vf_x0 = np.dot(phi_inv, data)
            vf_x0 /= np.sum(np.clip(vf_x0, 0, np.inf))
            vf = fmin_cobyla(self.cobyla_cost_function, x0=vf_x0,
                             cons=[cobyla_positivity_constraint,
                                   cobyla_unity_constraint],
                             args=(phi, data),
                             maxfun=2000)

            vf_nested = np.ones(len(vf) - 1)
            vf_nested[0] = vf[0]
            for i in np.arange(1, len(vf_nested)):
                vf_nested[i] = vf[i] / vf[i - 1]

            # Convert to nested volume fractions
            x0_refine = np.r_[res_de_x[:-len(vf)], vf_nested]
            bounds_ = bounds[:-1]
        else:
            x0_refine = res_de_x
            bounds_ = bounds

        # step 3: refine using gradient method
        x_fine_nested = minimize(self.objective_function, x0_refine,
                                 (data, self.acquisition_scheme),
                                 bounds=bounds_).x

        # Convert back to normal volume fractions
        if self.Nmodels > 1:
            nested_fractions = x_fine_nested[-(self.Nmodels - 1):]
            normalized_fractions = nested_to_normalized_fractions(
                nested_fractions)
            fitted_parameters = np.r_[
                x_fine_nested[:-(self.Nmodels - 1)], normalized_fractions]
        else:
            fitted_parameters = x_fine_nested
        return fitted_parameters

    def stochastic_objective_function(self, parameter_vector,
                                      data, acquisition_scheme):
        """Objective function for stochastic non-linear parameter estimation
        using differential_evolution
        """

        if self.Nmodels > 1:
            parameter_vector = np.r_[parameter_vector, np.ones(self.Nmodels)]
        parameter_vector = (
            parameter_vector * self.model.scales_for_optimization)

        parameters = {}
        parameters.update(
            self.model.parameter_vector_to_parameters(parameter_vector)
        )

        phi_x = self.model(acquisition_scheme,
                           quantity="stochastic cost function", **parameters)

        phi_inv = np.dot(np.linalg.inv(np.dot(phi_x.T, phi_x)), phi_x.T)
        vf = np.dot(phi_inv, data)
        E_hat = np.dot(phi_x, vf)
        objective = np.dot(data - E_hat, data - E_hat).squeeze()
        return objective

    def cobyla_cost_function(self, fractions, phi, data):
        "Objective function of linear parameter estimation using COBYLA."
        E_hat = np.dot(phi, fractions)
        diff = data - E_hat
        objective = np.dot(diff, diff)
        return objective

    def objective_function(self, parameter_vector, data, acquisition_scheme):
        "Objective function of final refining step using L-BFGS-B"
        if self.Nmodels > 1:
            nested_fractions = parameter_vector[-(self.Nmodels - 1):]
            normalized_fractions = nested_to_normalized_fractions(
                nested_fractions)
            parameter_vector_ = np.r_[
                parameter_vector[:-(self.Nmodels - 1)], normalized_fractions]
        else:
            parameter_vector_ = parameter_vector
        parameter_vector_ = (
            parameter_vector_ * self.model.scales_for_optimization)
        parameters = {}
        parameters.update(
            self.model.parameter_vector_to_parameters(parameter_vector_)
        )
        E_model = self.model(acquisition_scheme, **parameters)
        E_diff = E_model - data
        objective = np.dot(E_diff, E_diff) / len(data)
        return objective


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
