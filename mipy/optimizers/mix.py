import numpy as np
from scipy.optimize import differential_evolution, minimize, fmin_cobyla


class MixOptimizer:
    def __init__(self, model, maxiter=150):
        self.model = model
        self.maxiter = maxiter
        self.Nmodels = len(self.model.models)

    def __call__(self, data, x0_vector):
        """
        differential_evolution
        cobyla
        least squares

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Nature Scientific
               reports 6 (2016).
        """
        bounds = list(self.model.bounds_for_optimization)
        for i, x0_ in enumerate(x0_vector):
            if (x0_ is not None and
                    self.model.opt_params_for_optimization[i] is False):
                bounds[i] = np.r_[x0_, x0_ + 1e-6]
        # step 1: Variable separation using genetic algorithm
        bounds_de = list(bounds)
        if self.Nmodels > 1:
            bounds_de = bounds_de[:-self.Nmodels]

        res_one = differential_evolution(self.stochastic_objective_function,
                                         bounds=bounds_de,
                                         maxiter=self.maxiter,
                                         args=(data, self.model.scheme))
        res_one_x = res_one.x
        if self.Nmodels > 1:
            res_one_x = np.r_[res_one_x, np.ones(self.Nmodels)]
        parameters = self.model.parameter_vector_to_parameters(
            res_one_x * self.model.scales_for_optimization)

        # step 2: Estimating linear variables using cvx (if there are any)
        if self.Nmodels > 1:
            phi = self.model(self.model.scheme,
                             quantity="stochastic cost function", **parameters)
            phi_inv = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
            f_x0 = np.dot(phi_inv, data)
            f_x0 /= np.sum(np.clip(f_x0, 0, np.inf))
            vf = fmin_cobyla(self.cobyla_cost_function, x0=f_x0,
                             cons=[cobyla_positivity_constraint,
                                   cobyla_unity_constraint],
                             args=(phi, data),
                             maxfun=2000)

            # step 3: refine using gradient method / convert nested fractions
            vf_nested = np.ones(len(vf) - 1)
            vf_nested[0] = vf[0]
            for i in np.arange(1, len(vf_nested)):
                vf_nested[i] = vf[i] / vf[i - 1]

            x0_refine = np.r_[res_one_x[:-len(vf)], vf_nested]
            bounds_ = bounds[:-1]
        else:
            x0_refine = res_one_x
            bounds_ = bounds

        x_fine_nested = minimize(self.objective_function, x0_refine,
                                 (data, self.model.scheme),
                                 bounds=bounds_).x

        if self.Nmodels > 1:
            nested_fractions = x_fine_nested[-(self.Nmodels - 1):]
            normalized_fractions = nested_to_normalized_fractions(
                nested_fractions)
            x_fine = np.r_[
                x_fine_nested[:-(self.Nmodels - 1)], normalized_fractions]
        else:
            x_fine = x_fine_nested
        return x_fine

    def objective_function(self, parameter_vector, data, acquisition_scheme):
        if self.Nmodels > 1:
            nested_fractions = parameter_vector[-(self.Nmodels - 1):]
            normalized_fractions = nested_to_normalized_fractions(
                nested_fractions)
            parameter_vector_ = np.r_[
                parameter_vector[:-(self.Nmodels - 1)], normalized_fractions]
        else:
            parameter_vector_ = parameter_vector
        parameter_vector_ = parameter_vector_ * self.model.scales_for_optimization
        parameters = {}
        parameters.update(
            self.model.parameter_vector_to_parameters(parameter_vector_)
        )
        E_model = self.model(acquisition_scheme, **parameters)
        E_diff = E_model - data
        objective = np.dot(E_diff, E_diff) / len(data)
        return objective

    def stochastic_objective_function(self, parameter_vector,
                                      data, acquisition_scheme):
        if self.Nmodels > 1:
            parameter_vector = np.r_[parameter_vector, np.ones(self.Nmodels)]
        parameter_vector = parameter_vector * self.model.scales_for_optimization

        parameters = {}
        parameters.update(
            self.model.parameter_vector_to_parameters(parameter_vector)
        )

        phi_x = self.model(acquisition_scheme,
                           quantity="stochastic cost function", **parameters)

        phi_mp = np.dot(np.linalg.inv(np.dot(phi_x.T, phi_x)), phi_x.T)
        f = np.dot(phi_mp, data)
        yhat = np.dot(phi_x, f)
        cost = np.dot(data - yhat, data - yhat).squeeze()
        return cost

    def cobyla_cost_function(self, fractions, phi, data):
        E_hat = np.dot(phi, fractions)
        diff = data - E_hat
        objective = np.dot(diff, diff)
        return objective


def nested_to_normalized_fractions(nested_fractions):
    N = len(nested_fractions)
    normalized_fractions = np.zeros(N + 1)
    remaining_fraction = 1.
    for i in range(N):
        normalized_fractions[i] = remaining_fraction * nested_fractions[i]
        remaining_fraction -= normalized_fractions[i]
    normalized_fractions[-1] = remaining_fraction
    return normalized_fractions


def cobyla_positivity_constraint(volume_fractions, *args):
    return volume_fractions - 0.001


def cobyla_unity_constraint(volume_fractions, *args):
    return np.sum(volume_fractions) - 1
