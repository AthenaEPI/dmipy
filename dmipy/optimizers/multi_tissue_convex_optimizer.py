import numpy as np
from scipy.optimize import fmin_cobyla


class MultiTissueConvexOptimizer:
    """
    Secondary optimizer for including S0 tissue response values into the volume
    fraction estimation.

    Following the suggestion by [1]_, when including S0 responses, the
    volume fractions are no longer unity constrained. This means that the
    optimization of linear volume fractions and non-linear parameters is
    independent, and thus this secondary optimization is just a simple convex
    optimization on the volume fractions only.

    Parameters
    ----------
    model: dmipy multi-compartment model instance,
        dmipy initialized mc model.
    S0_tissue_responses: list,
        constains the positive S0 tissue responses that are associated with the
        tissue that each compartment model in the mc-model represents.

    References
    ----------
    .. [1] Dell'Acqua, Flavio, and J-Donald Tournier. "Modelling white matter
           with spherical deconvolution: How and why?." NMR in Biomedicine 32.4
           (2019): e3945.
    """

    def __init__(self, acquisition_scheme, model, S0_tissue_responses):
        self.acquisition_scheme = acquisition_scheme
        self.model = model
        self.S0_tissue_responses = S0_tissue_responses

    def cobyla_cost_function(self, fractions, phi, data):
        "Objective function of linear parameter estimation using COBYLA."
        E_hat = np.dot(phi, fractions)
        diff = data - E_hat
        objective = np.dot(diff, diff)
        return objective * 1e5

    def __call__(self, data, x0):
        params = x0 * self.model.scales_for_optimization
        params_dict = self.model.parameter_vector_to_parameters(params)
        phi = self.model(self.acquisition_scheme,
                         quantity="stochastic cost function", **params_dict)
        phi *= self.S0_tissue_responses

        if self.model.N_models == 1:
            vf_x0 = [1.]
        else:
            vf_x0 = x0[-self.model.N_models:]

        vf = fmin_cobyla(self.cobyla_cost_function, x0=vf_x0,
                         cons=[cobyla_positivity_constraint],
                         args=(phi, data),
                         maxfun=2000)
        return vf


def cobyla_positivity_constraint(volume_fractions, *args):
    "COBYLA positivity constraint on volume fractions"
    return volume_fractions - 0.001
