import numpy as np
from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy")


__all__ = [
    'AmicoCvxpyOptimizer'
]


class AmicoCvxpyOptimizer:
    """
    Accelerated microstructure imaging via convex optimization (AMICO). The
    algorithm implementation is based on the work presented in [1]_, but it is
    generalized in the sense that it can perform estimation of any multi-
    compartment model parameter as well as the parameter distribution. The
    optimizer is implemented using CVXPY library [2]_.

    Limitations:
        - ...

    Parameters
    ----------
    acquisition_scheme: DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using Dmipy.
    model: Dmipy MultiCompartmentModel instance,
        Contains Dmipy multi-compartment model information.
    x0_vector: Array of size (Nx),
        Vector containing probability distributions of the parameters that are
        being estimated
    lambda_1: Array of size (Nc)
        Vector containing L2 regularization weights for each compartment
    lambda_2: Array of size (Nc)
        Vector containing L1 regularization weights for each compartment
    Nt: int
        Integer equal to the number of equidistant sampling points over each
        parameter range used to create atoms of observation matrix M

    References
    ----------
    .. [1] Daducci, Alessandro, et al. "Accelerated microstructure imaging
        via convex optimization (AMICO) from diffusion MRI data."
        NeuroImage 105 (2015): 32-44.

    .. [2] Diamond, Steven, and Stephen Boyd. "CVXPY: A Python-embedded
        modeling language for convex optimization." The Journal of Machine
        Learning Research 17.1 (2016): 2909-2913.
    """

    def __init__(self, acquisition_scheme, model,
                 lambda_1=None, lambda_2=None):
        self.model = model
        self.acquisition_scheme = acquisition_scheme

        if len(lambda_1) != self.model.N_models or \
                len(lambda_2) != self.model.N_models:
            raise ValueError("Number of regularization weights should"
                             "correspond to the number of compartments!")
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def __call__(self, data):
        """
        The fitting function of AMICO optimizer.
        Parameters
        ----------
        data : Array of size (Ndata)
            The normalized dMRI signal attenuation to be fitted.

        Returns
        -------
        fitted_parameter_vector : Array of size (Nx),
            Vector containing probability distributions of the parameters that
            are being estimated
        """

        x0 = cvxpy.Variable(len(self.x0_vector))

        cost = 0.5 * cvxpy.sum_squares(self.M * x0 -
                                       data[~self.acquisition_scheme.b0_mask])
        for m_idx, model_name in enumerate(self.model.model_names):
            cost += self.lambda_1[m_idx] *\
                cvxpy.norm(x0[self.idx[model_name]], 1)
            cost += 0.5 * self.lambda_2[m_idx] *\
                cvxpy.norm(x0[self.idx[model_name]], 2) ** 2

        problem = cvxpy.Problem(cvxpy.Minimize(cost), [x0 >= 0])
        problem.solve()

        # TODO:
        # M-pruning
        # estimate x0 vector with non negative least squares

        self.x0_vector = x0.value

        return self.x0_vector
