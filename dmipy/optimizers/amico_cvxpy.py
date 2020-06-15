import numpy as np
import cvxpy


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

    def __init__(self, model, acquisition_scheme, x0_vector=None,
                 lambda_1=None, lambda_2=None):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.x0_vector = x0_vector

        if len(lambda_1) != self.model.N_models or \
                len(lambda_2) != self.model.N_models:
            raise ValueError("Number of regularization weights should"
                             "correspond to the number of compartments!")
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def __call__(self, data, M, grid, idx, x0_th=1.e-4):
        """
        The fitting function of AMICO optimizer.
        Parameters
        ----------
        data : Array of size (Ndata)
            The normalized dMRI signal attenuation to be fitted.
        M: Array of size (Ndata, Nx)
            The observation matrix containing Nx model atoms
        grid: dict
            Dictionary containing tessellation of parameters to be estimated
            for each model within multi-compartment model
        idx: dict
            Dictionary containing indices that correspond to the parameters
            to be estimated for each model within multi-compartment model
        x0_th: float
            Threshold for selecting important atoms after solving NNLS
            with L1 and L2 regularization terms

        Returns
        -------
        fitted_parameter_vector : Array of size (Np),
            Vector containing estimated parameters

        """

        # 1. Contracting matrix M and data to have one b=0 value
        M = np.vstack((np.mean(M[self.acquisition_scheme.b0_mask, :], axis=0),
                      M[~self.acquisition_scheme.b0_mask, :]))
        data = np.append(np.mean(data[self.acquisition_scheme.b0_mask]),
                         data[~self.acquisition_scheme.b0_mask])

        # 2. Selecting important atoms by solving NNLS
        # regularized with L1 and L2 norms
        x0 = cvxpy.Variable(len(self.x0_vector))

        cost = 0.5 * cvxpy.sum_squares(M * x0 - data)
        for m_idx, model_name in enumerate(self.model.model_names):
            cost += self.lambda_1[m_idx] * \
                cvxpy.norm(x0[idx[model_name]], 1)
            cost += 0.5 * self.lambda_2[m_idx] * \
                cvxpy.norm(x0[idx[model_name]], 2) ** 2
        problem = cvxpy.Problem(cvxpy.Minimize(cost), [x0 >= 0])
        problem.solve()
        self.x0_vector = x0.value

        # 3. Computing distribution vector x0_vector by solving NNLS
        x0_idx_i = self.x0_vector > x0_th
        x0_i = cvxpy.Variable(sum(x0_idx_i))
        cost = cvxpy.sum_squares(M[:, x0_idx_i] * x0_i - data)
        problem = cvxpy.Problem(cvxpy.Minimize(cost), [x0_i >= 0])
        problem.solve()
        self.x0_vector[~x0_idx_i] = 0.
        self.x0_vector[x0_idx_i] = x0_i.value
        self.x0_vector /= (np.sum(self.x0_vector) + 1.e-8)

        # 4. Estimating parameters based using estimated distribution
        # vector and tessellation grids
        fitted_parameter_vector = []
        for m_idx, model_name in enumerate(self.model.model_names):
            m = self.model.models[m_idx]
            if 'partial_volume_' + str(m_idx) in grid:
                p_estim = np.sum(self.x0_vector[idx[model_name]])
                fitted_parameter_vector.append(p_estim)
            for p in m.parameter_names:
                if p.endswith('mu'):
                    continue
                if model_name + p in grid:
                    p_estim = \
                        np.sum(grid[model_name + p][idx[model_name]] *
                               self.x0_vector[idx[model_name]]) /\
                        (np.sum(self.x0_vector[idx[model_name]]) + 1.e-8)
                    fitted_parameter_vector.append(p_estim)

        return fitted_parameter_vector
