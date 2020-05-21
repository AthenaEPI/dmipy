import numpy as np
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

    def __init__(self, acquisition_scheme, model, x0_vector=None,
                 lambda_1=None, lambda_2=None, Nt=10):
        self.model = model
        self.acquisition_scheme = acquisition_scheme

        if len(lambda_1) != self.model.N_models or \
                len(lambda_2) != self.model.N_models:
            raise ValueError("Number of regularization weights should"
                             "correspond to the number of compartments!")

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.Nt = Nt

        # Make a list of parameters that are not fixed and that require
        # tessellation of parameter ranges
        dir_params = [p for p in self.model.parameter_names
                      if p.endswith('mu')]
        grid_params =\
            [p for p in self.model.parameter_names
             if not p.endswith('mu') and not p.startswith('partial_volume')]

        # Compute length of the vector x0
        x0_len = 0
        for m_idx in range(self.model.N_models):
            m_atoms = 1
            for p in self.model.models[m_idx].parameter_names:
                if self.model.model_names[m_idx] + p in grid_params:
                    m_atoms *= Nt
            x0_len += m_atoms

        # Creating parameter tessellation grids and corresponding indices
        self.grids, self.idx = {}, {}
        for m_idx in range(self.model.N_models):
            model = self.model.models[m_idx]
            model_name = self.model.model_names[m_idx]

            param_sampling, grid_params_names = [], []
            m_atoms = 1
            for p in model.parameter_names:
                if model_name + p not in grid_params:
                    continue
                grid_params_names.append(model_name + p)
                p_range = self.model.parameter_ranges[model_name + p]
                self.grids[model_name + p] = np.full(x0_len, np.mean(p_range))
                param_sampling.append(np.linspace(p_range[0], p_range[1],
                                                  self.Nt, endpoint=True))
                m_atoms *= self.Nt

            self.idx[model_name] =\
                sum([len(self.idx[k]) for k in self.idx]) + np.arange(m_atoms)
            params_mesh = np.meshgrid(*param_sampling)
            for p_idx, p in enumerate(grid_params_names):
                self.grids[p][self.idx[model_name]] =\
                    np.ravel(params_mesh[p_idx])

            self.grids['partial_volume_' + str(m_idx)] = np.zeros(x0_len)
            self.grids['partial_volume_' +
                       str(m_idx)][self.idx[model_name]] = 1.

        arguments = self.grids.copy()
        arguments[dir_params[0]] = [0, 0]
        self.M = self.model.simulate_signal(acquisition_scheme, arguments)
        self.M = self.M[:, ~acquisition_scheme.b0_mask].T

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
