import numpy as np
from dipy.data import get_sphere, HemiSphere
from dipy.reconst.shm import real_sym_sh_mrtrix
from dipy.utils.optpkg import optional_package
from dipy.reconst.shm import sph_harm_ind_list
cvxpy, have_cvxpy, _ = optional_package("cvxpy")
sphere = get_sphere('symmetric724')


__all__ = [
    'CsdCvxpyOptimizer'
]


class CsdPlusOptimizer:
    """
    Generalized multi-compartment constrained spherical deconvolution (MC-CSD)
    optimizer. The algorithm follows the formulation of Multi-Tissue (MT)-CSD
    [1]_, but is completely generalized in that it can take any number or type
    of convolution kernels, have static or voxel-varying kernels parameters,
    and can have fixed volume fractions or estimate them. The algorithm is
    implemented using CVXPY [2]_.

    Limitations:
    - It cannot estimate the volume fractions of multiple kernels that each
      have an orientation. E.g. it is possible to have a cylinder and a ball
      kernel as input, but not two cylinders.
    - Currently, the different compartment kernels are directly fitted to the
      signal attenuation, implicitly assuming each compartment has the same
      b0-intensity. True MT-CSD requires that these B0-intensities are also
      included.

    Parameters
    ----------
    acquisition_scheme: DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using Dmipy.
    model: Dmipy MultiCompartmentSphericalHarmonicsModel instance,
        Contains Dmipy multi-compartment model information.
    x0_vector: Array of size (Nparameters)
        Possible parameters for model kernels.
    sh_order: positive even integer,
        Spherical harmonics order for deconvolution.
    unity_constraint: boolean,
        whether or not to constrain the volume fractions of the FOD to
        unity. In the case of one model this means that the SH-coefficients
        represent a distribution on the sphere.

    References
    ----------
    .. [1] Haije, Tom Dela, Evren Özarslan, and Aasa Feragen. "Enforcing
        necessary non-negativity constraints for common diffusion MRI models
        using sum of squares programming." NeuroImage 209 (2020): 116405.
    """

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 unity_constraint=True):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = sh_order
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Nmodels = len(self.model.models)
        self.unity_constraint = unity_constraint
        self.sphere_jacobian = 2 * np.sqrt(np.pi)

        x0_single_voxel = np.reshape(
            x0_vector, (-1, x0_vector.shape[-1]))[0]
        if np.all(np.isnan(x0_single_voxel)):
            self.single_convolution_kernel = True
            parameters_dict = self.model.parameter_vector_to_parameters(
                x0_single_voxel)
            self.A = self.model._construct_convolution_kernel(
                **parameters_dict)
        else:
            self.single_convolution_kernel = False

        # load the sh SOS constraints for the given spherical harmonics order.
        ########### TO BE UPDATED
                conf = 'sh_constraint_' + str(sh_order) + '.csv'
        arr = np.loadtxt(conf, delimiter=",")
        pos = arr[:, :3].astype(int)
        val = arr[:, 3]
        dim = list(map(max, zip(*(pos + 1))))
        self.sdp_constraints = np.zeros(dim)
        for i in range(arr.shape[0]):
            self.sdp_constraints[tuple(pos[i])] = val[i]
        ###########################

        self.Ncoef_total = 0
        vf_array = []

        if self.model.volume_fractions_fixed:
            self.sh_start = 0
            self.Ncoef_total = self.Ncoef
            self.vf_indices = np.array([0])
        else:
            for model in self.model.models:
                if 'orientation' in model.parameter_types.values():
                    self.sh_start = self.Ncoef_total
                    sh_model = np.zeros(self.Ncoef)
                    sh_model[0] = 1
                    vf_array.append(sh_model)
                    self.Ncoef_total += self.Ncoef
                else:
                    vf_array.append(1)
                    self.Ncoef_total += 1
            self.vf_indices = np.where(np.hstack(vf_array))[0]

    def __call__(self, data, x0_vector):
        """
        The fitting function of Multi-Compartment CSD optimizer using
        sum-of-squares constraints.

        If there is only one convolution kernel, it loads the precalculated
        values. If the kernel is voxel-varying, it calculates it now.

        Adds a constraint that the volume fractions add up to one depending on
        the unity_constraint being True or False.

        Parameters
        ----------
        data : Array of size (Ndata)
            The normalized dMRI signal attenuation to be fitted.
        x0_vector : Array of size (Nparameters)
            Possible initial guess vector for parameter initiation.

        Returns
        -------
        fitted_parameter_vector : Array of size (Nparameters),
            The fitted MC model parameters using CSD.
        """

        if self.single_convolution_kernel:
            A = self.A
        else:
            parameters_dict = self.model.parameter_vector_to_parameters(
                x0_vector)
            A = self.model._construct_convolution_kernel(**parameters_dict)

        sh_coef = cvxpy.Variable(self.Ncoef_total)
        sh_fod = sh_coef[self.sh_start: self.Ncoef + self.sh_start]

        constraints = []
        ################## ADD SOS CONSTRAINTS
        A = self.sdp_constraints
        m = M.shape[1]
        n = A.shape[0] - m - 1
        s = cvxpy.Variable(n)
        X = A[0]
        for i in range(m):
            X = X + c[i] * A[i + 1]
        for i in range(n):
            X = X + s[i] * A[m + i + 1]
        constraints.append(X >> 0)
        ##################

        vf = sh_coef[self.vf_indices] * self.sphere_jacobian
        constraints.append(vf >= 0)
        if self.unity_constraint:
            constraints.append(cvxpy.sum(vf) == 1.)

        # fix volume fractions if only some of them are fixed.
        # not if all of them are fixed - in that case the convolution
        # matrix is joined into a single composite response function.
        if not self.model.volume_fractions_fixed:
            params = self.model.parameter_vector_to_parameters(x0_vector)
            params = self.model.add_linked_parameters_to_parameters(params)
            for i, vf_name in enumerate(self.model.partial_volume_names):
                if not self.model.parameter_optimization_flags[vf_name]:
                    constraints.append(vf[i] == params[vf_name])

        cost = cvxpy.sum_squares(A * sh_coef - data)
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        try:
            problem.solve()
        except cvxpy.error.SolverError:
            msg = 'cvxpy solver failed'
            print(msg)
            return np.zeros_like(x0_vector)

        if problem.status in ["infeasible", "unbounded"]:
            msg = 'cvxpy found {} problem'.format(problem.status)
            print(msg)
            return np.zeros_like(x0_vector)

        # return optimized fod sh coefficients
        fitted_params = self.model.parameter_vector_to_parameters(x0_vector)
        fitted_params['sh_coeff'] = np.array(sh_fod.value).squeeze()

        if not self.model.volume_fractions_fixed:  # if vf was estimated
            fractions_array = np.array(
                sh_coef[self.vf_indices].value).squeeze() * 2 * np.sqrt(np.pi)
            for i, name in enumerate(self.model.partial_volume_names):
                fitted_params[name] = fractions_array[i]
        fitted_parameter_vector = self.model.parameters_to_parameter_vector(
            **fitted_params)
        return fitted_parameter_vector
