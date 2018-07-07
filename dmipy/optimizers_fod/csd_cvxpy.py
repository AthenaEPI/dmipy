import numpy as np
from .construct_observation_matrix import construct_model_based_A_matrix
from dipy.data import get_sphere, HemiSphere
from dipy.reconst.shm import real_sym_sh_mrtrix
from dipy.utils.optpkg import optional_package
from dipy.reconst.shm import sph_harm_ind_list
cvxpy, have_cvxpy, _ = optional_package("cvxpy")
sphere = get_sphere('symmetric724')


__all__ = [
    'CsdCvxpyOptimizer'
]


class CsdCvxpyOptimizer:
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
    lambda_lb: positive float,
        Laplace-Belrami regularization weight to impose smoothness in the
        FOD. Same as is done in [3]_.

    References
    ----------
    .. [1] Jeurissen, Ben, et al. "Multi-tissue constrained spherical
        deconvolution for improved analysis of multi-shell diffusion MRI
        data." NeuroImage 103 (2014): 411-426.
    .. [2] Diamond, Steven, and Stephen Boyd. "CVXPY: A Python-embedded
        modeling language for convex optimization." The Journal of Machine
        Learning Research 17.1 (2016): 2909-2913.
    .. [3] Descoteaux, Maxime, et al. "Regularized, fast, and robust analytical
        Q-ball imaging." Magnetic Resonance in Medicine: An Official Journal of
        the International Society for Magnetic Resonance in Medicine 58.3
        (2007): 497-510.
    """

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 unity_constraint=True, lambda_lb=0.):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = sh_order
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Nmodels = len(self.model.models)
        self.lambda_lb = lambda_lb
        self.unity_constraint = unity_constraint
        self.sphere_jacobian = 2 * np.sqrt(np.pi)

        sphere = get_sphere('symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        self.L_positivity = real_sym_sh_mrtrix(
            self.sh_order, hemisphere.theta, hemisphere.phi)[0]

        x0_single_voxel = np.reshape(
            x0_vector, (-1, x0_vector.shape[-1]))[0]
        if np.all(np.isnan(x0_single_voxel)):
            self.single_convolution_kernel = True
            self.A = self._construct_convolution_kernel(
                x0_single_voxel)
        else:
            self.single_convolution_kernel = False

        self.Ncoef_total = 0
        vf_array = []

        if self.model.volume_fractions_fixed:
            self.sh_start = 0
            self.Ncoef_total = self.Ncoef
            self.vf_indices = np.array([0])
        else:
            for model in self.model.models:
                if ('orientation' in model.parameter_types.values() or
                        model._model_type is "AnisotropicTissueResponse"):
                    self.sh_start = self.Ncoef_total
                    sh_model = np.zeros(self.Ncoef)
                    sh_model[0] = 1
                    vf_array.append(sh_model)
                    self.Ncoef_total += self.Ncoef
                else:
                    vf_array.append(1)
                    self.Ncoef_total += 1
            self.vf_indices = np.where(np.hstack(vf_array))[0]

        sh_l = sph_harm_ind_list(sh_order)[1]
        lb_weights = sh_l ** 2 * (sh_l + 1) ** 2  # laplace-beltrami
        if self.model.volume_fractions_fixed:
            self.R_smoothness = np.diag(lb_weights)
        else:
            diagonal = np.zeros(self.Ncoef_total)
            diagonal[self.sh_start: self.sh_start + self.Ncoef] = lb_weights
            self.R_smoothness = np.diag(diagonal)

    def __call__(self, data, x0_vector):
        """
        The fitting function of Multi-Compartment CSD optimizer.

        If there is only one convolution kernel, it loads the precalculated
        values. If the kernel is voxel-varying, it calculates it now.

        Adds a constraint that the volume fractions add up to one depending on
        the unity_constraint being True or False.

        Adds Laplace-Beltrami (smoothness) regularization if lambda_lb>0.

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
            A = self._construct_convolution_kernel(x0_vector)

        sh_coef = cvxpy.Variable(self.Ncoef_total)
        sh_fod = sh_coef[self.sh_start: self.Ncoef + self.sh_start]

        constraints = []
        constraints.append(
            self.L_positivity * sh_fod >= 0)
        vf = sh_coef[self.vf_indices] * self.sphere_jacobian
        constraints.append(vf >= 0)
        if self.unity_constraint:
            constraints.append(cvxpy.sum(vf) == 1.)

        cost = cvxpy.sum_squares(A * sh_coef - data)
        if self.lambda_lb > 0:
            cost += (
                self.lambda_lb * cvxpy.quad_form(sh_coef, self.R_smoothness))
        problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        problem.solve()

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

    def _construct_convolution_kernel(self, x0_vector):
        """
        Helper function that constructs the convolution kernel for the given
        multi-compartment model and the initial condition x0_vector.

        First the parameter vector is converted to a dictionary with the
        corresponding parameter names. Then, the linked parameters are added to
        the given ones. Finally, the rotational harmonics of the model is
        passed to the construct_model_based_A_matrix, which constructs the
        kernel for an arbitrary PGSE-acquisition scheme.

        For multiple models with fixed volume fractions, the A-matrices
        are combined to have a combined convolution kernel.

        For multiple models without fixed volume fractions, the convolution
        kernels for anisotropic and isotropic models are concatenated, with
        the isotropic kernels always having a spherical harmonics order of 0.

        Parameters
        ----------
        x0_vector: array of size (N_parameters),
            Contains the fixed parameters of the convolution kernel.

        Returns
        -------
        kernel: array of size (N_coef, N_data),
            Observation matrix that maps the FOD spherical harmonics
            coefficients to the DWI signal values.
        """
        parameters_dict = self.model.parameter_vector_to_parameters(
            x0_vector)
        parameters_dict = self.model.add_linked_parameters_to_parameters(
            parameters_dict)

        if self.model.volume_fractions_fixed:
            if len(self.model.models) > 1:
                partial_volumes = [
                    parameters_dict[p] for p in self.model.partial_volume_names
                ]
            else:
                partial_volumes = [1.]
            kernel = 0.
            for model, partial_volume in zip(self.model.models,
                                             partial_volumes):
                parameters = {}
                for parameter in model.parameter_ranges:
                    parameter_name = self.model._inverted_parameter_map[
                        (model, parameter)
                    ]
                    parameters[parameter] = parameters_dict.get(
                        parameter_name
                    )
                model_rh = (
                    model.rotational_harmonics_representation(
                        self.acquisition_scheme, **parameters))
                kernel += partial_volume * construct_model_based_A_matrix(
                    self.acquisition_scheme, model_rh, self.sh_order)
        else:
            kernel = []
            for model in self.model.models:
                parameters = {}
                for parameter in model.parameter_ranges:
                    parameter_name = self.model._inverted_parameter_map[
                        (model, parameter)
                    ]
                    parameters[parameter] = parameters_dict.get(
                        parameter_name
                    )
                model_rh = (
                    model.rotational_harmonics_representation(
                        self.acquisition_scheme, **parameters))
                if 'orientation' in model.parameter_types.values():
                    kernel.append(construct_model_based_A_matrix(
                        self.acquisition_scheme, model_rh, self.sh_order))
                else:
                    kernel.append(construct_model_based_A_matrix(
                        self.acquisition_scheme, model_rh, 0))
            kernel = np.hstack(kernel)
        return kernel
