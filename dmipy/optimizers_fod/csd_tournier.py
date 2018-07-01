import numpy as np
from .construct_observation_matrix import construct_model_based_A_matrix
from dipy.data import get_sphere, HemiSphere
from dipy.reconst.shm import real_sym_sh_mrtrix
from dipy.utils.optpkg import optional_package
from dipy.reconst.shm import sph_harm_ind_list
sphere = get_sphere('symmetric724')
numba, have_numba, _ = optional_package("numba")


__all__ = [
    'CsdTournierOptimizer'
]


class CsdTournierOptimizer:
    """
    The classical Constrained Spherical Deconvolution (CSD) optimizer as
    proposed by Tournier et al. (2007) [1]_.

    It is a bit less accurate than the general CVXPY optimizer, but
    significantly faster.

    TODO: multicore processing makes this solver MUCH slower for some
    reason. It is likely the fact that it's copying the kernel and
    positivity matrices to every process.

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
    lambda_pos: positive float,
        Positivity regularization parameter.
    lambda_lb: positive float,
        Laplace-Belrami regularization weight to impose smoothness in the
        FOD. Same as is done in [2]_.
    tau: positive float,
        Scales positivity threshold relative to maximum FOD amplitude.
    max_iter: positive integer,
        Maximum number of iterations for optimization.
    unity_constraint: boolean,
        whether or not to constrain the volume fractions of the FOD to
        unity. In the case of one model this means that the SH-coefficients
        represent a distribution on the sphere.
    init_sh_order: positive even integer,
        Spherical harmonics order used to calculate unconstrained initial
        guess for the FOD coefficients.

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Robust determination of the fibre orientation distribution in
        diffusion MRI: non-negativity constrained super-resolved spherical
        deconvolution." Neuroimage 35.4 (2007): 1459-1472.
    .. [2] Descoteaux, Maxime, et al. "Regularized, fast, and robust analytical
        Q‐ball imaging." Magnetic Resonance in Medicine: An Official Journal of
        the International Society for Magnetic Resonance in Medicine 58.3
        (2007): 497-510.
    """

    def __init__(self, acquisition_scheme, model, x0_vector=None, sh_order=8,
                 lambda_pos=1., lambda_lb=5e-4, tau=0.1, max_iter=50,
                 unity_constraint=True, init_sh_order=4):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = sh_order
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Ncoef4 = int((init_sh_order + 2) * (init_sh_order + 1) // 2)
        self.Nmodels = len(self.model.models)
        self.lambda_pos = lambda_pos
        self.lambda_lb = lambda_lb
        self.tau = tau
        self.max_iter = max_iter
        self.unity_constraint = unity_constraint
        self.sphere_jacobian = 1 / (2 * np.sqrt(np.pi))

        # step 1: prepare positivity grid on sphere
        sphere = get_sphere('symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        self.L_positivity = real_sym_sh_mrtrix(
            self.sh_order, hemisphere.theta, hemisphere.phi)[0]

        sh_l = sph_harm_ind_list(sh_order)[1]
        self.R_smoothness = np.diag(sh_l ** 2 * (sh_l + 1) ** 2)

        # check if there is only one model. If so, precompute rh array.
        if self.model.volume_fractions_fixed:
            x0_single_voxel = np.reshape(
                x0_vector, (-1, x0_vector.shape[-1]))[0]
            if np.all(np.isnan(x0_single_voxel)):
                self.single_convolution_kernel = True
                self.A = self._construct_convolution_kernel(
                    x0_single_voxel)
                self.AT_A = np.dot(self.A.T, self.A)
            else:
                self.single_convolution_kernel = False
        else:
            msg = "This CSD optimizer cannot estimate volume fractions."
            raise ValueError(msg)

    def __call__(self, data, x0_vector):
        """
        The fitting function of Tournier's CSD optimizer.

        If there is only one convolution kernel, it loads the precalculated
        values. If the kernel is voxel-varying, it calculates it now.

        Calls a different optimizing procedure depending on if
        the unity_constraint is True or False.

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
            AT_A = self.AT_A + self.lambda_lb * self.R_smoothness
        else:
            A = self._construct_convolution_kernel(x0_vector)
            AT_A = np.dot(A.T, A) + self.lambda_lb * self.R_smoothness

        if self.unity_constraint:
            return self._optimize_with_unity_constraint(
                A, AT_A, data, x0_vector)
        else:
            return self._optimize_without_unity_constraint(
                A, AT_A, data, x0_vector)

    def _optimize_without_unity_constraint(self, A, AT_A, data, x0_vector):
        """
        CSD optimizer that does not take into account the unity constraint of
        the FOD distribution.

        Parameters
        ----------
        A: array of size (N_data, N_coef),
            Observation matrix that maps FOD coefficients to DWI intensities.
        AT_A: array of size (N_coef, Ncoef),
            Precomputation of np.dot(A.T, A)
        data: array of size (N_data,)
            Measured DWI data.
        x0_vector: array of size (N_parameters),
            Initial condition for parameters.

        Returns
        -------
        fitted_parameter_vector: array of size (N_parameters),
            Fitted parameters including SH coefficients of FOD.
        """
        f_sh = np.zeros(self.Ncoef, dtype=float)

        # initialize coefficients with initial guess
        f_sh[:self.Ncoef4] = np.dot(np.linalg.pinv(A[:, :self.Ncoef4]), data)

        # minimum threshold of FOD amplitude based on FOD mean (without
        # jacobian) according to dipy.
        threshold = self.tau * self.L_positivity[0, 0] * f_sh[0]
        negative_fod_check = np.dot(self.L_positivity, f_sh) < threshold

        for iteration in range(self.max_iter):
            L = self.L_positivity[negative_fod_check]
            Q = AT_A + self.lambda_pos * np.dot(L.T, L)
            f_sh = np.dot(np.dot(np.linalg.inv(Q), A.T), data)
            negative_fod_check_old = negative_fod_check
            negative_fod_check = np.dot(self.L_positivity, f_sh) < threshold
            if np.array_equal(negative_fod_check, negative_fod_check_old):
                break
        # return optimized fod sh coefficients
        fitted_params = self.model.parameter_vector_to_parameters(x0_vector)
        fitted_params['sh_coeff'] = f_sh
        fitted_parameter_vector = self.model.parameters_to_parameter_vector(
            **fitted_params)
        return fitted_parameter_vector

    def _optimize_with_unity_constraint(self, A, AT_A, data, x0_vector):
        """
        CSD optimizer that constrains the first coefficient to be equal to
        1 / (2 * np.sqrt(np.pi)), in order for the spherical integration of
        the FOD to add up to unity.

        In the current implementation the first coefficient is just set to
        this value and never updated, only allowing other to change in the
        iteration.

        A better approach is to precalculate the remained of the data by
        subtracting the convolution of the first known coefficient and the
        convolution kernel from the data. Then only fitting the remaining
        coefficients to the remaining data should result in a better
        estimate of the coefficients. However, the positivity regularization
        seems to not behave the same way when doing this, which is why this
        is the current solution.

        Parameters
        ----------
        A: array of size (N_data, N_coef),
            Observation matrix that maps FOD coefficients to DWI intensities.
        AT_A: array of size (N_coef, Ncoef),
            Precomputation of np.dot(A.T, A)
        data: array of size (N_data,)
            Measured DWI data.
        x0_vector: array of size (N_parameters),
            Initial condition for parameters.

        Returns
        -------
        fitted_parameter_vector: array of size (N_parameters),
            Fitted parameters including SH coefficients of FOD.
        """
        f_sh = np.zeros(self.Ncoef, dtype=float)
        f_sh[0] = self.sphere_jacobian

        # initialize coefficients with initial guess
        f_sh[1:self.Ncoef4] = np.dot(
            np.linalg.pinv(A[:, :self.Ncoef4]), data)[1:]

        # minimum threshold of FOD amplitude based on FOD mean (without
        # jacobian) according to dipy.
        threshold = self.tau * self.L_positivity[0, 0] * f_sh[0]
        negative_fod_check = np.dot(self.L_positivity, f_sh) < threshold

        for iteration in range(self.max_iter):
            L = self.L_positivity[negative_fod_check]
            Q = AT_A + self.lambda_pos * np.dot(L.T, L)
            f_sh[1:] = np.dot(np.dot(np.linalg.inv(Q), A.T), data)[1:]
            negative_fod_check_old = negative_fod_check
            negative_fod_check = np.dot(self.L_positivity, f_sh) < threshold
            if np.array_equal(negative_fod_check, negative_fod_check_old):
                break
        # return optimized fod sh coefficients
        fitted_params = self.model.parameter_vector_to_parameters(x0_vector)
        fitted_params['sh_coeff'] = f_sh
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

        if len(self.model.models) > 1:
            partial_volumes = [
                parameters_dict[p] for p in self.model.partial_volume_names
            ]
        else:
            partial_volumes = [1.]

        kernel = 0.
        for model, partial_volume in zip(self.model.models, partial_volumes):
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
        return kernel
