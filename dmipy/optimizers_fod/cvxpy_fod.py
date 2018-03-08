import numpy as np
import cvxpy
from itertools import chain
from dipy.data import get_sphere, HemiSphere
from dipy.reconst.shm import real_sym_sh_mrtrix


__all__ = [
    'CvxpyOptimizer'
]


class CvxpyOptimizer:

    def __init__(self, acquisition_scheme, model, sh_order,
                 optimize_volume_fractions, unity_constraint=True
                 ):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = sh_order
        self.Ncoefficients = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Nmodels = len(self.model.models)
        if self.Nmodels == 1:
            self.optimize_volume_fractions = False
        else:
            self.optimize_volume_fractions = optimize_volume_fractions
        self.unity_constraint = unity_constraint

        # prepare positivity grid on sphere
        sphere = get_sphere('symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        self.sh_matrix_positivity = real_sym_sh_mrtrix(
            self.sh_order, hemisphere.theta, hemisphere.phi)[0]

    def __call__(self, data, x0_vector):
        # step 1: determine rotational_harmonics of kernel
        if self.optimize_volume_fractions is False:
            rh_matrix = self.recover_rotational_harmonics_fixed_fractions(
                x0_vector)
        else:
            rh_matrix = self.recover_rotational_harmonics_opt_fractions()

        # step 2: set up cvxpy problem
        # need to pull out the sh convolution to work with cvxpy variables.
        c = cvxpy.Variable(self.Ncoefficients)
        mse = 0.

        # if model has mu then it has orientation, use sh_order of input
        # if model is round (no mu) en use only one coefficient.
        # the single coefficient doubles as the volume fraction
        # separate positivity constraint per model
        # multiple models with mu cannot be dealt with...
        for j, model in enumerate(self.model.models):
            for i, shell_index in enumerate(
                    self.acquisition_scheme.unique_dwi_indices):
                shell_mask = (
                    self.acquisition_scheme.shell_indices == shell_index)
                shell_data = data[shell_mask]
                rh_order = int(
                    self.acquisition_scheme.shell_sh_orders[shell_index])
                sh_mat = self.acquisition_scheme.shell_sh_matrices[shell_index]

                rh_shell_prepared = self.prepare_rotational_harmonics(
                    rh_matrix[j, i], rh_order)
                if self.sh_order >= rh_order:
                    sh_shell = (
                        cvxpy.diag(
                            rh_shell_prepared) * c[:len(rh_shell_prepared)])
                    mse += cvxpy.sum_squares(sh_mat * sh_shell - shell_data)
                else:
                    sh_shell = (
                        cvxpy.diag(rh_shell_prepared[:self.Ncoefficients]) * c)
                    mse += cvxpy.sum_squares(
                        sh_mat[:, :self.Ncoefficients] * sh_shell - shell_data)
        objective = cvxpy.Minimize(mse)
        constraints = [self.sh_matrix_positivity * c > 0.]
        # c[0] == 1. / (2 * np.sqrt(np.pi))
        # ]
        # if self.unity_constraint:
        #     constraints.append(c[0] == 1. / (2 * np.sqrt(np.pi)))
        problem = cvxpy.Problem(objective, constraints)
        problem.solve()
        sh_coefficients = np.array(c.value).squeeze()
        return sh_coefficients

    def prepare_rotational_harmonics(self, rh_array, rh_order):
        "Function to extend rotational harmonics and prepare them for MSE."
        rh_coef = np.zeros(int((rh_order + 2) * (rh_order + 1) // 2))
        counter = 0
        for n_ in range(0, rh_order + 1, 2):
            coef_in_order = 2 * n_ + 1
            rh_coef[counter: counter + coef_in_order] = (
                rh_array[n_ // 2] * np.sqrt((4 * np.pi) / (2 * n_ + 1)))
            counter += coef_in_order
        return rh_coef

    def recover_rotational_harmonics_fixed_fractions(self, x0_vector):
        """ Recover rotational harmonics in case there is only one model given
        or self.optimize_volume_fractions is False.
        """
        parameters = self.model.parameter_vector_to_parameters(x0_vector)
        parameters = self.model.add_linked_parameters_to_parameters(parameters)
        rh_models = []
        for model_name, model in zip(
            self.model.model_names, self.model.models
        ):
            model_parameters = {}
            for parameter in model.parameter_ranges:
                parameter_name = self.model._inverted_parameter_map[
                    (model, parameter)
                ]
                model_parameters[parameter] = parameters.get(
                    parameter_name
                )
            rh_model = model.rotational_harmonics_representation(
                self.acquisition_scheme, **model_parameters)
            rh_models.append(rh_model)
        return np.array(rh_models)
