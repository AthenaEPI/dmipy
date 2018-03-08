import numpy as np
import cvxpy
from dipy.data import get_sphere, HemiSphere
from dipy.reconst.shm import real_sym_sh_mrtrix


__all__ = [
    'CvxpyOptimizer'
]


class CvxpyOptimizer:

    def __init__(self, acquisition_scheme, model, sh_order,
                 unity_constraint=True
                 ):
        self.model = model
        self.acquisition_scheme = acquisition_scheme
        self.sh_order = sh_order
        self.Ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
        self.Nmodels = len(self.model.models)
        self.unity_constraint = unity_constraint

        # prepare positivity grid on sphere
        sphere = get_sphere('symmetric724')
        hemisphere = HemiSphere(phi=sphere.phi, theta=sphere.theta)
        self.sh_matrix_positivity = real_sym_sh_mrtrix(
            self.sh_order, hemisphere.theta, hemisphere.phi)[0]

        orientation_counter = 0
        for model in self.model.models:
            if 'orientation' in model.parameter_types.values():
                orientation_counter += 1
        if orientation_counter > 1:
            msg = 'Cannot optimize the volume fractions of multiple models '
            msg += 'with an orientation at the same time without fixing all '
            msg += 'volume fractions.'
            raise ValueError(msg)

    def __call__(self, data, x0_vector):
        # step 1: determine rotational_harmonics of kernel
        rh_matrix = self.recover_rotational_harmonics(x0_vector)

        # step 2: set up cvxpy problem
        scheme = self.acquisition_scheme
        volume_fraction_sum = 0.
        volume_fractions = []
        mse = 0.
        constraints = []
        for j, model in enumerate(self.model.models):
            if 'orientation' in model.parameter_types.values():
                sh_coef = cvxpy.Variable(self.Ncoef)
                volume_fraction = sh_coef[0] * (2 * np.sqrt(np.pi))
                volume_fractions.append(volume_fraction)
                volume_fraction_sum += volume_fraction
                constraints.append(self.sh_matrix_positivity * sh_coef > 0.)
                for i, shell_index in enumerate(scheme.unique_dwi_indices):
                    shell_mask = scheme.shell_indices == shell_index
                    shell_data = data[shell_mask]
                    rh_order = int(scheme.shell_sh_orders[shell_index])
                    sh_mat = scheme.shell_sh_matrices[shell_index]
                    rh_shell_prepared = self.prepare_rotational_harmonics(
                        rh_matrix[j][i], rh_order)
                    if self.sh_order >= rh_order:
                        sh_shell = (
                            cvxpy.diag(rh_shell_prepared) *
                            sh_coef[:len(rh_shell_prepared)])
                        mse += cvxpy.sum_squares(
                            sh_mat * sh_shell - shell_data)
                    else:
                        sh_shell = (cvxpy.diag(
                            rh_shell_prepared[:self.Ncoef]) * sh_coef)
                        mse += cvxpy.sum_squares(
                            sh_mat[:, :self.Ncoef] * sh_shell - shell_data)
            else:
                c00_coef = cvxpy.Variable(1)
                volume_fraction = c00_coef[0] * (2 * np.sqrt(np.pi))
                volume_fractions.append(volume_fraction)
                volume_fraction_sum += volume_fraction
                constraints.append(c00_coef > 0.)
                for i, shell_index in enumerate(scheme.unique_dwi_indices):
                    shell_mask = scheme.shell_indices == shell_index
                    shell_data = data[shell_mask]
                    sh_mat = scheme.shell_sh_matrices[shell_index][0, 0]
                    rh_order = 0
                    rh_shell_prepared = self.prepare_rotational_harmonics(
                        rh_matrix[j][i], rh_order)
                    sh_shell = rh_shell_prepared * c00_coef
                    mse += cvxpy.sum_squares(sh_mat * sh_shell - shell_data)
        objective = cvxpy.Minimize(mse)
        if self.unity_constraint:
            constraints.append(volume_fraction_sum == 1.)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve()

        fitted_params = self.model.parameter_vector_to_parameters(x0_vector)
        fitted_params['sh_coeff'] = np.array(sh_coef.value).squeeze()
        if self.Nmodels > 1:
            fractions = [fraction.value for fraction in volume_fractions]
            fractions_array = np.array(fractions).squeeze()
            for i, name in enumerate(self.model.partial_volume_names):
                fitted_params[name] = fractions_array[i]
        return self.model.parameters_to_parameter_vector(**fitted_params)

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

    def recover_rotational_harmonics(self, x0_vector):
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
        return rh_models
