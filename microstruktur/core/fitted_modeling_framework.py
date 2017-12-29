import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from microstruktur.utils.utils import unitsphere2cart_Nd
from microstruktur.utils.spherical_mean import (
    estimate_spherical_mean_multi_shell)


class FittedMultiCompartmentModel:
    def __init__(self, model, S0, mask, fitted_parameters_vector):
        self.model = model
        self.S0 = S0
        self.mask = mask
        self.fitted_parameters_vector = fitted_parameters_vector

    @property
    def fitted_parameters(self):
        return self.model.parameter_vector_to_parameters(
            self.fitted_parameters_vector)

    def fod(self, vertices):
        if not self.model.fod_available:
            msg = ('FODs not available for current model.')
            raise ValueError(msg)
        dataset_shape = self.fitted_parameters_vector.shape[:-1]
        N_samples = len(vertices)
        fods = np.zeros(np.r_[dataset_shape, N_samples])
        mask_pos = np.where(self.mask)
        for pos in zip(*mask_pos):
            parameters = self.model.parameter_vector_to_parameters(
                self.fitted_parameters_vector[pos])
            fods[pos] = self.model(vertices, quantity='FOD', **parameters)
        return fods

    def fod_sh(self, sh_order=8, basis_type=None):
        if not self.model.fod_available:
            msg = ('FODs not available for current model.')
            raise ValueError(msg)
        sphere = get_sphere(name='repulsion724')
        vertices = sphere.vertices
        _, inv_sh_matrix = sh_to_sf_matrix(
            sphere, sh_order, basis_type=basis_type, return_inv=True)
        fods_sf = self.fod(vertices)

        dataset_shape = self.fitted_parameters_vector.shape[:-1]
        number_coef_used = int((sh_order + 2) * (sh_order + 1) // 2)
        fods_sh = np.zeros(np.r_[dataset_shape, number_coef_used])
        mask_pos = np.where(self.mask)
        for pos in zip(*mask_pos):
            fods_sh[pos] = np.dot(inv_sh_matrix.T, fods_sf[pos])
        return fods_sh

    def peaks_spherical(self):
        mu_params = []
        for name, card in self.model.parameter_cardinality.items():
            if name[-2:] == 'mu' and card == 2:
                mu_params.append(self.fitted_parameters[name])
        if len(mu_params) == 0:
            msg = ('peaks not available for current model.')
            raise ValueError(msg)
        if len(mu_params) == 1:
            return mu_params[0]
        return np.concatenate([mu[..., None] for mu in mu_params], axis=-1)

    def peaks_cartesian(self):
        peaks_spherical = self.peaks_spherical()
        peaks_cartesian = unitsphere2cart_Nd(peaks_spherical)
        return peaks_cartesian

    def predict(self, acquisition_scheme=None, S0=None, mask=None):
        if acquisition_scheme is None:
            acquisition_scheme = self.model.scheme
        dataset_shape = self.fitted_parameters_vector.shape[:-1]
        if S0 is None:
            S0 = self.S0
        elif isinstance(S0, float):
            S0 = np.ones(dataset_shape) * S0
        if mask is None:
            mask = self.mask

        if self.model.spherical_mean:
            N_samples = len(acquisition_scheme.shell_bvalues)
        else:
            N_samples = len(acquisition_scheme.bvalues)

        predicted_signal = np.zeros(np.r_[dataset_shape, N_samples])
        mask_pos = np.where(mask)
        for pos in zip(*mask_pos):
            parameters = self.model.parameter_vector_to_parameters(
                self.fitted_parameters_vector[pos])
            predicted_signal[pos] = self.model(
                acquisition_scheme, **parameters) * S0[pos]
        return predicted_signal

    def R2_coefficient_of_determination(self, data):
        "Calculates the R-squared of the model fit."
        if self.model.spherical_mean:
            Nshells = len(self.model.scheme.shell_bvalues)
            data_ = np.zeros(np.r_[data.shape[:-1], Nshells])
            for pos in zip(*np.where(self.mask)):
                data_[pos] = estimate_spherical_mean_multi_shell(
                    data[pos] / self.S0[pos], self.model.scheme)
        else:
            data_ = data / self.S0[..., None]

        y_hat = self.predict(S0=1.)
        y_bar = np.mean(data_, axis=-1)
        SStot = np.sum((data_ - y_bar[..., None]) ** 2, axis=-1)
        SSres = np.sum((data_ - y_hat) ** 2, axis=-1)
        R2 = 1 - SSres / SStot
        R2[~self.mask] = 0
        return R2

    def mean_squared_error(self, data):
        "Calculates the mean squared error of the model fit."
        if self.model.spherical_mean:
            Nshells = len(self.model.scheme.shell_bvalues)
            data_ = np.zeros(np.r_[data.shape[:-1], Nshells])
            for pos in zip(*np.where(self.mask)):
                data_[pos] = estimate_spherical_mean_multi_shell(
                    data[pos] / self.S0[pos], self.model.scheme)
        else:
            data_ = data / self.S0[..., None]

        y_hat = self.predict(S0=1.)
        mse = np.mean((data_ - y_hat) ** 2, axis=-1)
        mse[~self.mask] = 0
        return mse
