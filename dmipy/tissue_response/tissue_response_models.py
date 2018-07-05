import numpy as np
from dmipy.utils.utils import cart2mu
from dmipy.utils.spherical_convolution import real_sym_rh_basis
from ..core.modeling_framework import ModelProperties
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
from dipy.reconst import dti


class AnisotropicTissueResponseModel(ModelProperties):
    r""" Isotropic tissue response kernel.

    Parameters
    ----------
    rotational_harmonics : array, shape(Nshells, N_rh_coef),
        Rotational harmonics coefficients for each shell.

    References
    ----------
    .. [1] Tournier, J‐Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high‐angular‐resolution diffusion‐weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """

    _parameter_ranges = {}
    _parameter_scales = {}
    _parameter_types = {}
    _model_type = 'AnisotropicTissueResponse'

    def __init__(self, acquisition_scheme, data):
        gtab = gtab_mipy2dipy(acquisition_scheme)
        tenmod = dti.TensorModel(gtab)
        tenfit = tenmod.fit(data)
        evecs = tenfit.evecs
        N_shells = acquisition_scheme.shell_indices.max()
        rh_matrices = np.zeros(
            (len(data),
             N_shells + 1,
             acquisition_scheme.shell_sh_orders.max() // 2 + 1))

        for i in range(len(data)):
            bvecs_rot = np.dot(acquisition_scheme.gradient_directions,
                               evecs[i][:, ::-1])
            for shell_index in range(N_shells + 1):
                shell_sh = acquisition_scheme.shell_sh_orders[shell_index]
                shell_mask = acquisition_scheme.shell_indices == shell_index
                if acquisition_scheme.b0_mask[shell_mask][0]:
                    rh_matrices[i, shell_index, 0] = (
                        np.mean(data[i][shell_mask]) * 2 * np.sqrt(np.pi))
                else:
                    shell_bvecs = bvecs_rot[shell_mask]
                    theta, phi = cart2mu(shell_bvecs).T
                    rh_mat = real_sym_rh_basis(shell_sh, theta, phi)
                    rh_matrices[i, shell_index, :shell_sh // 2 + 1] = np.dot(
                        np.linalg.pinv(rh_mat), data[i][shell_mask])
        self._rotational_harmonics_representation = np.mean(
            rh_matrices, axis=0)
        self._spherical_mean = (
            self._rotational_harmonics_representation[:, 0] /
            (2 * np.sqrt(np.pi)))

    def rotational_harmonics_representation(self, *kwargs):
        return self._rotational_harmonics_representation

    def spherical_mean(self, *kwargs):
        return self._spherical_mean


class IsotropicTissueResponseModel(ModelProperties):
    r""" Isotropic tissue response kernel.

    Parameters
    ----------
    rotational_harmonics : array, shape(Nshells, N_rh_coef),
        Rotational harmonics coefficients for each shell.

    References
    ----------
    .. [1] Tournier, J‐Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high‐angular‐resolution diffusion‐weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """

    _parameter_ranges = {}
    _parameter_scales = {}
    _parameter_types = {}
    _model_type = 'IsotropicTissueResponse'

    def __init__(self, acquisition_scheme, data):
        N_shells = acquisition_scheme.shell_indices.max()
        rh_matrices = np.zeros((len(data), N_shells + 1, 1))

        for i in range(len(data)):
            for shell_index in range(N_shells + 1):
                shell_mask = acquisition_scheme.shell_indices == shell_index
                rh_matrices[i, shell_index, 0] = (
                    np.mean(data[i][shell_mask]) * 2 * np.sqrt(np.pi))
        self._rotational_harmonics_representation = np.mean(
            rh_matrices, axis=0)
        self._spherical_mean = (
            self._rotational_harmonics_representation[:, 0] /
            (2 * np.sqrt(np.pi)))

    def rotational_harmonics_representation(self, *kwargs):
        return self._rotational_harmonics_representation

    def spherical_mean(self, *kwargs):
        return self._spherical_mean
