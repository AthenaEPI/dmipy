import numpy as np
from ..core.modeling_framework import ModelProperties


class RF1AnisotropicTissueResponseModel(ModelProperties):
    r""" Anisotropic tissue response kernel.

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

    def __init__(self, rotational_harmonics):
        self._rotational_harmonics_representation = rotational_harmonics
        self._spherical_mean = (
            rotational_harmonics[:, 0] / (2 * np.sqrt(np.pi)))

    def rotational_harmonics_representation(self, *kwargs):
        return self._rotational_harmonics_representation

    def spherical_mean(self, *kwargs):
        return self._spherical_mean


class RF2IsotropicTissueResponseModel(ModelProperties):
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

    def __init__(self, rotational_harmonics):
        self._rotational_harmonics_representation = rotational_harmonics
        self._spherical_mean = (
            rotational_harmonics[:, 0] / (2 * np.sqrt(np.pi)))

    def rotational_harmonics_representation(self, *kwargs):
        return self._rotational_harmonics_representation

    def spherical_mean(self, *kwargs):
        return self._spherical_mean
