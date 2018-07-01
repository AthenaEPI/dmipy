import numpy as np


class RF1AnisotropicTissueResponseModel():
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
    _model_type = 'CompartmentModel'

    def __init__(self, rotational_harmonics):
        self.rotational_harmonics_representation = rotational_harmonics
        self.spherical_mean = rotational_harmonics[:, 0] / (2 * np.sqrt(np.pi))


class RF2IsotropicTissueResponseModel():
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
    _model_type = 'CompartmentModel'

    def __init__(self, rotational_harmonics):
        self.rotational_harmonics_representation = rotational_harmonics
        self.spherical_mean = rotational_harmonics[:, 0] / (2 * np.sqrt(np.pi))