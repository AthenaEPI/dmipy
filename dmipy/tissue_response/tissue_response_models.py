import numpy as np
from dmipy.utils.utils import cart2mu
from dmipy.utils.spherical_convolution import real_sym_rh_basis
from ..core.modeling_framework import ModelProperties
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
from dipy.reconst import dti


class AnisotropicTissueResponseModel(ModelProperties):
    r""" Estimates anistropic TissueResponseModel describing the convolution
    kernel of e.g. anistropic white matter from array of candidate voxels [1]_.

    First, Each candidate voxel is rotated such that the DTI eigenvector with
    the largest eigenvalue is aligned with the z-axis. The rotational harmonic
    (RH) coefficients (corresponding to Y_l0 spherical harmonics) are then
    estimated and saved per acquisition shell. From the estimated
    RH-coefficients the spherical mean per shell is also estimated.

    Once estimated, this class behaves as a CompartmentModel object (so as if
    it were e.g. a cylinder or Gaussian compartment), but has no parameters and
    - if the S0 of the input is not normalized to one - also its S0-value will
    not be one.

    A TissueResponseModel has a rotational_harmonics_representation and a
    spherical_mean, but no regular DWI representation. This means a
    TissueResponseModel can be input to a MultiCompartmentSphericalMeanModel or
    a MultiCompartmentSphericalHarmonicsModel, but NOT a regular
    MultiCompartmentModel.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : 2D array of size (N_voxels, N_DWIs),
            Candidate diffusion signal array to generate anisotropic tissue
            response from.

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
    r""" Estimates istropic TissueResponseModel describing the convolution
    kernel of e.g. CSF or grey matter from array of candidate voxels [1]_.

    First, for each acquisition shell, the zeroth order rotational harmonic
    (RH) coefficient (so actually only the Y00 coefficient) is estimated. From
    the estimated RH-coefficients the spherical mean per shell is also
    estimated.

    Once estimated, this class behaves as a CompartmentModel object (so as if
    it were e.g. a cylinder or Gaussian compartment), but has no parameters and
    - if the S0 of the input is not normalized to one - also its S0-value will
    not be one.

    A TissueResponseModel has a rotational_harmonics_representation and a
    spherical_mean, but no regular DWI representation. This means a
    TissueResponseModel can be input to a MultiCompartmentSphericalMeanModel or
    a MultiCompartmentSphericalHarmonicsModel, but NOT a regular
    MultiCompartmentModel.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : 2D array of size (N_voxels, N_DWIs),
            Candidate diffusion signal array to generate anisotropic tissue
            response from.

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
