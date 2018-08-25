import numpy as np
from dmipy.utils.utils import cart2mu
from dmipy.utils.spherical_convolution import real_sym_rh_basis
from ..core.modeling_framework import ModelProperties
from ..utils import utils
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
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b-value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """
    _parameter_ranges = {'mu': ([0, np.pi], [-np.pi, np.pi])}
    _parameter_scales = {'mu': np.r_[1., 1.]}
    _parameter_types = {'mu': 'orientation'}
    _model_type = 'TissueResponseModel'

    def __init__(self, acquisition_scheme, data, mu=None):
        self.mu = mu
        gtab = gtab_mipy2dipy(acquisition_scheme)
        tenmod = dti.TensorModel(gtab)
        tenfit = tenmod.fit(data)
        evecs = tenfit.evecs
        N_shells = acquisition_scheme.shell_indices.max()
        rh_matrices = np.zeros(
            (len(data),
             N_shells + 1,
             int(acquisition_scheme.shell_sh_orders.max() // 2 + 1)))
        self.S0_response = np.mean(data[:, acquisition_scheme.b0_mask])

        for i in range(len(data)):
            # dipy's evecs are automatically ordered such that
            # lambda1 > lambda2 > lambda3 in xyz coordinate system. To estimate
            # rotational harmonics we need the eigenvector corresponding to the
            # largest lambda1 to be along the z-axis. This is why we rotate
            # the gradient directions with the reverse of the dti eigenvectors.
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
            rh_matrices, axis=0) / self.S0_response
        self._spherical_mean = (
            self._rotational_harmonics_representation[:, 0] /
            (2 * np.sqrt(np.pi)))

    def __call__(self, acquisition_scheme, **kwargs):
        # rotate gradient vectors according to orientation parameters
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        R = utils.rotation_matrix_001_to_xyz(*mu)
        bvecs = acquisition_scheme.gradient_directions
        bvecs_rot = np.dot(bvecs, R)

        rh_coef = self._rotational_harmonics_representation
        E = np.ones_like(acquisition_scheme.bvalues)
        N_shells = acquisition_scheme.shell_indices.max()
        for shell_index in range(N_shells + 1):
            shell_sh = acquisition_scheme.shell_sh_orders[shell_index]
            shell_mask = acquisition_scheme.shell_indices == shell_index
            if acquisition_scheme.b0_mask[shell_mask][0]:
                E[shell_mask] = rh_coef[shell_index, 0] / (2 * np.sqrt(np.pi))
            else:
                shell_bvecs = bvecs_rot[shell_mask]
                theta, phi = cart2mu(shell_bvecs).T
                rh_mat = real_sym_rh_basis(shell_sh, theta, phi)
                E[shell_mask] = np.dot(
                    rh_mat, rh_coef[shell_index, :shell_sh // 2 + 1])
        return E

    def rotational_harmonics_representation(
            self, acquisition_scheme=None, **kwargs):
        return self._rotational_harmonics_representation[1:]

    def spherical_mean(self, acquisition_scheme=None, **kwargs):
        return self._spherical_mean

    def tissue_response(self, **kwargs):
        return self.S0_response * self._spherical_mean


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

    A TissueResponseModel has a standard, rotational_harmonics_representation
    and a spherical_mean, just like a regular CompartmentModel. As such, a
    TissueResponseModel can be input to all the usual MultiCompartmentModel
    representations.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : 2D array of size (N_voxels, N_DWIs),
            Candidate diffusion signal array to generate anisotropic tissue
            response from.

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b-value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """
    _parameter_ranges = {}
    _parameter_scales = {}
    _parameter_types = {}
    _model_type = 'TissueResponseModel'

    def __init__(self, acquisition_scheme, data):
        N_shells = acquisition_scheme.shell_indices.max()
        rh_matrices = np.zeros((len(data), N_shells + 1, 1))
        self.S0_response = np.mean(data[:, acquisition_scheme.b0_mask])

        for i in range(len(data)):
            for shell_index in range(N_shells + 1):
                shell_mask = acquisition_scheme.shell_indices == shell_index
                rh_matrices[i, shell_index, 0] = (
                    np.mean(data[i][shell_mask]) * 2 * np.sqrt(np.pi))
        self._rotational_harmonics_representation = np.mean(
            rh_matrices, axis=0) / self.S0_response
        self._spherical_mean = (
            self._rotational_harmonics_representation[:, 0] /
            (2 * np.sqrt(np.pi)))

    def __call__(self, **kwargs):
        """ for isotropic models, the spherical mean is the same as the DWI
        intensity. """
        return self._spherical_mean

    def rotational_harmonics_representation(
            self, acquisition_scheme=None, **kwargs):
        return self._rotational_harmonics_representation[1:]

    def spherical_mean(self, acquisition_scheme=None, **kwargs):
        return self._spherical_mean

    def tissue_response(self, **kwargs):
        return self.S0_response * self._spherical_mean
