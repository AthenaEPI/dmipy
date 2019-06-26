import numpy as np
from dmipy.utils.utils import cart2mu
from dmipy.utils.spherical_convolution import real_sym_rh_basis
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import (
    AnisotropicSignalModelProperties,
    IsotropicSignalModelProperties)
from ..utils import utils
from dmipy.core.acquisition_scheme import gtab_dmipy2dipy
from dipy.reconst import dti


def estimate_TR1_isotropic_tissue_response_model(acquisition_scheme, data):
    """
    Estimates isotropic TissueResponseModel describing the convolution
    kernel of e.g. CSF or grey matter from array of candidate voxels [1]_.

    First, for each acquisition shell, the zeroth order rotational harmonic
    (RH) coefficient (so actually only the Y00 coefficient) is estimated.

    From the signal RH coefficients the S0 response and the signal attenuation
    RH coefficients are then separated. The TR2 model is then instantiated with
    these RH coefficients and returns allong with the S0 response.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : 2D array of size (N_voxels, N_DWIs),
            Candidate diffusion signal array to generate anisotropic tissue
            response from.

    Returns
    -------
    S0_response: positive float,
        average positive S0 response value of the input data.
    TR1: Dmipy TR1IsotropicTissueResponse instance,
        average shape model of the input data, defined only on the input data
        shells. Can be used as usual CompartmentModel.

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b-value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """

    if acquisition_scheme.N_TE > 1:
        msg = 'tissue response estimation not yet implemented for data with '\
              'multiple TE. Current data has {} TEs.'
        raise NotImplementedError(msg.format(acquisition_scheme.N_TE))

    rh_matrices = np.zeros((len(data), acquisition_scheme.N_dwi_shells, 1))
    S0_response = np.mean(data[:, acquisition_scheme.b0_mask])

    inv_sh_mats = {}
    for shell_index, sh_mat in acquisition_scheme.shell_sh_matrices.items():
        inv_sh_mats[shell_index] = np.linalg.pinv(sh_mat)

    for i in range(len(data)):
        for j, shell_index in enumerate(acquisition_scheme.unique_dwi_indices):
            shell_mask = acquisition_scheme.shell_indices == shell_index
            shell_data = data[i][shell_mask]
            rh_matrices[i, j, 0] = np.dot(
                inv_sh_mats[shell_index], shell_data)[0]
    rotational_harmonics_representation = np.mean(
        rh_matrices, axis=0) / S0_response
    TR1 = TR1IsotropicTissueResponseModel(
        acquisition_scheme, rotational_harmonics_representation)
    return S0_response, TR1


def estimate_TR2_anisotropic_tissue_response_model(acquisition_scheme, data):
    """
    Estimates TR2 anistropic TissueResponseModel describing the convolution
    kernel of e.g. anistropic white matter from array of candidate voxels [1]_

    First, Each candidate voxel is rotated such that the DTI eigenvector with
    the largest eigenvalue is aligned with the z-axis. The rotational harmonic
    (RH) coefficients (corresponding to Y_l0 spherical harmonics) are then
    estimated and saved per acquisition shell.

    From the signal RH coefficients the S0 response and the signal attenuation
    RH coefficients are then separated. The TR2 model is then instantiated with
    these RH coefficients and returns allong with the S0 response.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using Dmipy.
    data : 2D array of size (N_voxels, N_DWIs),
            Candidate diffusion signal array to generate anisotropic tissue
            response from.

    Returns
    -------
    S0_response: positive float,
        average positive S0 response value of the input data.
    TR2: Dmipy TR2AnisotropicTissueResponse instance,
        average shape model of the input data, defined only on the input data
        shells. Can be used as usual CompartmentModel.

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b-value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """
    if acquisition_scheme.N_TE > 1:
        msg = 'tissue response estimation not yet implemented for data with '\
              'multiple TE. Current data has {} TEs.'
        raise NotImplementedError(msg.format(acquisition_scheme.N_TE))
    gtab = gtab_dmipy2dipy(acquisition_scheme)
    tenmod = dti.TensorModel(gtab)
    tenfit = tenmod.fit(data)
    evecs = tenfit.evecs
    max_sh_order = max(acquisition_scheme.shell_sh_orders.values())
    rh_matrices = np.zeros(
        (len(data),
         acquisition_scheme.N_dwi_shells, int(max_sh_order // 2 + 1)))
    S0_response = np.mean(data[:, acquisition_scheme.b0_mask])

    for i in range(len(data)):
        # dipy's evecs are automatically ordered such that
        # lambda1 > lambda2 > lambda3 in xyz coordinate system. To estimate
        # rotational harmonics we need the eigenvector corresponding to the
        # largest lambda1 to be along the z-axis. This is why we rotate
        # the gradient directions with the reverse of the dti eigenvectors.
        bvecs_rot = np.dot(acquisition_scheme.gradient_directions,
                           evecs[i][:, ::-1])
        for j, shell_index in enumerate(acquisition_scheme.unique_dwi_indices):
            shell_sh = acquisition_scheme.shell_sh_orders[shell_index]
            shell_mask = acquisition_scheme.shell_indices == shell_index
            shell_bvecs = bvecs_rot[shell_mask]
            theta, phi = cart2mu(shell_bvecs).T
            rh_mat = real_sym_rh_basis(shell_sh, theta, phi)
            rh_matrices[i, j, :shell_sh // 2 + 1] = np.dot(
                np.linalg.pinv(rh_mat), data[i][shell_mask])
    rotational_harmonics_representation = np.mean(
        rh_matrices, axis=0) / S0_response

    TR2 = TR2AnisotropicTissueResponseModel(
        acquisition_scheme, rotational_harmonics_representation)

    return S0_response, TR2


class TR1IsotropicTissueResponseModel(
        ModelProperties, IsotropicSignalModelProperties):
    """
    The isotropic tissue response model is a non-parametric multi-shell
    representation of some signal attenuation. It can be instantiated with any
    rotational harmonics and accompanying DmipyAcquisitionScheme, but is
    usually instantiated from a set of segmented input data using the
    estimate_TR1_anisotropic_tissue_response_model function.

    Once estimated, this class behaves as a CompartmentModel object (so as if
    it were e.g. a cylinder or Gaussian compartment), but has no parameters.
    A TissueResponseModel can be input any MultiCompartment model
    representation in Dmipy, including together with parametric models.

    NOTE: TR models can ONLY generate signal attenuation at the same shells as
    the input rotational harmonics. However, TR models can be called with
    partial acquisition schemes. Meaning, ifthe TR model was instantiated with
    multi-shell acquisition scheme, and there is another scheme which is a
    subset of that scheme, it can be used to generate the partial data.

    Parameters
    ----------
    acquisition_scheme: DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using Dmipy.
    rotational_harmonics: array, shape(Nshells, 1),
            Y00 rotational harmonics coefficients for each shell.
    """
    _required_acquisition_parameters = []
    _parameter_ranges = {}
    _parameter_scales = {}
    _parameter_types = {}
    _model_type = 'TissueResponseModel'

    def __init__(self, acquisition_scheme, rotational_harmonics):
        self.acquisition_scheme = acquisition_scheme
        self._rotational_harmonics = rotational_harmonics

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Returns the signal attenuation. As a tissue response model, it cannot
        estimate any signal intensities outside the shells that were in the
        acquisition scheme that was used to estimate the model. However, it can
        estimate any gradient directions on the estimated shells.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E : float or array, shape(N),
            signal attenuation
        '''
        E = np.ones(acquisition_scheme.number_of_measurements, dtype=float)
        E_mean = self.spherical_mean(acquisition_scheme)
        for shell_index in acquisition_scheme.unique_dwi_indices:
            shell_mask = acquisition_scheme.shell_indices == shell_index
            E[shell_mask] = E_mean[shell_index]
        return E

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, 1),
            Y00 rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        rh_array = np.zeros((rh_scheme.N_dwi_shells, 1))
        for i, dwi_shell_index in enumerate(rh_scheme.unique_dwi_indices):
            b_value = acquisition_scheme.shell_bvalues[dwi_shell_index]
            shell_overlap = self.acquisition_scheme.shell_bvalues[
                ~self.acquisition_scheme.shell_b0_mask] == b_value
            if not np.any(shell_overlap):
                raise ValueError(
                    'bvalue {}s/mm^2 not in tissue response bvalues'.format(
                        b_value / 1e6))
            native_dwi_shell_index = np.where(shell_overlap)[0][0]
            rh_array[i] = self._rotational_harmonics[native_dwi_shell_index]
        return rh_array

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model for every acquisition shell.
        """
        E_mean = np.ones(acquisition_scheme.N_shells)
        rh_coef = self.rotational_harmonics_representation(acquisition_scheme)
        E_mean[~acquisition_scheme.shell_b0_mask] = (
            rh_coef[:, 0] / (2 * np.sqrt(np.pi)))
        return E_mean


class TR2AnisotropicTissueResponseModel(
        ModelProperties, AnisotropicSignalModelProperties):
    """
    The anisotropic tissue response model is a non-parametric multi-shell
    representation of some signal attenuation. It can be instantiated with any
    rotational harmonics and accompanying DmipyAcquisitionScheme, but is
    usually instantiated from a set of segmented input data using the
    estimate_TR2_anisotropic_tissue_response_model function.

    Once instantiated, this class behaves as a CompartmentModel object (so as
    if it were e.g. a cylinder or Gaussian compartment), but only has an
    orientation parameter. A TissueResponseModel can be input any
    MultiCompartment model representation in Dmipy, including together with
    parametric models.

    NOTE: TR models can ONLY generate signal attenuation at the same shells as
    the input rotational harmonics. However, TR models can be called with
    partial acquisition schemes. Meaning, ifthe TR model was instantiated with
    multi-shell acquisition scheme, and there is another scheme which is a
    subset of that scheme, it can be used to generate the partial data.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using Dmipy.
    rotational_harmonics: array of shape (N_shell_DWI, N_rh_coeff),
        Rotational harmonics that define the shape of the TR model.
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    """
    _required_acquisition_parameters = []
    _parameter_ranges = {'mu': ([0, np.pi], [-np.pi, np.pi])}
    _parameter_scales = {'mu': np.r_[1., 1.]}
    _parameter_types = {'mu': 'orientation'}
    _model_type = 'TissueResponseModel'

    def __init__(self, acquisition_scheme, rotational_harmonics, mu=None):
        self.acquisition_scheme = acquisition_scheme
        self._rotational_harmonics = rotational_harmonics
        self.mu = mu

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Returns the signal attenuation. As a tissue response model, it cannot
        estimate any signal intensities outside the shells that were in the
        acquisition scheme that was used to estimate the model. However, it can
        estimate any gradient directions on the estimated shells as it just
        maps the rotational harmonics to the shell for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E : float or array, shape(N),
            signal attenuation
        '''
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)

        # rotate gradient vectors according to orientation parameters
        R = utils.rotation_matrix_001_to_xyz(*mu)
        bvecs = acquisition_scheme.gradient_directions
        bvecs_rot = np.dot(bvecs, R)

        # now map the rotational harmonics to the rotated gradient vectors.
        rh_coef = self.rotational_harmonics_representation(acquisition_scheme)
        E = np.ones(acquisition_scheme.number_of_measurements, dtype=float)
        scheme = acquisition_scheme
        for i, dwi_shell_index in enumerate(scheme.unique_dwi_indices):
            shell_sh = scheme.shell_sh_orders[dwi_shell_index]
            shell_mask = scheme.shell_indices == dwi_shell_index
            shell_bvecs = bvecs_rot[shell_mask]
            theta, phi = cart2mu(shell_bvecs).T
            rh_mat = real_sym_rh_basis(shell_sh, theta, phi)
            E[shell_mask] = np.dot(rh_mat, rh_coef[i, :shell_sh // 2 + 1])
        return E

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        max_sh_order = max(rh_scheme.shell_sh_orders.values())
        rh_array = np.zeros((rh_scheme.N_dwi_shells, max_sh_order // 2 + 1))
        for i, dwi_shell_index in enumerate(rh_scheme.unique_dwi_indices):
            b_value = acquisition_scheme.shell_bvalues[dwi_shell_index]
            shell_overlap = self.acquisition_scheme.shell_bvalues[
                ~self.acquisition_scheme.shell_b0_mask] == b_value
            if not np.any(shell_overlap):
                raise ValueError(
                    'bvalue {}s/mm^2 not in tissue response bvalues'.format(
                        b_value / 1e6))
            native_dwi_shell_index = np.where(shell_overlap)[0][0]
            rh_array[i] = self._rotational_harmonics[
                native_dwi_shell_index, :max_sh_order // 2 + 1]
        return rh_array

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme for
        Zeppelin model.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the Zeppelin model for every acquisition shell.
        """
        E_mean = np.ones(acquisition_scheme.N_shells)
        rh_coef = self.rotational_harmonics_representation(acquisition_scheme)
        E_mean[~acquisition_scheme.shell_b0_mask] = (
            rh_coef[:, 0] / (2 * np.sqrt(np.pi)))
        return E_mean
