from dipy.reconst import dti
from ..core.acquisition_scheme import gtab_dmipy2dipy
from ..core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
import numpy as np
from dipy.segment.mask import median_otsu
from dipy.data import get_sphere, HemiSphere
from ..signal_models.tissue_response_models import (
    AnisotropicTissueResponseModel)
from scipy.ndimage import binary_erosion


def white_matter_response_tournier07(
        acquisition_scheme, data, N_candidate_voxels=300, **kwargs):
    """The original white matter response estimation algorithm according to
    [1]_. In essence, it just takes the 300 voxels with the highest FA, aligns
    them with the z-axis, and estimates the averaged white matter response from
    that.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : NDarray,
        Measured diffusion signal array.

    Returns
    -------
    wm_model : Dmipy Anisotropic ModelFree Model
        ModelFree representation of white matter response.
    selected_voxel_indices : array of size (N_candidate_voxels,),
        indices of selected voxels for white matter response.

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly. "Robust
        determination of the fibre orientation distribution in diffusion MRI:
        non-negativity constrained super-resolved spherical deconvolution."
        Neuroimage 35.4 (2007): 1459-1472.
    """
    data_shape = np.atleast_2d(data).shape
    N_voxels = int(np.prod(data_shape[:-1]))
    if N_voxels < N_candidate_voxels:
        msg = "The original algorithm uses 300 candidate voxels to estimate "
        msg += "the tissue response. Currently only {} ".format(N_voxels)
        msg += "candidate voxels given."
        print(msg)
        N_candidate_voxels = N_voxels

    if data.ndim == 4:
        # calculate brain mask on 4D data (x, y, z, DWI)
        b0_mask, mask = median_otsu(data, 2, 1)
        # needs to be eroded 3 times.
        mask_eroded = binary_erosion(mask, iterations=3)
        data_to_fit = data[mask_eroded]
    else:
        # can't calculate brain mask on other than 4D data.
        # assume the data was prepared.
        data_to_fit = data.reshape([-1, data_shape[-1]])

    gtab = gtab_dmipy2dipy(acquisition_scheme)
    tenmod = dti.TensorModel(gtab)
    tenfit = tenmod.fit(data_to_fit)
    fa = tenfit.fa

    # selected based on FA
    selected_voxel_indices = np.argsort(fa)[-N_candidate_voxels:]
    selected_data = data_to_fit[selected_voxel_indices]
    wm_model = AnisotropicTissueResponseModel(
        acquisition_scheme, selected_data)
    return wm_model, selected_voxel_indices


def white_matter_response_tournier13(
        acquisition_scheme, data, max_iter=5, sh_order=10,
        N_candidate_voxels=300, peak_ratio_setting='mrtrix'):
    """
    Iterative model-free white matter response function estimation according to
    [1]_. Quoting the paper, the steps are the following:

    - 1) The 300 brain voxels with the highest FA were identified within a
        brain mask (eroded by three voxels to remove any noisy voxels at the
        brain edges).
    - 2) The single-fibre 'response function' was estimated within these
        voxels, and used to compute the fibre orientation distribution (FOD)
        employing constrained spherical deconvolution (CSD) up to lmax = 10.
    - 3) Within each voxel, a peak-finding procedure was used to identify the
        two largest FOD peaks, and their amplitude ratio was computed.
    - 4) The 300 voxels with the lowest second to first peak amplitude ratios
        were identified, and used as the current estimate of the set of
        'single-fibre' voxels. It should be noted that these voxels were not
        required to be a subset of the original set of 'single-fibre' voxels.
    - 5) To ensure minimal bias from the initial estimate of the 'response
        function', steps (2) to (4) were re-iterated until convergence (no
        difference in the set of 'single-fibre' voxels). It should be noted
        that, in practice, convergence was achieved within a single iteration
        in all cases.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : NDarray,
        Measured diffusion signal array.
    max_iter : Positive integer,
        Defines the maximum amount of iterations to be done for the single-
        fibre response kernel.
    sh_order : Positive even integer,
        Maximum spherical harmonics order to be used in the FOD estimation for
        the single-fibre response kernel.
    N_candidate_voxels : integer,
        Number of voxels to be included in the final white matter response
        estimation. Default is 300 following [1]_.
    peak_ratio_setting : string,
        Can be either 'ratio' or 'mrtrix', meaning the 'ratio' parameter
        between two peaks is actually calculated as the ratio, or a more
        complicated version as 1 / sqrt(peak1 * (1 - peak2 / peak1)) ** 2, to
        avoid favouring small, yet low SNR FODs [2]_.

    Returns
    -------
    wm_model : Dmipy Anisotropic ModelFree Model
        ModelFree representation of white matter response.
    selected_indices : array of size (N_candidate_voxels,),
        indices of selected voxels for white matter response.

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high-angular-resolution diffusion-weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    .. [2] MRtrix 3.0 readthedocs
    """
    data_shape = np.atleast_2d(data).shape
    N_voxels = int(np.prod(data_shape[:-1]))
    if N_voxels < N_candidate_voxels:
        msg = "The parameter N_candidate voxels is set to {} but only ".format(
            N_candidate_voxels)
        msg += "{} voxels are given. N_candidate_voxels".format(N_voxels)
        msg += " reset to number of voxels given."
        print(msg)
        N_candidate_voxels = N_voxels

    ratio_settings = ['ratio', 'mrtrix']
    if peak_ratio_setting not in ratio_settings:
        msg = 'peak_ratio_setting must be in {}'.format(ratio_settings)
        raise ValueError(msg)

    if data.ndim == 4:
        # calculate brain mask on 4D data (x, y, z, DWI)
        b0_mask, mask = median_otsu(data, 2, 1)
        # needs to be eroded 3 times.
        mask_eroded = binary_erosion(mask, iterations=3)
        data_to_fit = data[mask_eroded]
    else:
        # can't calculate brain mask on other than 4D data.
        # assume the data was prepared.
        data_to_fit = data.reshape([-1, data_shape[-1]])

    gtab = gtab_dmipy2dipy(acquisition_scheme)
    tenmod = dti.TensorModel(gtab)
    tenfit = tenmod.fit(data_to_fit)
    fa = tenfit.fa

    # selected based on FA
    selected_indices = np.argsort(fa)[-N_candidate_voxels:]
    sphere = get_sphere('symmetric724')
    hemisphere = HemiSphere(theta=sphere.theta, phi=sphere.phi)
    # iterate until convergence
    it = 0
    while True:
        print('Tournier13 white matter response iteration {}'.format(it + 1))
        selected_data = data_to_fit[selected_indices]

        wm_model = AnisotropicTissueResponseModel(
            acquisition_scheme, selected_data)
        sh_model = MultiCompartmentSphericalHarmonicsModel([wm_model],
                                                           sh_order=sh_order)
        sh_fit = sh_model.fit(acquisition_scheme, data_to_fit,
                              solver='csd_tournier07',
                              use_parallel_processing=False,
                              lambda_lb=0.)
        peaks, values, indices = sh_fit.peaks_directions(
            hemisphere, max_peaks=2, relative_peak_threshold=0.)
        if peak_ratio_setting == 'ratio':
            ratio = values[..., 1] / values[..., 0]
        elif peak_ratio_setting == 'mrtrix':
            ratio = 1. / np.sqrt(
                values[..., 0] * (1 - values[..., 1] / values[..., 0])) ** 2
        selected_indices_old = selected_indices
        selected_indices = np.argsort(ratio)[:N_candidate_voxels]
        percentage_overlap = 100 * float(len(np.intersect1d(
            selected_indices, selected_indices_old))) / N_candidate_voxels
        print('{:.1f} percent candidate voxel overlap.'.format(
            percentage_overlap))
        if percentage_overlap == 100.:
            print('White matter response converged')
            break
        it += 1
        if it > max_iter:
            print('Maximum iterations reached without convergence')
            break
    return wm_model, selected_indices
