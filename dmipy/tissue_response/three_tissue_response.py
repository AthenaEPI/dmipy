from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np
from dipy.reconst import dti
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
from dipy.segment.mask import median_otsu
import white_matter_response
from ..signal_models.tissue_response_models import IsotropicTissueResponseModel

_white_matter_response_algorithms = {
    'tournier07': white_matter_response.white_matter_response_tournier07,
    'tournier13': white_matter_response.white_matter_response_tournier13
}


def three_tissue_response_dhollander16(
        acquisition_scheme, data, wm_algorithm='tournier07', **kwargs):
    """
    Heuristic approach to estimating the white matter, grey matter and CSF
    tissue response kernels [1]_, to be used in e.g. Multi-Tissue CSD [2]_. The
    method makes used of so-called 'optimal' thresholds between grey-scale
    images and segmentations [3]_, with iteratively refined binary thresholds
    based on an ad-hoc 'signal decay metric', to finally find candidate voxels
    from which to estimate the three tissue response kernels.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : NDarray,
        Measured diffusion signal array.

    Returns
    -------
    wm_model : Dmipy Anisotropic ModelFree Model,
            ModelFree representation of white matter response.
    gm_model : Dmipy Isotropic ModelFree Model,
        ModelFree representation of grey matter response.
    csf_model : Dmipy Isotropic ModelFree Model,
            ModelFree representation of csf response.

    References
    ----------
    .. [1] Dhollander, T.; Raffelt, D. & Connelly, A. Unsupervised 3-tissue
        response function estimation from single-shell or multi-shell diffusion
        MR data without a co-registered T1 image. ISMRM Workshop on Breaking
        the Barriers of Diffusion MRI, 2016, 5
    .. [2] Tournier, J‐Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high‐angular‐resolution diffusion‐weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    .. [3] Ridgway, Gerard R., et al. "Issues with threshold masking in
        voxel-based morphometry of atrophied brains." Neuroimage 44.1 (2009):
        99-111.
    """
    # Create Signal Decay Metric (SDM)
    mean_b0 = np.mean(data[..., acquisition_scheme.b0_mask], axis=-1)
    SDM = signal_decay_metric(acquisition_scheme, data)

    # Make Mask
    b0_mask, mask = median_otsu(data, 2, 1)
    gtab = gtab_mipy2dipy(acquisition_scheme)
    tenmod = dti.TensorModel(gtab)
    tenfit = tenmod.fit(b0_mask)
    fa = tenfit.fa
    mask_WM = fa > 0.2

    # Separate grey and CSF based on optimal threshold
    opt = optimal_threshold(SDM, fa < 0.2)
    mask_CSF = np.all([mean_b0 > 0, mask, fa < 0.2, SDM > opt], axis=0)
    mask_GM = np.all([mean_b0 > 0, mask, fa < 0.2, SDM < opt], axis=0)

    # Refine Mask, high WM SDM outliers above Q 3 +(Q 3 -Q 1 ) are removed.
    median_WM = np.median(SDM[mask_WM])
    Q1 = (SDM[mask_WM].min() + median_WM) / 2.0
    Q3 = (SDM[mask_WM].max() + median_WM) / 2.0
    SDM_upper_threshold = Q3 + (Q3 - Q1)
    mask_WM_refine = np.all([mask_WM, SDM < SDM_upper_threshold], axis=0)
    WM_outlier = np.all([mask_WM, SDM > SDM_upper_threshold], axis=0)

    # For both the voxels below and above the GM SDM median, optimal thresholds
    # [4] are computed and both parts closer to the initial GM median are
    # retained.
    SDM_GM = SDM[mask_GM]
    median_GM = np.median(SDM_GM)
    optimal_threshold_upper = optimal_threshold(SDM_GM, SDM_GM > median_GM)
    optimal_threshold_lower = optimal_threshold(SDM_GM, SDM_GM < median_GM)
    mask_GM_refine = np.all(
        [mask_GM,
         SDM > optimal_threshold_lower,
         SDM < optimal_threshold_upper], axis=0)

    # The high SDM outliers that were removed from the WM are reconsidered for
    # the CSF if they have higher SDM than the current minimal CSF SDM.
    SDM_CSF_min = SDM[mask_CSF].min()
    WM_outlier_to_include = np.all([WM_outlier, SDM > SDM_CSF_min], axis=0)
    mask_CSF_updated = np.any([mask_CSF, WM_outlier_to_include], axis=0)

    # An optimal threshold [4] is computed for the resulting CSF and only the
    # higher SDM valued voxels are retained.
    optimal_threshold_CSF = optimal_threshold(SDM, mask_CSF_updated)
    mask_CSF_refine = np.all(
        [mask_CSF_updated, SDM > optimal_threshold_CSF], axis=0)

    data_wm = data[mask_WM_refine]

    # for WM we use WM response selection algorithm
    response_wm_algorithm = _white_matter_response_algorithms[wm_algorithm]
    response_wm, indices_wm_selected = response_wm_algorithm(
        acquisition_scheme, data_wm, **kwargs)

    # for GM, the voxels closest 2% to GM SDM median are selected.
    median_GM = np.median(SDM[mask_GM_refine])
    N_threshold = int(np.sum(mask_GM_refine) * 0.02)
    indices_gm_selected = np.argsort(
        np.abs(SDM[mask_GM_refine] - median_GM))[:N_threshold]
    response_gm = IsotropicTissueResponseModel(
        acquisition_scheme, data[mask_GM_refine][indices_gm_selected])

    # for GM, the 10% highest SDM valued voxels are selected.
    N_threshold = int(np.sum(mask_CSF_refine) * 0.1)
    indices_csf_selected = np.argsort(SDM[mask_CSF_refine])[::-1][:N_threshold]
    response_csf = IsotropicTissueResponseModel(
        acquisition_scheme, data[mask_CSF_refine][indices_csf_selected])

    pos_WM_refine = np.c_[np.where(mask_WM_refine)]
    mask_WM_selected = np.zeros_like(mask_WM_refine)
    pos_WM_selected = pos_WM_refine[indices_wm_selected]
    for pos in pos_WM_selected:
        mask_WM_selected[pos[0], pos[1], pos[2]] = 1

    pos_GM_refine = np.c_[np.where(mask_GM_refine)]
    mask_GM_selected = np.zeros_like(mask_GM_refine)
    pos_GM_selected = pos_GM_refine[indices_gm_selected]
    for pos in pos_GM_selected:
        mask_GM_selected[pos[0], pos[1], pos[2]] = 1

    pos_CSF_refine = np.c_[np.where(mask_CSF_refine)]
    mask_CSF_selected = np.zeros_like(mask_CSF_refine)
    pos_CSF_selected = pos_CSF_refine[indices_csf_selected]
    for pos in pos_CSF_selected:
        mask_CSF_selected[pos[0], pos[1], pos[2]] = 1

    three_tissue_selection = np.array(
        [mask_WM_selected, mask_GM_selected, mask_CSF_selected], dtype=float)
    three_tissue_selection = np.transpose(three_tissue_selection, (1, 2, 3, 0))

    return response_wm, response_gm, response_csf, three_tissue_selection


def signal_decay_metric(acquisition_scheme, data):
    """
    Estimation of the Signal Decay Metric (SDM) for the three-tissue tissue
    response kernel estimation [1]_. The metric is a simple division of the S0
    signal intensity by the b>0 shell's signal intensity - of the mean of their
    intensities if there are multiple shells.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
        An acquisition scheme that has been instantiated using dMipy.
    data : NDarray,
        Measured diffusion signal array.

    Returns
    -------
    SDM : array of size data,
        Estimated Signal Decay Metric (SDK)

    References
    ----------
    .. [1] Dhollander, T.; Raffelt, D. & Connelly, A. Unsupervised 3-tissue
        response function estimation from single-shell or multi-shell diffusion
        MR data without a co-registered T1 image. ISMRM Workshop on Breaking
        the Barriers of Diffusion MRI, 2016, 5
    """
    mean_b0 = np.mean(data[..., acquisition_scheme.b0_mask], axis=-1)
    data_shape = data.shape[:-1]
    mean_dwi_shells = np.zeros(
        np.r_[data_shape, len(acquisition_scheme.unique_dwi_indices)])
    for i, index in enumerate(acquisition_scheme.unique_dwi_indices):
        shell_mask = acquisition_scheme.shell_indices == index
        mean_dwi_shells[..., i] = np.mean(data[..., shell_mask], axis=-1)

    SDM = np.zeros(data_shape)
    mask = mean_b0 > 0
    ratio = np.log(mean_b0[mask, None] / mean_dwi_shells[mask])
    SDM[mask] = np.mean(ratio, axis=-1)
    return SDM


def optimal_threshold(image, mask):
    """Optimal image threshold based on pearson correlation [1]_.
    T* = argmin_T (\rho(image, image>T))

    References
    ----------
    .. [1] Ridgway, Gerard R., et al. "Issues with threshold masking in
        voxel-based morphometry of atrophied brains." Neuroimage 44.1 (2009):
        99-111.
    """
    masked_voxels = image[mask]
    min_bound = masked_voxels.min()
    max_bound = masked_voxels.max()
    optimal_threshold = minimize(
        fun=_cost_function,
        x0=(min_bound + max_bound) / 2.0,
        args=(masked_voxels,),
        bounds=([min_bound, max_bound],)).x
    return optimal_threshold[0]


def _cost_function(threshold, image):
    "The cost function used by the optimal_threshold function."
    rho = pearsonr(image, image > threshold)[0]
    return rho
