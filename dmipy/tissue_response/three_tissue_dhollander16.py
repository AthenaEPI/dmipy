from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np
from dipy.reconst import dti
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
from dipy.segment.mask import median_otsu
from .wm_tournier12 import white_matter_response_tournier13
from dmipy.utils.utils import cart2mu
from dmipy.utils.spherical_convolution import real_sym_rh_basis
from .tissue_response_models import RF2IsotropicTissueResponseModel


def three_tissue_response_dhollander16(acquisition_scheme, data):
    """
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
    response_wm = white_matter_response_tournier13(acquisition_scheme, data_wm)

    response_csf = isotropic_tissue_response(
        acquisition_scheme, data[mask_CSF_refine])
    response_gm = isotropic_tissue_response(
        acquisition_scheme, data[mask_GM_refine])

    return response_wm, response_gm, response_csf


def signal_decay_metric(acquisition_scheme, data):
    """
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
    """optimal image threshold based on pearson correlation [1]_.
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
    rho = pearsonr(image, image > threshold)[0]
    return rho


def isotropic_tissue_response(acquisition_scheme, data):
    """
    calculates the spherical mean and rotational harmonics for isotropic tissue
    responses like grey matter and CSF.
    """
    sh_order = 0  # by definition
    N_shells = acquisition_scheme.shell_indices.max()
    rh_matrices = np.zeros((len(data),
                            N_shells,
                            sh_order // 2 + 1))
    for i in range(len(data)):
        for shell_index in acquisition_scheme.unique_dwi_indices:
            shell_mask = acquisition_scheme.shell_indices == shell_index
            shell_bvecs = acquisition_scheme.gradient_directions[shell_mask]
            theta, phi = cart2mu(shell_bvecs).T
            rh_mat = real_sym_rh_basis(sh_order, theta, phi)
            rh_matrices[i, shell_index - 1] = np.dot(
                np.linalg.pinv(rh_mat), data[i][shell_mask])
    kernel_rh_coeff = np.mean(rh_matrices, axis=0)
    return RF2IsotropicTissueResponseModel(kernel_rh_coeff)
