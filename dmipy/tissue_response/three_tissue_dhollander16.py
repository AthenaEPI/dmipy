from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np


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
    pass


def create_signal_decay_metric(acquisition_scheme, data):
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

    ratio = np.log(mean_b0[..., None] / mean_dwi_shells)
    SDM = np.mean(ratio, axis=-1)
    SDM[mean_b0 == 0.] = 0
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
