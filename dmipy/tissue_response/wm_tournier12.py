from dipy.reconst import dti
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
import numpy as np
from dipy.segment.mask import median_otsu
from .tissue_response_models import AnisotropicTissueResponseModel
from scipy.ndimage import binary_erosion
from dipy.data import get_sphere
from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)


def white_matter_response_tournier13(
        acquisition_scheme, data, rh_order=10, max_iter=5, sh_order=10):
    """
    Iterative model-free white matter response function estimation according to
    [1]_. Quoting the paper, the steps are the following:

    - 1) The 300 brain voxels with the highest FA were identified within a
        brain mask (eroded by three voxels to remove any noisy voxels at the
        brain edges).
    - 2) The single-fibre ‘response function’ was estimated within these
        voxels, and used to compute the fibre orientation distribution (FOD)
        employing constrained spherical deconvolution (CSD) up to lmax = 10.
    - 3) Within each voxel, a peak-finding procedure was used to identify the
        two largest FOD peaks, and their amplitude ratio was computed.
    - 4) The 300 voxels with the lowest second to first peak amplitude ratios
        were identified, and used as the current estimate of the set of
        ‘single-fibre’ voxels. It should be noted that these voxels were not
        required to be a subset of the original set of ‘single-fibre’ voxels.
    - 5) To ensure minimal bias from the initial estimate of the ‘response
        function’, steps (2) to (4) were re-iterated until convergence (no
        difference in the set of ‘single-fibre’ voxels). It should be noted
        that, in practice, convergence was achieved within a single iteration
        in all cases.

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

    References
    ----------
    .. [1] Tournier, J‐Donald, Fernando Calamante, and Alan Connelly.
        "Determination of the appropriate b value and number of gradient
        directions for high‐angular‐resolution diffusion‐weighted imaging."
        NMR in Biomedicine 26.12 (2013): 1775-1786.
    """
    data_shape = np.atleast_2d(data).shape
    N_voxels = int(np.prod(data_shape[:-1]))
    N_select = 300
    if N_voxels < 300:
        msg = "The original algorithm uses 300 candidate voxels to estimate "
        msg += "the tissue response. Currently only {} ".format(N_voxels)
        msg += "candidate voxels given."
        print(msg)
        N_select = N_voxels

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

    gtab = gtab_mipy2dipy(acquisition_scheme)
    tenmod = dti.TensorModel(gtab)
    tenfit = tenmod.fit(data_to_fit)
    fa = tenfit.fa

    # selected based on FA
    selected_indices = np.argsort(fa)[-N_select:]
    sphere = get_sphere('symmetric724')
    # iterate until convergence
    for it in range(max_iter):
        print('Tournier13 white matter response iteration {}'.format(it + 1))
        selected_data = data_to_fit[selected_indices]

        wm_model = AnisotropicTissueResponseModel(
            acquisition_scheme, selected_data)
        sh_model = MultiCompartmentSphericalHarmonicsModel([wm_model],
                                                           sh_order=sh_order)
        sh_fit = sh_model.fit(acquisition_scheme, data_to_fit,
                              solver='csd_tournier07',
                              use_parallel_processing=False)
        peaks, values, indices = sh_fit.peaks_directions(
            sphere, max_peaks=2, relative_peak_threshold=0.)
        ratio = values[..., 1] / values[..., 0]
        selected_indices_old = selected_indices
        selected_indices = np.argsort(ratio)[-N_select:]
        if np.array_equal(selected_indices, selected_indices_old):
            break
    print('White matter response converged')
    return wm_model
