from dipy.reconst import dti
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
import numpy as np
from dipy.segment.mask import median_otsu
from ..signal_models.tissue_response_models import (
    AnisotropicTissueResponseModel)
from scipy.ndimage import binary_erosion
from dipy.data import get_sphere, HemiSphere
from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)


def white_matter_response_tournier07(acquisition_scheme, data, **kwargs):
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

    References
    ----------
    .. [1] Tournier, J-Donald, Fernando Calamante, and Alan Connelly. "Robust
        determination of the fibre orientation distribution in diffusion MRI:
        non-negativity constrained super-resolved spherical deconvolution."
        Neuroimage 35.4 (2007): 1459-1472.
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
    selected_data = data_to_fit[selected_indices]
    wm_model = AnisotropicTissueResponseModel(
        acquisition_scheme, selected_data)
    return wm_model
