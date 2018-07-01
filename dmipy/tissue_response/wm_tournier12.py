from dti.reconst import dti
from dmipy.core.acquisition_scheme import gtab_mipy2dipy
import numpy as np
from dmipy.utils.spherical_convolution import real_sym_rh_basis
from dmipy.utils.utils import cart2mu
from .tissue_response_models import RF1AnisotropicTissueResponse
from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)


def white_matter_response_tournier13(acquisition_scheme, data, rh_order=10):
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
    gtab = gtab_mipy2dipy(acquisition_scheme)
    tenmod = dti.TensorModel(gtab)
    tenfit = tenmod.fit(data)
    fa = tenfit.fa
    evecs = tenfit.evecs
    largest_fa_indices = np.argsort(fa)[-300:]
    largest_fa_data = data[largest_fa_indices]
    largest_fa_evecs = evecs[largest_fa_indices]

    rh_matrices = np.zeros((len(largest_fa_data),
                            acquisition_scheme.N_shells,
                            rh_order // 2 + 1))
    for i in range(len(largest_fa_data)):
        for shell_index in acquisition_scheme.unique_dwi_indices:
            bvecs_rot = np.dot(acquisition_scheme.gradient_directions,
                               largest_fa_evecs[i])
            shell_mask = acquisition_scheme.shell_indices == shell_index
            shell_bvecs_rot = bvecs_rot[shell_mask]
            theta, phi = cart2mu(shell_bvecs_rot).T
            rh_mat = real_sym_rh_basis(10, theta, phi)
            rh_matrices[i, shell_index - 1] = np.dot(
                np.linalg.pinv(rh_mat), largest_fa_data[i][shell_mask])
    kernel_rh_coeff = np.mean(rh_matrices, axis=0)
    response_model = RF1AnisotropicTissueResponse(kernel_rh_coeff)
    sh_model = MultiCompartmentSphericalHarmonicsModel([response_model])
    sh_fit = sh_model.fit(acquisition_scheme, largest_fa_data)
