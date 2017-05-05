# -*- coding: utf-8 -*-
from dipy.reconst.shm import real_sym_sh_mrtrix
import numpy as np
from . import three_dimensional_models, spherical_convolution, utils
BINGHAM_SH_ORDER = 14


def disperse_data(E, bvecs, shell_indices, mu, psi, kappa, beta):
    """Function to disperse micro-environment (multi-shell) data using a
    Bingham distribution. We use it to disperse data that was simulated using
    Camino [1], which only simulates parallel cylinders. We use convolution
    using spherical harmonics to make this function as general as possible.

    Parameters
    ----------
    E : array, shape (N)
        signal attenuation of a single-shell or multi-shell acquisition of the
        simulated micro-environment substrate. Will be used as kernel.
        data must be aligned with the z-axis, i.e. mu = (0, 0)
    bvecs : array, shape (Nx3)
        gradient orientations associated with E.
    shell_indices : array, shape (N)
        index of the shell that is associated with each gradient direction.
        0 for b0 measurement. [1, 2, ... etc] for subsequent b-shells.
    mu : tuple
        desired orientation of dispersed micro-environment in radians.
        (theta, phi) ranges are [0, pi] and [-pi, pi].
    psi : float
        rotation of the distribution about the axis mu. Only has an influence
        if beta>0. range is [0, pi].
    kappa : float
        principal concentration parameter. range is [0, 16] for realistic
        tissue dispersion [2].
    beta : float
        secondary concentration parameter for anisotropic dispersion.
        in general beta<kappa should be true [2].

    Returns
    -------
    E_dispersed : array, shape (N)
        dispersed signal attenuation of micro-environment data E.

    References
    ----------
    .. [1] Cook, P. A., et al. "Camino: open-source diffusion-MRI
        reconstruction and processing." ISMRM Vol. 2759. Seattle WA, USA, 2006.
    .. [2] Tariq, Maira, et al. "Bingham–NODDI: Mapping anisotropic orientation
        dispersion of neurites using diffusion MRI." NeuroImage 133 (2016):
        207-223.
    """
    sh_order = BINGHAM_SH_ORDER
    bingham = three_dimensional_models.SD2Bingham(mu, psi, kappa, beta)
    sh_bingham = bingham.spherical_harmonics_representation(sh_order=sh_order)
    E_dispersed = np.ones_like(E)
    for shell_index in np.arange(1, shell_indices.max() + 1):  # per shell
        shell_mask = shell_indices == shell_index
        bvecs_shell = bvecs[shell_mask]  # what bvecs in that shell
        data_shell = E[shell_mask]
        _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]
        sh_data = np.dot(np.linalg.pinv(sh_mat), data_shell)
        rh_data = spherical_convolution.kernel_sh_to_rh(sh_data, sh_order)
        # convolving micro-environment with bingham distribution
        E_dispersed_sh = spherical_convolution.sh_convolution(
            sh_bingham, rh_data, sh_order
        )
        # recover signal values from convolved spherical harmonics
        E_dispersed[shell_mask] = np.dot(sh_mat, E_dispersed_sh)
    return E_dispersed
