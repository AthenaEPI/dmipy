import numpy as np
from dipy.reconst.shm import real_sym_sh_mrtrix
from microstruktur.utils import utils


def estimate_spherical_mean_multi_shell(E_attenuation, acquisition_scheme,
                                        sh_order=6):
    r""" Estimates the spherical mean per shell of multi-shell acquisition.
    Uses spherical harmonics to do the estimation.

    Parameters
    ----------
    E_attenuation : array, shape (N),
        signal attenuation.
    bvecs : array, shape (N x 3),
        x, y, z components of cartesian unit b-vectors.
    b_shell_indices : array, shape (N)
        array of integers indicating which measurement belongs to which shell.
        0 should be used for b0 measurements, 1 for the lowest b-value shell,
        2 for the second lowest etc.

    Returns
    -------
    E_mean : array, shape (number of b-shells)
        spherical means of the signal attenuation per shell of the. For
        example, if there are three shells in the acquisition then the array
        is of length 3.
    """
    shell_indices = acquisition_scheme.shell_indices
    gradient_directions = acquisition_scheme.gradient_directions
    unique_dwi_indices = acquisition_scheme.unique_dwi_indices

    E_mean = np.ones(len(acquisition_scheme.shell_bvalues))
    for b_shell_index in unique_dwi_indices:  # per shell
        sh_mat = acquisition_scheme.shell_sh_matrices[b_shell_index]
        shell_mask = shell_indices == b_shell_index
        bvecs_shell = gradient_directions[shell_mask]
        E_shell = E_attenuation[shell_mask]
        E_mean[b_shell_index] = estimate_spherical_mean_shell(
            E_shell, bvecs_shell, sh_order, sh_mat)
    return E_mean


def estimate_spherical_mean_shell(E_shell, bvecs_shell, sh_order=6, sh_mat=None):
    """ Estimates spherical mean of a shell of measurements using
    spherical harmonics.
    The spherical mean is contained only in the Y00 spherical harmonic, as long
    as the basis expansion order is sufficient to capture the spherical signal.

    Parameters
    ----------
    E_shell : array, shape(N),
        signal attenuation values.
    bvecs_shell : array, shape (N x 3),
        Cartesian unit vectors describing the orientation of the signal
        attenuation values.
    sh_order : integer,
        maximum spherical harmonics order. It needs to be high enough to
        describe the spherical profile of the signal attenuation. The order 6
        is sufficient to describe a stick at b-values up to 10,000 s/mm^2.

    Returns
    -------
    E_mean : float,
        spherical mean of the signal attenuation.
    """
    if sh_mat is None:
        _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]
    sh_mat_inv = np.linalg.pinv(sh_mat)
    E_sh_coef = np.dot(sh_mat_inv, E_shell)
    # Integral of sphere is 1 / (4 * np.pi)
    # Integral of Y00 spherical harmonic is 2 * np.sqrt(np.pi)
    # Multiplication results in normalization of 1 / (2 * np.sqrt(np.pi))
    E_mean = E_sh_coef[0] / (2 * np.sqrt(np.pi))
    return E_mean
