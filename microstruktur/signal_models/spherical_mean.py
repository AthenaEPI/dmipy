import numpy as np
from scipy.special import erf
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_mrtrix


def spherical_mean_stick(bvalue, lambda_par):
    """ Spherical mean of the signal attenuation of the Stick model [1] for a
    given b-value and parallel diffusivity. Analytic expression from Eq. (7)
    in [2].

    Parameters
    ----------
    bvalue : float,
        b-value in s/mm^2.
    lambda_par : float,
        parallel diffusivity in mm^2/s.

    Returns
    -------
    E_mean : float,
        spherical mean of the Stick model.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    .. [2] Kaden et al. "Multi-compartment microscopic diffusion imaging."
           NeuroImage 139 (2016): 346-359.
    """
    E_mean = ((np.sqrt(np.pi) * erf(np.sqrt(bvalue * lambda_par))) /
              (2 * np.sqrt(bvalue * lambda_par)))
    return E_mean


def spherical_mean_zeppelin(bvalue, lambda_par, lambda_perp):
    """ Spherical mean of the signal attenuation of the Zeppelin model e.g. [1]
    for a given b-value and parallel and perpendicular diffusivity. Analytic
    expression from Eq. (8) in [1]).

    Parameters
    ----------
    bvalue : float,
        b-value in s/mm^2.
    lambda_par : float,
        parallel diffusivity in mm^2/s.
    lambda_perp : float,
        perpendicular diffusivity in mm^2/s.

    Returns
    -------
    E_mean : float,
        spherical mean of the Zeppelin model.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    .. [2] Kaden et al. "Multi-compartment microscopic diffusion imaging."
           NeuroImage 139 (2016): 346-359.
    """
    exp_bl = np.exp(-bvalue * lambda_perp)
    sqrt_bl = np.sqrt(bvalue * (lambda_par - lambda_perp))
    E_mean = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
    return E_mean


def estimate_spherical_mean_multi_shell(E_attenuation, bvecs, b_shell_indices,
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
    E_mean = np.zeros(b_shell_indices.max())
    for b_shell_index in np.arange(1, b_shell_indices.max() + 1):  # per shell
        shell_mask = b_shell_indices == b_shell_index
        bvecs_shell = bvecs[shell_mask]
        E_shell = E_attenuation[shell_mask]
        E_mean[b_shell_index - 1] = estimate_spherical_mean_shell(E_shell,
                                                                  bvecs_shell,
                                                                  sh_order)
    return E_mean


def estimate_spherical_mean_shell(E_shell, bvecs_shell, sh_order=6):
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
    _, theta_, phi_ = cart2sphere(bvecs_shell[:, 0],
                                  bvecs_shell[:, 1],
                                  bvecs_shell[:, 2])
    sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]
    sh_mat_inv = np.linalg.pinv(sh_mat)
    E_sh_coef = np.dot(sh_mat_inv, E_shell)
    # Integral of sphere is 1 / (4 * np.pi)
    # Integral of Y00 spherical harmonic is 2 * np.sqrt(np.pi)
    # Multiplication results in normalization of 1 / (2 * np.sqrt(np.pi))
    E_mean = E_sh_coef[0] / (2 * np.sqrt(np.pi))
    return E_mean
