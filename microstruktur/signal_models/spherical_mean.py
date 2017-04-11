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


def estimated_spherical_mean_shell(E_shell, bvecs_shell, sh_order=6):
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
