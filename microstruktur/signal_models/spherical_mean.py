import numpy as np
from scipy.special import erf
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_mrtrix


def spherical_mean_stick(b, lambda_par):
    """ Spherical mean of a stick, see (Eq. (7) in [1])
    """
    E_mean = ((np.sqrt(np.pi) * erf(np.sqrt(b * lambda_par))) /
              (2 * np.sqrt(b * lambda_par)))
    return E_mean


def spherical_mean_zeppelin(b, lambda_par, lambda_perp):
    """ Spherical mean of a Zeppelin, see (Eq. (8) in [1])
    """
    exp_bl = np.exp(-b * lambda_perp)
    sqrt_bl = np.sqrt(b * (lambda_par - lambda_perp))
    E_mean = exp_bl * np.sqrt(np.pi) * erf(sqrt_bl) / (2 * sqrt_bl)
    return E_mean


def estimated_spherical_mean_shell(E_shell, bvecs_shell, sh_order=6):
    """ Estimates spherical mean of a shell of measurements using
    spherical harmonics.
    The spherical mean is contained only in the Y00 spherical harmonic, as long
    as the basis expansion is sufficient to capture the spherical signal.
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
