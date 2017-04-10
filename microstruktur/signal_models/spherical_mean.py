import numpy as np
from scipy.special import erf


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
