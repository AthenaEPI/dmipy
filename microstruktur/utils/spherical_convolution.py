# -*- coding: utf-8 -*-
import numpy as np
from dipy.reconst.shm import sph_harm_ind_list


def kernel_sh_to_rh(sh_coef, sh_order):
    """Conversion from spherical harmonics to rotational (zonal) harmonics.
    Takes spherical harmonics Ymn with degree (n) and order (m), and returns
    Ym0.

    Parameters
    ----------
    sh_coef : array, shape (sh_coef)
        spherical harmonic coefficients.
    sh_order : integer,
        maximum spherical harmonics order.

    Returns
    -------
    rh_coef : array, shape (sh_coef)
        rotational harmonics.
    """
    m, n = sph_harm_ind_list(sh_order)
    rh_coef = np.zeros_like(sh_coef)
    rh_coef[0] = sh_coef[0]
    counter = 1
    for n_ in xrange(2, sh_order + 1, 2):
        coef_in_order = 2 * n_ + 1
        rh_coef[counter: counter + coef_in_order] = sh_coef[counter + n_]
        counter += coef_in_order
    return rh_coef


def sh_convolution(f_distribution_sh, kernel_rh, sh_order):
    """Spherical convolution between a fiber distribution (f) in spherical
    harmonics and a kernel in terms of rotational harmonics (oriented along the
    z-axis).

    Parameters
    ----------
    f_distribution_sh : array, shape (sh_coef)
        spherical harmonic coefficients of a fiber distribution.
    kernel_rh : array, shape (sh_coef),
        rotational harmonic coefficients of the convolution kernel. In our case
        this is the spherical signal of one micro-environment at one b-value.

    Returns
    -------
    f_kernel_convolved : array, shape (sh_coef)
        spherical harmonic coefficients of the convolved kernel and
        distribution.
    """
    m, n = sph_harm_ind_list(sh_order)
    lambda_ = np.sqrt((4 * np.pi) / (2 * n + 1))
    f_kernel_convolved = lambda_ * f_distribution_sh * kernel_rh
    return f_kernel_convolved
