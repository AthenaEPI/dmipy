# -*- coding: utf-8 -*-
import numpy as np
from dipy.utils.optpkg import optional_package
numba, have_numba, _ = optional_package("numba")


def kernel_sh_to_rh(sh_coef):
    """Conversion from spherical harmonics to rotational (zonal) harmonics.
    Takes spherical harmonics Ymn with degree (n) and order (m), and returns
    Ym0.

    Parameters
    ----------
    sh_coef : array, shape (sh_coef)
        spherical harmonic coefficients.

    Returns
    -------
    rh_coef : array, shape (sh_coef)
        rotational harmonics.
    """
    number_of_coef = len(sh_coef)
    rh_coef = np.zeros_like(sh_coef)
    rh_coef[0] = sh_coef[0]
    counter = 1
    for n_ in xrange(2, 100, 2):
        coef_in_order = 2 * n_ + 1
        rh_coef[counter: counter + coef_in_order] = sh_coef[counter + n_]
        counter += coef_in_order
        if counter == number_of_coef:
            break
    return rh_coef


def sh_convolution(f_distribution_sh, kernel_rh):
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
    Ncoef_sh = len(f_distribution_sh)
    Ncoef_rh = len(kernel_rh)
    Ncoef_rel = min((Ncoef_sh, Ncoef_rh))  # relevant coefficients
    f_kernel_convolved = f_distribution_sh[:Ncoef_rel] * kernel_rh[:Ncoef_rel]
    counter = 0
    for n_ in xrange(0, 100, 2):
        coef_in_order = 2 * n_ + 1
        f_kernel_convolved[counter: counter + coef_in_order] *= (
            np.sqrt((4 * np.pi) / (2 * n_ + 1))
        )
        counter += coef_in_order
        if counter == Ncoef_rel:
            break
    return f_kernel_convolved


if have_numba:
    kernel_sh_to_rh = numba.njit()(kernel_sh_to_rh)
    sh_convolution = numba.njit()(sh_convolution)
