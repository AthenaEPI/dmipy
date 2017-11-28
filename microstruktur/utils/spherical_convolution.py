# -*- coding: utf-8 -*-
import numpy as np
from dipy.reconst.shm import real_sph_harm
from dipy.utils.optpkg import optional_package
numba, have_numba, _ = optional_package("numba")


def real_sym_rh_basis(sh_order, theta, phi):
    """Samples a real symmetric rotational harmonic basis at point on the sphere

    Samples the basis functions up to order `sh_order` at points on the sphere
    given by `theta` and `phi`. The basis functions are defined here the same
    way as in fibernavigator [1]_ where the real harmonic $Y^m_n$ is defined to
    be:

        $Y^0_n$                     if m = 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    -----------
    sh_order : int
        even int > 0, max spherical harmonic degree
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    --------
    real_rh_matrix : array of shape ()
        The real harmonic $Y^0_n$ sampled at `theta` and `phi`
    """
    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros(sh_order // 2 + 1)

    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    real_rh_matrix = real_sph_harm(m, n, theta, phi)
    return real_rh_matrix


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
    sh_order_rh = 2 * (len(kernel_rh) - 1)
    number_coef_sh = len(f_distribution_sh)
    sh_order_sh = int(-3 + np.sqrt(9 - 4 * (2 - 2 * number_coef_sh))) // 2

    sh_order_used = min(sh_order_rh, sh_order_sh)
    number_coef_used = int((sh_order_used + 2) * (sh_order_used + 1) // 2)

    f_kernel_convolved = f_distribution_sh[:number_coef_used]

    counter = 0
    for n_ in xrange(0, sh_order_used + 1, 2):
        coef_in_order = 2 * n_ + 1
        f_kernel_convolved[counter: counter + coef_in_order] *= (
            np.sqrt((4 * np.pi) / (2 * n_ + 1)) * kernel_rh[n_ // 2]
        )
        counter += coef_in_order
    return f_kernel_convolved


if have_numba:
    sh_convolution = numba.njit()(sh_convolution)
