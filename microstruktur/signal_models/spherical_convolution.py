import numpy as np
from dipy.reconst.shm import sph_harm_ind_list


def kernel_sh_to_rh(sh_coef, sh_order):
    m, n = sph_harm_ind_list(sh_order)
    k_rh = np.zeros_like(sh_coef)
    for i, m_ in enumerate(m):
        if m_ == 0:
            k_rh[i] = sh_coef[i]
        else:
            k_rh[i] = sh_coef[np.all([n == n[i], m == 0], axis=0)]
    return k_rh


def sh_convolution(f_distribution_sh, kernel_rh, sh_order):
    m, n = sph_harm_ind_list(sh_order)
    lambda_ = np.sqrt((4 * np.pi) / (2 * n + 1))
    f_kernel_convolved = lambda_ * f_distribution_sh * kernel_rh
    return f_kernel_convolved
