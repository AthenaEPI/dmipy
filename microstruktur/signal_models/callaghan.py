from __future__ import division
from six.moves import range

import numpy as np
import scipy.special
import scipy.linalg


from .constants import CONSTANTS
from .gradient_conversions import q_from_g


class Disk:
    def __init__(
        self,
        N=20, K=7, epsilon=1e-10, max_dim=60,
        length=None,
        diffusion_constant=CONSTANTS['water_diffusion_constant'],
        gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
        eps=1e-20
    ):
        '''
        length is the radius
        '''
        self.eps = eps
        self.N = N
        self.K = K
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.length = length
        self.betas = np.empty((N, K))
        self.betas[0, 0] = 0
        self.betas[0, 1:] = scipy.special.jnp_zeros(0, K - 1)
        for n in range(1, N):
            self.betas[n] = scipy.special.jnp_zeros(n, K)

        Ngrid, Kgrid = np.mgrid[0:N, 0:K]
        flat_betas = self.betas.ravel()
        m_map = np.argsort(flat_betas)[:max_dim]
        self.sorted_nk = np.c_[
            Ngrid.ravel()[m_map],
            Kgrid.ravel()[m_map]
        ]
        self.selected_Ns = np.unique(self.sorted_nk[:, 0])
        self.sorted_betas = flat_betas[m_map]

    def attenuation(
        self, gradient_strength, delta, Delta,
        length=None, radius=None
    ):
        '''
        length is the radius
        '''
        if length is None:
            length = radius
        elif radius is None:
            radius = length

        if length is None:
            length = self.length
        if length is None:
            raise ValueError('Length should be specified')

        length /= 2.

        q = q_from_g(
            gradient_strength, delta,
            gyromagnetic_ratio=self.gyromagnetic_ratio
        )

        argument = 2 * np.pi * q * length

        E = np.zeros(len(q), dtype=float)
        jv1_factor = {}
        exponent = self.diffusion_constant * Delta / length ** 2
        for n in self.selected_Ns:
            jv1_factor[n] = (
                argument * scipy.special.jvp(n, argument)
            ) ** 2

        for i, nk in enumerate(self.sorted_nk):
            n, k = nk
            prefix_constant = 4.
            if n > 0:
                prefix_constant *= 2
            beta2 = self.sorted_betas[i] ** 2

            middle_factor = beta2 / (beta2 - n)
            if beta2 == 0 and n == 0:
                middle_factor = 1.
            E[:] += np.nan_to_num(
                prefix_constant *
                np.exp(-beta2 * exponent) *
                middle_factor *
                jv1_factor[n] / (
                    argument ** 2 - beta2
                ) ** 2
            )
        E[q < self.eps] = 1
        return E
