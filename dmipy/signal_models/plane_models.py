
import numpy as np
from ..core.modeling_framework import ModelProperties
from ..core.constants import CONSTANTS

LENGTH_SCALING = 1e-6


class P3PlaneCallaghanApproximation(ModelProperties):
    r""" The Callaghan model [1] of diffusion between two parallel infinite plates.

        Parameters
        ----------
        length : float
            distance between the two plates in meters.

        References
        ----------
        [1] Callaghan, "Pulsed-Gradient Spin-Echo NMR for Planar, Cylindrical,
        and Spherical Pores under Conditions of Wall Relaxation", JMR 1995
        """

    _parameter_ranges = {
        'length': (1e-2, 20)
    }

    _parameter_scales = {
        'length': LENGTH_SCALING
    }

    spherical_mean = False
    _model_type = 'plane'

    def __init__(
        self,
        length=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
        n_roots=40,
    ):

        self.length = length
        self.Dintra = diffusion_constant
        self.xi = np.arange(n_roots) * np.pi
        self.zeta = np.arange(n_roots) * np.pi + np.pi / 2.0

    def plane_attenuation(self, q, tau, length):
        """Implements the finite time Callaghan model for planes [1]
        """
        radius = length / 2.0
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)
        for n in range(len(self.xi)):
            xi_n = self.xi[n]
            xi_n2 = self.xi[n] ** 2

            if xi_n == 0.:
                div = 1.
            else:
                div = np.sin(2 * xi_n) / 2 * xi_n

            update = (
                2 * np.exp(-xi_n2 * self.Dintra * tau / radius ** 2) /
                (1 + div) *
                (q_argument * np.sin(q_argument) * np.cos(xi_n) - xi_n *
                 np.cos(q_argument) * np.sin(xi_n)) ** 2 /
                (q_argument_2 - xi_n2) ** 2
            )

            update[~np.isfinite(update)] = 0.

            res += update

        for m in xrange(len(self.zeta)):
            zeta_m = self.zeta[m]
            zeta_m2 = self.zeta[m] ** 2

            update = (
                2 * np.exp(-zeta_m2 * self.Dintra * tau / radius ** 2) /
                (1 - np.sin(2 * zeta_m) / (2 * zeta_m)) *
                (q_argument * np.cos(q_argument) * np.sin(zeta_m) - zeta_m *
                 np.sin(q_argument) * np.cos(zeta_m)) ** 2 /
                (q_argument_2 - zeta_m2) ** 2
            )

            update[~np.isfinite(update)] = 0.
            res += update
        return res

    def __call__(self, acquisition_scheme, **kwargs):
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau
        length = kwargs.get('length', self.length)

        E_plane = np.ones_like(q)
        q_nonzero = q > 0
        E_plane[q_nonzero] = self.plane_attenuation(
            q[q_nonzero], tau[q_nonzero], length
        )
        return E_plane
