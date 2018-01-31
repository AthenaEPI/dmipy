from . import cylinder_models, plane_models
from ..utils import utils
from ..core.modeling_framework import ModelProperties
import numpy as np


class CC3CappedCylinderCallaghanApproximation(ModelProperties):
    r""" The Callaghan model [1]_ - a cylinder with finite radius - for
    intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    diameter : float,
        cylinder (axon) diameter in meters.
    length : float,
        cylinder length in meters.


    References
    ----------
    .. [1]_ Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
            cylindrical, and spherical pores under conditions of wall
            relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
            53-59.
    """
    _cylinder_model = cylinder_models.C3CylinderCallaghanApproximation()
    _plane_model = plane_models.P3PlaneCallaghanApproximation()

    _parameter_ranges = _cylinder_model._parameter_ranges.copy()
    _parameter_ranges.update(_plane_model._parameter_ranges)

    _parameter_scales = _cylinder_model._parameter_scales.copy()
    _parameter_scales.update(_plane_model._parameter_scales)

    spherical_mean = False
    _model_type = 'experimental'

    def __init__(
        self,
        mu=None,
        diameter=None,
        length=None,
    ):
        self.mu = mu
        self.diameter = diameter
        self.length = length

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        n = acquisition_scheme.gradient_directions
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau

        diameter = kwargs.get('diameter', self.diameter)
        length = kwargs.get('length', self.length)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        q_parallel = q * np.dot(n, mu)
        E_parallel = np.ones_like(q)
        q_nonzero = q_parallel > 0
        E_parallel[q_nonzero] = self._plane_model.plane_attenuation(
            q_parallel[q_nonzero], tau[q_nonzero], length)
        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular

        q_nonzero = q_perp > 0
        E_perpendicular[q_nonzero] = (
            self._cylinder_model.perpendicular_attenuation(
                q_perp[q_nonzero], tau[q_nonzero], diameter)
        )
        return E_parallel * E_perpendicular
