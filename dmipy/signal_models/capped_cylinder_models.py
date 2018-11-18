from . import cylinder_models, plane_models
from ..utils import utils
from ..core.modeling_framework import ModelProperties
from ..core.signal_model_properties import AnisotropicSignalModelProperties
from ..core.constants import CONSTANTS
import numpy as np

__all__ = [
    'CC3CappedCylinderCallaghanApproximation',
]


class CC2CappedCylinderStejskalTannerApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Stejskal-Tanner model for intra-cylindrical diffusion inside
    a capped cylinder with finite radius and length. The perpendicular
    diffusion is modelled after Soderman's solution for the disk [1]_. The
    parallel diffusion between planes has been implemented according to
    Balinov [2]_.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    diameter : float,
        capped cylinder (axon) diameter in meters.
    length : float,
        capped cylinder length in meters.

    References
    ----------
    .. [1] Soderman, Olle, and Bengt Jonsson. "Restricted diffusion in
        cylindrical geometry." Journal of Magnetic Resonance, Series A 117.1
        (1995): 94-97.
    .. [2] Balinov, Balin, et al. "The NMR self-diffusion method applied to
        restricted diffusion. Simulation of echo attenuation from molecules in
        spheres and between planes." Journal of Magnetic Resonance, Series A
        104.1 (1993): 17-25.
    """
    _required_acquisition_parameters = ['gradient_directions', 'qvalues']
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None,
        diameter=None,
        length=None,
    ):
        self.mu = mu
        self.diameter = diameter
        self.length = length

        self._cylinder_model = (
            cylinder_models.C2CylinderStejskalTannerApproximation(
                mu=self.mu,
                diameter=self.diameter))
        self._plane_model = plane_models.P2PlaneStejskalTannerApproximation(
            diameter=length)

        self._parameter_ranges = self._cylinder_model._parameter_ranges.copy()
        self._parameter_ranges.update(self._plane_model._parameter_ranges)

        self._parameter_scales = self._cylinder_model._parameter_scales.copy()
        self._parameter_scales.update(self._plane_model._parameter_scales)

        self._parameter_types = self._cylinder_model._parameter_types.copy()
        self._parameter_types.update(self._plane_model._parameter_types)

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        n = acquisition_scheme.gradient_directions
        q = acquisition_scheme.qvalues

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
            q_parallel[q_nonzero], length)

        E_perpendicular = np.ones_like(q)
        q_perp = q * magnitude_perpendicular
        q_nonzero = q_perp > 0
        E_perpendicular[q_nonzero] = (
            self._cylinder_model.perpendicular_attenuation(
                q_perp[q_nonzero], diameter)
        )
        return E_parallel * E_perpendicular


class CC3CappedCylinderCallaghanApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Callaghan model [1]_ - a cylinder with finite radius - for
    intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk. The parallel diffusion of the
    capped cylinder is modelled using the same Callaghan approximation but
    between two parallel planes with a certain distance or 'length' between
    them.

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
    diffusion_intra : float,
        The diffusion constant of the water particles inside the cylinder.
        The default value is the approximate diffusivity of water inside axons
        as 1.7e-9 m^2/s.
    number_of_roots_cylinder : integer,
        number of roots for the cylinder Callaghan approximation.
    number_of_functions_cylinder : integer,
        number of functions for the cylinder Callaghan approximation.
    number_of_roots_plane : integer,
        number of roots for the plane Callaghan approximation.

    References
    ----------
    .. [1] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
        cylindrical, and spherical pores under conditions of wall
        relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
        53-59.
    """
    _required_acquisition_parameters = [
        'gradient_directions', 'qvalues', 'tau']
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        mu=None,
        diameter=None,
        length=None,
        diffusion_intra=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots_cylinder=20,
        number_of_functions_cylinder=50,
        number_of_roots_plane=40
    ):
        self.mu = mu
        self.diameter = diameter
        self.length = length
        self.diffusion_intra = diffusion_intra
        self.number_of_roots_cylinder = number_of_roots_cylinder
        self.number_of_functions_cylinder = number_of_functions_cylinder
        self.number_of_roots_plane = number_of_roots_plane

        self._cylinder_model = (
            cylinder_models.C3CylinderCallaghanApproximation(
                mu=self.mu,
                diameter=self.diameter,
                diffusion_perpendicular=self.diffusion_intra,
                number_of_roots=self.number_of_roots_cylinder,
                number_of_functions=self.number_of_functions_cylinder)
        )
        self._plane_model = plane_models.P3PlaneCallaghanApproximation(
            diameter=length,
            diffusion_constant=self.diffusion_intra,
            n_roots=self.number_of_roots_plane)

        self._parameter_ranges = self._cylinder_model._parameter_ranges.copy()
        self._parameter_ranges.update(self._plane_model._parameter_ranges)

        self._parameter_scales = self._cylinder_model._parameter_scales.copy()
        self._parameter_scales.update(self._plane_model._parameter_scales)

        self._parameter_types = self._cylinder_model._parameter_types.copy()
        self._parameter_types.update(self._plane_model._parameter_types)

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenuation.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

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
