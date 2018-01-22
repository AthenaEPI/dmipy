from mipy.core.modeling_framework import ModelProperties
import numpy as np

DIAMETER_SCALING = 1e-6


class S1Dot(ModelProperties):
    r""" The Dot model [1] - an non-diffusing compartment.

    Parameters
    ----------
    no parameters

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """

    _parameter_ranges = {
    }
    _parameter_scales = {
    }
    spherical_mean = False
    _model_type = 'other'

    def __init__(self, dummy=None):
        self.dummy = dummy

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

        E_dot = np.ones(acquisition_scheme.number_of_measurements)
        return E_dot


class S2SphereSodermanApproximation(ModelProperties):
    _parameter_ranges = {
        'diameter': (1e-2, 20)
    }
    _parameter_scales = {
        'diameter': DIAMETER_SCALING
    }
    spherical_mean = False
    _model_type = 'sphere'

    def __init__(self, diameter=None):
        self.diameter = diameter

    def sphere_attenuation(self, q, diameter):
        radius = diameter / 2
        factor = 2 * np.pi * q * radius
        E = (
            3 / (factor ** 2) *
            (
                np.sin(factor) / factor -
                np.cos(factor)
            )
        ) ** 2
        return E

    def __call__(self, acquisition_scheme, **kwargs):
        q = acquisition_scheme.qvalues
        diameter = kwargs.get('diameter', self.diameter)
        E_sphere = np.ones_like(q)
        q_nonzero = q > 0  # only q>0 attenuate
        E_sphere[q_nonzero] = self.sphere_attenuation(
            q[q_nonzero], diameter)
        return E_sphere
