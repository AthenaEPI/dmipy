from ..core.modeling_framework import ModelProperties
import numpy as np

DIAMETER_SCALING = 1e-6

__all__ = [
    'S1Dot',
    'S2SphereSodermanApproximation'
]


class S1Dot(ModelProperties):
    r"""
    The Dot model [1]_ - an non-diffusing compartment.
    It has no parameters and returns 1 no matter the input.

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
    _spherical_mean = False
    _model_type = 'other'

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Calculates the signal attenation.

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
        E_dot = np.ones(acquisition_scheme.number_of_measurements)
        return E_dot

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the model for every acquisition shell.
        """
        return self(acquisition_scheme.spherical_mean_scheme, **kwargs)


class S2SphereSodermanApproximation(ModelProperties):
    r"""
    The Stejskal Tanner signal approximation of a sphere model. It assumes
    that pulse length is infinitessimally small and diffusion time large enough
    so that the diffusion is completely restricted. Only depends on q-value.

    Parameters
    ----------
    diameter : float,
        sphere diameter in meters.

    References
    ----------
    .. [1] Balinov, Balin, et al. "The NMR self-diffusion method applied to
        restricted diffusion. Simulation of echo attenuation from molecules in
        spheres and between planes." Journal of Magnetic Resonance, Series A
        104.1 (1993): 17-25.
    """
    _parameter_ranges = {
        'diameter': (1e-2, 20)
    }
    _parameter_scales = {
        'diameter': DIAMETER_SCALING
    }
    _spherical_mean = False
    _model_type = 'sphere'

    def __init__(self, diameter=None):
        self.diameter = diameter

    def sphere_attenuation(self, q, diameter):
        "The signal attenuation for the sphere model."
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
        r'''
        Calculates the signal attenation.

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
        q = acquisition_scheme.qvalues
        diameter = kwargs.get('diameter', self.diameter)
        E_sphere = np.ones_like(q)
        q_nonzero = q > 0  # only q>0 attenuate
        E_sphere[q_nonzero] = self.sphere_attenuation(
            q[q_nonzero], diameter)
        return E_sphere

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the model for every acquisition shell.
        """
        return self(acquisition_scheme.spherical_mean_scheme, **kwargs)
