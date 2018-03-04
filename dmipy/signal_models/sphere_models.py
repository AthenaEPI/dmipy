from ..core.modeling_framework import ModelProperties
from ..core.constants import CONSTANTS
from scipy import special
import numpy as np

DIAMETER_SCALING = 1e-6

__all__ = [
    'S1Dot',
    'S2SphereStejskalTannerApproximation'
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
    _parameter_types = {
    }
    _model_type = 'CompartmentModel'

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

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped), 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[0],
                    E_reshaped[i])
            )
        return rh_array

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


class S2SphereStejskalTannerApproximation(ModelProperties):
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
    _parameter_types = {
        'diameter': 'sphere',
    }
    _model_type = 'CompartmentModel'

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

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped), 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[0],
                    E_reshaped[i])
            )
        return rh_array

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


class S3SphereCallaghanApproximation(ModelProperties):
    r"""
    The Callaghan model [1]_ of diffusion inside a sphere.

    Parameters
    ----------
    diameter : float
        Distance between the two plates in meters.
    diffusion_constant : float,
        The diffusion constant of the water particles between the two planes.
        The default value is the approximate diffusivity of water inside axons
        as 1.7e-9 m^2/s.
    number_of_roots : integer,
        The number of roots for the Callaghan approximation.

    References
    ----------
    [1] Callaghan, "Pulsed-Gradient Spin-Echo NMR for Planar, Cylindrical,
        and Spherical Pores under Conditions of Wall Relaxation", JMR 1995
    """

    _parameter_ranges = {
        'diameter': (1e-2, 20)
    }

    _parameter_scales = {
        'diameter': DIAMETER_SCALING
    }

    _parameter_types = {
        'diameter': 'sphere'
    }
    _model_type = 'CompartmentModel'

    def __init__(
        self,
        diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots=20,
        number_of_functions=50,
    ):

        self.diameter = diameter
        self.Dintra = diffusion_constant
        self.alpha = np.empty((number_of_roots, number_of_functions))
        self.alpha[0, 0] = 0
        if number_of_roots > 1:
            self.alpha[1:, 0] = special.jnp_zeros(0, number_of_roots - 1)
        for m in range(1, number_of_functions):
            self.alpha[:, m] = special.jnp_zeros(m, number_of_roots)

    def sphere_attenuation(self, q, tau, diameter):
        """Implements the finite time Callaghan model for planes."""
        radius = diameter / 2.0
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)

        J = special.spherical_jn(q_argument)
        Jder = special.spherical_jn(q_argument, derivative=True)
        for k in range(0, self.alpha.shape[0]):
            for n in range(0, self.alpha.shape[1]):
                a_nk2 = self.alpha[k, n] ** 2
                update = np.exp(-a_nk2 * self.Dintra * tau / radius ** 2)
                update *= ((2 * n + 1) * a_nk2) / \
                    (a_nk2 - (n - 0.5) ** 2 + 0.25)
                update *= q_argument * Jder
                update /= (q_argument_2 - a_nk2) ** 2
                res += update
        return res

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
        q = acquisition_scheme.qvalues
        tau = acquisition_scheme.tau
        diameter = kwargs.get('diameter', self.diameter)

        E_sphere = np.ones_like(q)
        q_nonzero = q > 0
        E_sphere[q_nonzero] = self.sphere_attenuation(
            q[q_nonzero], tau[q_nonzero], diameter
        )
        return E_sphere


class S4SphereGaussianPhaseApproximation(ModelProperties):
    r"""
    The gaussian phase approximation for diffusion inside a sphere according
    to [1]_. It is dependent on gradient strength, pulse separation and pulse
    length.

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
    _parameter_types = {
        'diameter': 'sphere'
    }
    _model_type = 'CompartmentModel'

    # According to Balinov et al., solutions of
    # 1/(alpha * R) * J(3/2,alpha * R) = J(5/2, alpha * R)
    # with R = 1 with alpha * R < 100 * pi
    SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035,
        100.5110653, 103.6532613, 106.7954217, 109.9375497,
        113.0796480, 116.2217188, 119.3637645, 122.5057870,
        125.6477880, 128.7897690, 131.9317315, 135.0736768,
        138.2156061, 141.3575204, 144.4994207, 147.6413080,
        150.7831829, 153.9250463, 157.0668989, 160.2087413,
        163.3505741, 166.4923978, 169.6342129, 172.7760200,
        175.9178194, 179.0596116, 182.2013968, 185.3431756,
        188.4849481, 191.6267147, 194.7684757, 197.9102314,
        201.0519820, 204.1937277, 207.3354688, 210.4772054,
        213.6189378, 216.7606662, 219.9023907, 223.0441114,
        226.1858287, 229.3275425, 232.4692530, 235.6109603,
        238.7526647, 241.8943662, 245.0360648, 248.1777608,
        251.3194542, 254.4611451, 257.6028336, 260.7445198,
        263.8862038, 267.0278856, 270.1695654, 273.3112431,
        276.4529189, 279.5945929, 282.7362650, 285.8779354,
        289.0196041, 292.1612712, 295.3029367, 298.4446006,
        301.5862631, 304.7279241, 307.8695837, 311.0112420,
        314.1528990
    ]

    def __init__(
        self, diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    ):
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self.diameter = diameter

    def sphere_attenuation(
        self, gradient_strength, delta, Delta, diameter
    ):
        "Calculates the sphere signal attenuation."

        D = self.diffusion_constant
        gamma = self.gyromagnetic_ratio
        radius = diameter / 2

        alpha = self.SPHERE_TRASCENDENTAL_ROOTS / radius
        alpha2 = alpha ** 2
        alpha2D = alpha2 * D

        first_factor = -2 * (gamma * gradient_strength) ** 2 / D
        summands = (
            alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
            (
                2 * delta - (
                    2 +
                    np.exp(-alpha2D * (Delta - delta)) -
                    2 * np.exp(-alpha2D * delta) -
                    2 * np.exp(-alpha2D * Delta) +
                    np.exp(-alpha2D * (Delta + delta))
                ) / (alpha2D)
            )
        )
        E = np.exp(
            first_factor *
            summands.sum()
        )
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
        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        diameter = kwargs.get('diameter', self.diameter)
        E_sphere = np.ones_like(g)

        g_nonzero = g > 0
        # for every unique combination get the perpendicular attenuation
        unique_deltas = np.unique([acquisition_scheme.shell_delta,
                                   acquisition_scheme.shell_Delta], axis=1)
        for delta_, Delta_ in zip(*unique_deltas):
            mask = np.all([g_nonzero, delta == delta_, Delta == Delta_],
                          axis=0)
            E_sphere[mask] = self.sphere_attenuation(
                g[mask], delta_, Delta_, diameter
            )
        return E_sphere

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped), 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[0],
                    E_reshaped[i])
            )
        return rh_array

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
