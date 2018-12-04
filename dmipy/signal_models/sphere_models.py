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
    _required_acquisition_parameters = []

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
    _required_acquisition_parameters = ['qvalues']

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


class _S3SphereCallaghanApproximation(ModelProperties):
    r"""
    The Callaghan model [1]_ of diffusion inside a sphere.

    Parameters
    ----------
    diameter : float
        Diameter of the sphere in meters.
    diffusion_constant : float,
        The diffusion constant of the water particles in the sphere.
        The default value is the approximate diffusivity of water inside axons
        as 1.7e-9 m^2/s.
    number_of_roots : integer,
        The number of roots for the Callaghan approximation.

    References
    ----------
    [1] Callaghan, "Pulsed-Gradient Spin-Echo NMR for Planar, Cylindrical,
        and Spherical Pores under Conditions of Wall Relaxation", JMR 1995
    """
    _required_acquisition_parameters = ['qvalues', 'tau']

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

    SPHERICAL_BESSEL_ROOTS = np.r_['1,2,0',
        [0., 4.49341, 7.72525, 10.90412, 14.06619, 17.22076, 20.37130,
        23.51945, 26.66605, 29.81160, 32.95639, 36.10062, 39.24443, 42.38791,
        45.53113, 48.67414, 51.81698, 54.95968, 58.10225, 61.24473],
        [2.08158, 5.94037, 9.20584, 12.40445, 15.57924, 18.74265, 21.89970,
        25.05283, 28.20336, 31.35209, 34.49951, 37.64596, 40.79166, 43.93676,
        47.08140, 50.22565, 53.36959, 56.51327, 59.65673, 62.80000],
        [3.34209, 7.28993, 10.61386, 13.84611, 17.04290, 20.22186, 23.39049,
        26.55259, 29.71028, 32.86485, 36.01714, 39.16769, 42.31690, 45.46505,
        48.61235, 51.75896, 54.90498, 58.05053, 61.19567, 64.34046],
        [4.51410, 8.58375, 11.97273, 15.24451, 18.46815, 21.66661, 24.85009,
        28.02387, 31.19102, 34.35339, 37.51223, 40.66836, 43.82239, 46.97473,
        50.12572, 53.27559, 56.42453, 59.57269, 62.72019, 65.86712],
        [5.64670, 9.84045, 13.29556, 16.60935, 19.86242, 23.08280, 26.28327,
        29.47064, 32.64889, 35.82054, 38.98723, 42.15011, 45.30998, 48.46745,
        51.62297, 54.77687, 57.92942, 61.08083, 64.23127, 67.38089],
        [6.75646, 11.07021, 14.59055, 17.94718, 21.23107, 24.47483, 27.69372,
        30.89600, 34.08660, 37.26863, 40.44419, 43.61472, 46.78129, 49.94465,
        53.10539, 56.26396, 59.42071, 62.57592, 65.72981, 68.88258],
        [7.85108, 12.27933, 15.86322, 19.26271, 22.57806, 25.84608,29.08435,
        32.30249, 35.50633, 38.69960, 41.88481, 45.06374, 48.23767, 51.40755,
        54.57410, 57.73788, 60.89934, 64.05881, 67.21660, 70.37291],
        [8.93484, 13.47203, 17.11751, 20.55943, 23.90645, 27.19925, 30.45750,
        33.69217, 36.90992, 40.11507, 43.31056, 46.49848, 49.68033, 52.85725,
        56.03010, 59.19955, 62.36614, 65.53029, 68.69233, 71.85256],
        [10.01037, 14.65126, 18.35632, 21.84001, 25.21865, 28.53646, 31.81511,
        35.06676, 38.29892, 41.51645, 44.72271, 47.92008, 51.11031, 54.29470,
        57.47426, 60.64978, 63.82187, 66.99103, 70.15767, 73.32212],
        [11.07942, 15.81922, 19.58189, 23.10657, 26.51660, 29.85949, 33.15875,
        36.42772, 39.67463, 42.90493, 46.12234, 49.32955, 52.52853, 55.72075,
        58.90737, 62.08928, 65.26718, 68.44166, 71.61318, 74.78213],
        [12.14320, 16.97755, 20.79597, 24.36079, 27.80189, 31.16981, 34.48979,
        37.77627, 41.03820, 44.28155, 47.51042, 50.72778, 53.93581, 57.13616,
        60.33013, 63.51871, 66.70270, 69.88276, 73.05941, 76.23310],
        [13.20262, 18.12756, 21.99996, 25.60406, 29.07582, 32.46863, 35.80937,
        39.11347, 42.39061, 45.64722, 48.88778, 52.11553, 55.33286, 58.54160,
        61.74316, 64.93865, 68.12897, 71.31484, 74.49684, 77.67548],
        [14.25834, 19.27029, 23.19500, 26.83752, 30.33950, 33.75703, 37.11847,
        40.44026, 43.73271, 47.00274, 50.25518, 53.49351, 56.72035, 59.93768,
        63.14704, 66.34966, 69.54650, 72.73837, 75.92592, 79.10969],
        [15.31089, 20.40658, 24.38204, 28.06214, 31.59389, 35.03588, 38.41795,
        41.75743, 45.06527, 48.34883, 51.61327, 54.86235, 58.09886, 61.32495,
        64.54229, 67.75220, 70.95575, 74.15379, 77.34706, 80.53613],
        [16.36067, 21.53712, 25.56187, 29.27873, 32.83978, 36.30598, 39.70855,
        43.06568, 46.38894, 49.68610, 52.96266, 56.22260, 59.46892, 62.70392,
        65.92938, 69.14673, 72.35713, 75.56150, 78.76063, 81.95515],
        [17.40803, 22.66249, 26.73518, 30.48800, 34.07787, 37.56801, 40.99093,
        44.36566, 47.70433, 51.01514, 54.30388, 57.57477, 60.83101, 64.07502,
        67.30873, 70.53366, 73.75104, 76.96187, 80.16699, 83.36708],
        [18.45324, 23.78319, 27.90253, 31.69056, 35.30877, 38.82256, 42.26566,
        45.65791, 49.01197, 52.33644, 55.63740, 58.91932, 62.18555, 65.43869,
        68.68075, 71.91336, 75.13782, 78.35523, 81.56645, 84.77224],
        [19.49652, 24.89964, 29.06442, 32.88693, 36.53303, 40.07017, 43.53327,
        46.94293, 50.31233, 53.65046, 56.96368, 60.25667, 63.53297, 66.79529,
        70.04579, 73.28617, 76.51782, 79.74189, 82.95932, 86.17090],
        [20.53807, 26.01219, 30.22129, 34.07758, 37.75111, 41.31131, 44.79422,
        48.22117, 51.60586, 54.95762, 58.28311, 61.58720, 64.87361, 68.14518,
        71.40418, 74.65242, 77.89134, 81.12216, 84.34589, 87.56334],
        [21.57805, 27.12116, 31.37352, 35.26292, 38.96344, 42.54640, 46.04892,
        49.49304, 52.89294, 56.25830, 59.59605, 62.91126, 66.20781, 69.48869,
        72.75625, 76.01240, 79.25866, 82.49631, 85.72640, 88.94981],
        [22.61660, 28.22684, 32.52145, 36.44331, 40.17039, 43.77581, 47.29775,
        50.75890, 54.17393, 57.55285, 60.90284, 64.22918, 67.53589, 70.82610,
        74.10227, 77.36638, 80.62004, 83.86458, 87.10110, 90.33054],
        [23.65384, 29.32945, 33.66537, 37.61908, 41.37230, 44.99989, 48.54104,
        52.01909, 55.44916, 58.84158, 62.20378, 65.54124, 68.85813, 72.15770,
        75.44251, 78.71463, 81.97573, 85.22721, 88.47023, 91.70575],
        [24.68987, 30.42923, 34.80556, 38.79053, 42.56948, 46.21895, 49.77911,
        53.27391, 56.71892, 60.12478, 63.49916, 66.84773, 70.17480, 73.48374,
        76.77722, 80.05737, 83.32596, 86.58443, 89.83399, 93.07564],
        [25.72479, 31.52635, 35.94224, 39.95791, 43.76220, 47.43327, 51.01223,
        54.52365, 57.98348, 61.40272, 64.78924, 68.14889, 71.48613, 74.80446,
        78.10662, 81.39484, 84.67095, 87.93644, 91.19258, 94.44042],
        [26.75869, 32.62100, 37.07563, 41.12147, 44.95071, 48.64310, 52.24066,
        55.76854, 59.24311, 62.67565, 66.07425, 69.44496, 72.79237, 76.12008,
        79.43093, 82.72724, 86.01089, 89.28344, 92.54620, 95.80026],
        [27.79162, 33.71332, 38.20594, 42.28142, 46.13524, 49.84869, 53.46464,
        57.00884, 60.49803, 63.94379, 67.35443, 70.73616, 74.09372, 77.43080,
        80.75036, 84.05477, 87.34598, 90.62562, 93.89502, 97.15534],
        [28.82366, 34.80345, 39.33333, 43.43796, 47.31601, 51.05024, 54.68440,
        58.24476, 61.74846, 65.20736, 68.62998, 72.02270, 75.39037, 78.73682,
        82.06508, 85.37760, 88.67639, 91.96313, 95.23921, 98.50581],
        [29.85486, 35.89153, 40.45797, 44.59127, 48.49319, 52.24795, 55.90011,
        59.47649, 62.99460, 66.46656, 69.90110, 73.30476, 76.68253, 80.03833,
        83.37528, 86.69591, 90.00230, 93.29615, 96.57892, 99.85183],
        [30.88528, 36.97766, 41.58000, 45.74151, 49.66698, 53.44201, 57.11199,
        60.70424, 64.23664, 67.72156, 71.16796, 74.58252, 77.97035, 81.33549,
        84.68113, 88.00987, 91.32385, 94.62483, 97.91432, 101.19355],
        [31.91497, 38.06196, 42.69956, 46.88884, 50.83753, 54.63259, 58.32019,
        61.92816, 65.47474, 68.97255, 72.43074, 75.85615, 79.25401, 82.62846,
        85.98277, 89.31962, 92.64120, 95.94932, 99.24553, 102.53111],
        [32.94396, 39.14452, 43.81678, 48.03340, 52.00499, 55.81984, 59.52487,
        63.14843, 66.70906, 70.21968, 73.68959, 77.12580, 80.53366, 83.91739,
        87.28037, 90.62531, 93.95448, 97.26975, 100.57269, 103.86463],
        [33.97230, 40.22542, 44.93175, 49.17532, 53.16950, 57.00390, 60.72619,
        64.36518, 67.93977, 71.46310, 74.94467, 78.39162, 81.80944, 85.20243,
        88.57404, 91.92708, 95.26384, 98.58625, 101.89593, 105.19423],
        [35.00002, 41.30475, 46.04460, 50.31471, 54.33119, 58.18492, 61.92429,
        65.57857, 69.16700, 72.70295, 76.19611, 79.65376, 83.08149, 86.48371,
        89.86394, 93.22505, 96.56940, 99.89895, 103.21537, 106.52005],
        [36.02715, 42.38258, 47.15542, 51.45169, 55.49019, 59.36302, 63.11929,
        66.78873, 70.39089, 73.93937, 77.44404, 80.91233, 84.34994, 87.76135,
        91.15018, 94.51935, 97.87127, 101.20797, 104.53112, 107.84218],
        [37.05373, 43.45899, 48.26430, 52.58637, 56.64660, 60.53832, 64.31132,
        67.99577, 71.61155, 75.17247, 78.68860, 82.16747, 85.61490, 89.03548,
        92.43288, 95.81009, 99.16958, 102.51340, 105.84330, 109.16074],
        [38.07977, 44.53404, 49.37132, 53.71883, 57.80054, 61.71093, 65.50048,
        69.19982, 72.82911, 76.40239, 79.92989, 83.41929, 86.87650, 90.30621,
        93.71215, 97.09738, 100.46442, 103.81537, 107.15201, 110.47583],
        [39.10532, 45.60778, 50.47657, 54.84918, 58.95209, 62.88095, 66.68690,
        70.40099, 74.04367, 77.62922, 81.16803, 84.66789, 88.13485, 91.57364,
        94.98810, 98.38134, 101.75591, 105.11398, 108.45734, 111.78755],
        [40.13038, 46.68027, 51.58011, 55.97748, 60.10135, 64.04848, 67.87066,
        71.59937, 75.25535, 78.85308, 82.40312, 85.91339, 89.39004, 92.83789,
        96.26083, 99.66204, 103.04414, 106.40931, 109.75940, 113.09598],
        [41.15499, 47.75157, 52.68201, 57.10384, 61.24841, 65.21361, 69.05187,
        72.79506, 76.46423, 80.07406, 83.63525, 87.15588, 90.64217, 94.09904,
        97.53044, 100.93961, 104.32920, 107.70146, 111.05827, 114.40122],
        [42.17916, 48.82172, 53.78235, 58.22831, 62.39334, 66.37643, 70.23061,
        73.98816, 77.67041, 81.29225, 84.86454, 88.39546, 91.89134, 95.35719,
        98.79701, 102.21411, 105.61119, 108.99052, 112.35403, 115.70336],
        [43.20291, 49.89078, 54.88117, 59.35097, 63.53623, 67.53701, 71.40696,
        75.17876, 78.87399, 82.50775, 86.09106, 89.63221, 93.13764, 96.61242,
        100.06064, 103.48564, 106.89018, 110.27658, 113.64678, 117.00246],
        [44.22626, 50.95877, 55.97853, 60.47188, 64.67715, 68.69545, 72.58102,
        76.36693, 80.07504, 83.72064, 87.31489, 90.86622, 94.38115, 97.86483,
        101.32141, 104.75428, 108.16626, 111.55970, 114.93659, 118.29862],
        [45.24922, 52.02575, 57.07450, 61.59111, 65.81617, 69.85180, 73.75284,
        77.55275, 81.27364, 84.93100, 88.53613, 92.09758, 95.62195, 99.11448,
        102.57939, 106.02011, 109.43951, 112.83998, 116.22353, 119.59190],
        [46.27182, 53.09174, 58.16911, 62.70872, 66.95334, 71.00613, 74.92252,
        78.73630, 82.46987, 86.13890, 89.75484, 93.32635, 96.86012, 100.36147,
        103.83467, 107.28321, 110.71001, 114.11748, 117.50769, 120.88239],
        [47.29406, 54.15679, 59.26242, 63.82475, 68.08874, 72.15852, 76.09010,
        79.91765, 83.66380, 87.34442, 90.97110, 94.55261, 98.09573, 101.60585,
        105.08731, 108.54364, 111.97781, 115.39227, 118.78912, 122.17014],
        [48.31596, 55.22093, 60.35446, 64.93927, 69.22241, 73.30903, 77.25566,
        81.09686, 84.85550, 88.54763, 92.18498, 95.77643, 99.32885, 102.84771,
        106.33738, 109.80148, 113.24300, 116.66443, 120.06790, 123.45522],
        [49.33753, 56.28419, 61.44529, 66.05231, 70.35442, 74.45770, 78.41926,
        82.27400, 86.04503, 89.74859, 93.39655, 96.99788, 100.55956, 104.08710,
        107.58496, 111.05679, 114.50563, 117.93401, 121.34408, 124.73770],
        [50.35878, 57.34660, 62.53495, 67.16394, 71.48481, 75.60461, 79.58096,
        83.44912, 87.23246, 90.94737, 94.60586, 98.21702, 101.78790, 105.32409,
        108.83010, 112.30964, 115.76577, 119.20108, 122.61774, 126.01763],
        [51.37973, 58.40818, 63.62346, 68.27419, 72.61364, 76.74979, 80.74082,
        84.62230, 88.41784, 92.14402, 95.81298, 99.43391, 103.01395, 106.55874,
        110.07287, 113.56008, 117.02349, 120.46570, 123.88893, 127.29509],
        [52.40039, 59.46897, 64.71087, 69.38310, 73.74095, 77.89331, 81.89888,
        85.79357, 89.60123, 93.33861, 97.01798, 100.64862, 104.23776, 107.79112,
        111.31333, 114.80818, 118.27883, 121.72792, 125.15771, 128.57011],
        [53.42076, 60.52900, 65.79722, 70.49073, 74.86679, 79.03521, 83.05519,
        86.96299, 90.78269, 94.53119, 98.22089, 101.86119, 105.45939, 109.02127,
        112.55152, 116.05399, 119.53186, 122.98781, 126.42413, 129.84277]]

    def __init__(
        self,
        diameter=None,
        diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
        number_of_roots=20,
        number_of_functions=50,
    ):

        self.diameter = diameter
        self.Dintra = diffusion_constant
        self.alpha = self.SPHERICAL_BESSEL_ROOTS[:number_of_roots,:number_of_functions]

    def sphere_attenuation(self, q, tau, diameter):
        """Implements the finite time Callaghan model for spheres."""
        radius = diameter / 2.0
        q_argument = 2 * np.pi * q * radius
        q_argument_2 = q_argument ** 2
        res = np.zeros_like(q)
        beta = self.Dintra * tau / radius ** 2

        for n in range(0, self.alpha.shape[1]):
            Jder = special.spherical_jn(n, q_argument, derivative=True)
            for k in range(0, self.alpha.shape[0]):
                a_nk2 = self.alpha[k, n] ** 2
                update = 6 * np.exp(-a_nk2 * beta)
                if k!=0 or n !=0:
                    update *= (
                        ((2 * n + 1) * a_nk2) /
                        (a_nk2 - (n + 0.5) ** 2 + 0.25)
                    )
                update *= (q_argument * Jder) ** 2
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

        E0 = self.sphere_attenuation(
            1e-50, tau[0], diameter
        )
        E_sphere = np.ones_like(q)
        q_nonzero = q > 0
        E_sphere[q_nonzero] = self.sphere_attenuation(
            q[q_nonzero], tau[q_nonzero], diameter
        ) / E0
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
    _required_acquisition_parameters = ['gradient_strengths', 'delta', 'Delta']

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
