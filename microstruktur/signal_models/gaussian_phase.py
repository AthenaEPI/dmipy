# -*- coding: utf-8 -*-
'''
Approximations from
Balinov B, Jonsson B, Linse P, Soderman O (1993)
The NMR Self-Diffusion Method Applied to Restricted Diffusion.
Simulation of Echo Attenuation from Molecules in Spheres and between Planes.
JMR 104:17–25. doi: 10.1006/jmra.1993.1184

which are based on
Murday JS, Cotts RM (1968) Self-Diffusion Coefficient of Liquid Lithium.
The Journal of Chemical Physics 48:4938. doi: 10.1063/1.1668160
'''
from __future__ import division

import numpy as np
import scipy.special

from .constants import CONSTANTS


CYLINDER_TRASCENDENTAL_ROOTS = np.sort(scipy.special.jnp_zeros(1, 1000))


def cylinder_attenuation(
    gradient_strength=None, Delta=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    N=10,
    **kwargs
):
    '''
    Delta is the pulse separation
    '''
    D = diffusion_constant
    gamma = gyromagnetic_ratio
    radius = diameter / 2

    first_factor = -2 * (gradient_strength * gamma) ** 2
    alpha = CYLINDER_TRASCENDENTAL_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D
    Delta = delta + Delta

    summands = (
        2 * alpha2D * delta - 2 +
        2 * np.exp(-alpha2D * delta) +
        2 * np.exp(-alpha2D * Delta) -
        np.exp(-alpha2D * (Delta - delta)) -
        np.exp(-alpha2D * (Delta + delta))
    ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))

    E = np.exp(
        first_factor *
        summands.sum()
    )

    return E


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


def sphere_attenuation(
    gradient_strength=None, Delta=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    N=10,
    **kwargs
):
    '''
    Delta is the pulse separation
    '''

    D = diffusion_constant
    gamma = gyromagnetic_ratio
    radius = diameter / 2
    Delta = delta + Delta

    alpha = SPHERE_TRASCENDENTAL_ROOTS[:N] / radius
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


def apparent_mean_squared_displacement(
    td=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    geometry='disk', N=10,
    **kwargs
):
    '''
    From Aslund et al 2009
    '''
    geometry = geometry.lower()
    if geometry == 'disk':
        alphas = CYLINDER_TRASCENDENTAL_ROOTS[:N]
        n_d = 2
    elif geometry == 'sphere':
        alphas = SPHERE_TRASCENDENTAL_ROOTS[:N]
        n_d = 3
    else:
        raise ValueError('geometry must be "disk" or "sphere"')

    if (td - 2 * delta / 3) < 0:
        raise ValueError("td - 2 delta / 3 must be positive")

    D = diffusion_constant
    radius = diameter / 2
    alphas2 = alphas ** 2
    L = lambda t: np.exp(-alphas2 * D * t)
    first_factor = (1 / (alphas2 * (alphas2 * radius ** 2 + 1 - n_d)))
    second_factor = (
        2 * alphas2 * D * delta - 2 +
        2 * L(delta) + 2 * L(td + delta / 3) -
        L(td - 2 * delta / 3) - L(td + 4 * delta / 3)
    ) / ((alphas2 * D * delta) ** 2)

    terms = first_factor * second_factor
    print(terms.sum())
    return 4 * terms.sum()


def apparent_mean_squared_displacement_short_delta(
    td=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    geometry='disk', N=10,
    **kwargs
):
    '''
    From Aslund et al 2009
    '''
    geometry = geometry.lower()
    if geometry == 'disk':
        alphas = CYLINDER_TRASCENDENTAL_ROOTS[:N]
        n_d = 2
    elif geometry == 'sphere':
        alphas = SPHERE_TRASCENDENTAL_ROOTS[:N]
        n_d = 3
    else:
        raise ValueError('geometry must be "disk" or "sphere"')
    D = diffusion_constant
    radius = diameter / 2
    alphas2 = alphas ** 2
    L = lambda t: np.exp(-alphas2 * D * t)

    terms = (
        (1 - L(td)) /
        (alphas2 * (alphas2 * radius ** 2 + 1 - n_d))
    )

    return 4 * terms.sum()


def apparent_mean_squared_displacement_short_td(
    td=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    **kwargs
):
    '''
    From Aslund et al 2009
    '''
    D = diffusion_constant

    return 2 * D * td


def apparent_mean_squared_displacement_long_td(
    td=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    geometry='disk', N=10,
    **kwargs
):
    '''
    From Aslund et al 2009
    '''
    geometry = geometry.lower()
    if geometry == 'disk':
        alphas = CYLINDER_TRASCENDENTAL_ROOTS[:N]
        n_d = 2
    elif geometry == 'sphere':
        alphas = SPHERE_TRASCENDENTAL_ROOTS[:N]
        n_d = 3
    else:
        raise ValueError('geometry must be "disk" or "sphere"')

    D = diffusion_constant
    radius = diameter / 2
    alphas2 = alphas ** 2
    L = lambda t: np.exp(-alphas2 * D * t)

    terms = (
        (1 - L(td)) /
        (alphas2 * (alphas2 * radius ** 2 + 1 - n_d))
    ) * (
        (alphas2 * D * delta - 1 + L(delta)) /
        (alphas2 * D * delta) ** 2
    )

    return 8 * terms.sum()


def apparent_mean_squared_displacement_long_delta_long_td(
    td=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    geometry='disk', N=10,
    **kwargs
):
    '''
    From Aslund et al 2009
    '''
    geometry = geometry.lower()
    if geometry == 'disk':
        C = 7. / 24
    elif geometry == 'sphere':
        C = 32. / 175
    else:
        raise ValueError('geometry must be "disk" or "sphere"')

    D = diffusion_constant
    radius = diameter / 2

    return C * radius ** 4 / (D * delta)


def apparent_mean_squared_displacement_short_delta_long_td(
    td=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    geometry='disk', N=10,
    **kwargs
):
    '''
    From Aslund et al 2009
    '''
    geometry = geometry.lower()
    if geometry == 'disk':
        n_d = 2
    elif geometry == 'sphere':
        n_d = 3
    else:
        raise ValueError('geometry must be "disk" or "sphere"')

    radius = diameter / 2

    return 2 / (2 + n_d) * radius ** 2


def attenuation(
    gradient_strength=None, Delta=None, delta=None, diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    N=10, geometry='disk', regime=apparent_mean_squared_displacement,
    **kwargs
):
    first_factor = -(gyromagnetic_ratio * gradient_strength * delta) ** 2 / 2
    Delta = delta + Delta
    td = Delta - delta / 3
    msd = regime(
        gradient_strength=gradient_strength,
        td=td, delta=delta, diameter=diameter,
        diffusion_constant=diffusion_constant,
        gyromagnetic_ratio=gyromagnetic_ratio,
        N=N, geometry=geometry,
        **kwargs
    )

    return np.exp(first_factor * msd)
