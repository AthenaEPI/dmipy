from __future__ import division
import numpy as np
import scipy.special

from .constants import CONSTANTS
from .gradient_conversions import q_from_g


def cylinder_attenuation_long_time_limit(
    gradient_strength=None, delta=None, diameter=None,
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    **kwargs
):
    q = q_from_g(gradient_strength, delta, gyromagnetic_ratio)
    mask = q > 0
    v = np.pi * q[mask] * diameter / 2
    E = np.ones_like(q)
    E[mask] = (scipy.special.j1(2 * v) / v) ** 2
    return E


def sphere_attenuation_long_time_limit(
    gradient_strength=None, delta=None, diameter=None,
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    **kwargs
):
    q = q_from_g(gradient_strength, delta, gyromagnetic_ratio)
    mask = q > 0
    radius = diameter / 2
    factor = 2 * np.pi * q[mask] * radius
    E = np.ones_like(q)
    E[mask] = (
        3 / (factor ** 2) *
        (
            np.sin(factor) / factor -
            np.cos(factor)
        )
    ) ** 2

    return E
