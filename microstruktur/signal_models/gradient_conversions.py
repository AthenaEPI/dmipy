import numpy as np
from .constants import CONSTANTS


def q_from_g(
    g, delta,
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio']
):
    """Compute q-value from gradient strength. Units are standard units."""
    q = g * delta * gyromagnetic_ratio / (2 * np.pi)
    return q


def g_from_q(
    q, delta,
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio']
):
    """Compute gradient strength from q-value. Units are standard units."""
    return q * (2 * np.pi) / (delta * gyromagnetic_ratio)


def b_from_g(
    g, delta, Delta,
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio']
):
    """Compute b-value from gradient strength. Units are standard units."""
    tau = Delta - delta / 3
    return (g * gyromagnetic_ratio * delta) ** 2 * tau


def g_from_b(
    b, delta, Delta,
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio']
):
    """Compute gradient strength from b-value. Units are standard units."""
    tau = Delta - delta / 3
    return np.sqrt(b / tau) / (gyromagnetic_ratio * delta)
