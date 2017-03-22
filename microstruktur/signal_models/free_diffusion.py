from .constants import CONSTANTS

import numpy as np


def free_diffusion_attenuation(
    gradient_strength=None, delta=None, Delta=None,
    diffusion_constant=CONSTANTS['water_diffusion_constant'],
    gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    **kwargs
):
    """Taken from the AxCaliber paper eq [2]."""
    tau = Delta - delta / 3
    D = diffusion_constant
    E = np.exp(
        -(gyromagnetic_ratio * delta * gradient_strength) ** 2 *
        D * tau
    )
    return E
