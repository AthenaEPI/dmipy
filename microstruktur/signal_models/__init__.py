"""Several tools for Diffusion MRI sequence simulation."""
from __future__ import division
from .constants import CONSTANTS
from . import multiple_correlation_function
from . import callaghan
from .free_diffusion import free_diffusion_attenuation
from . import gaussian_phase
from . import narrow_pulse
from .convert_units import (
    unit_conversion_factor,
    unit_conversion_factor_from_SI, unit_conversion_factor_to_SI
)
from .gradient_conversions import *
#from . import three_dimensional_models


def max_diffusion_time(
    diameter=None,
    diffusion_constant=CONSTANTS['water_in_axons_diffusion_constant'],
):
    '''
    Diffusion time to reach container walls
    '''
    radius = diameter / 2
    return radius ** 2 / diffusion_constant
