from microstruktur.signal_models import (
    three_dimensional_models)
from numpy.testing import assert_array_equal, assert_equal
import numpy as np
DIFFUSIVITY_SCALING = 1e-9


def test_dot():
    bvals = np.random.rand(10)
    dot = three_dimensional_models.E2Dot()
    E_dot = dot(bvals)
    assert_equal(np.all(E_dot == 1.), True)


def test_ball(lambda_iso=1.7):
    bvals = np.linspace(0, 1e9, 10)
    ball = three_dimensional_models.E3Ball(lambda_iso=lambda_iso)
    E_ball = ball(bvals=bvals)
    E = np.exp(-bvals * lambda_iso * DIFFUSIVITY_SCALING)
    assert_array_equal(E, E_ball)
