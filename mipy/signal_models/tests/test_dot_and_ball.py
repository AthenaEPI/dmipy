from mipy.signal_models import gaussian_models, sphere_models
from numpy.testing import assert_array_equal, assert_equal
import numpy as np
from mipy.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)

bvals = np.random.rand(10) * 1e9
bvecs = np.random.rand(10, 3)
bvecs /= np.linalg.norm(bvecs, axis=1)[:, None]
delta = 0.01
Delta = 0.03
scheme = acquisition_scheme_from_bvalues(bvals, bvecs, delta, Delta)


def test_dot():
    dot = sphere_models.S1Dot()
    E_dot = dot(scheme)
    assert_equal(np.all(E_dot == 1.), True)


def test_ball(lambda_iso=1.7e-9):
    ball = gaussian_models.G1Ball(lambda_iso=lambda_iso)
    E_ball = ball(scheme)
    E = np.exp(-bvals * lambda_iso)
    assert_array_equal(E, E_ball)
