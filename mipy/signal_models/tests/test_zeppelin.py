from numpy.testing import assert_almost_equal
import numpy as np
from mipy.utils import utils
from mipy.signal_models import gaussian_models
from mipy.utils.utils import perpendicular_vector
from dipy.data import get_sphere
from mipy.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)
sphere = get_sphere().subdivide()

delta = 0.01
Delta = 0.03


def test_orienting_zeppelin():
    # test for orienting the axis of the Zeppelin along mu
    # first test to see if Ezeppelin equals Gaussian with lambda_par along mu
    random_mu = np.random.rand(2) * np.pi
    n = np.array([utils.sphere2cart(np.r_[1, random_mu])])
    random_bval = np.r_[np.random.rand() * 1e9]
    scheme = acquisition_scheme_from_bvalues(random_bval, n, delta, Delta)
    random_lambda_par = np.random.rand() * 3 * 1e-9
    random_lambda_perp = random_lambda_par / 2.

    zeppelin = gaussian_models.G2Zeppelin(
        mu=random_mu, lambda_par=random_lambda_par,
        lambda_perp=random_lambda_perp)
    E_zep_par = zeppelin(scheme)
    E_check_par = np.exp(-random_bval * random_lambda_par)
    assert_almost_equal(E_zep_par, E_check_par)

    # second test to see if Ezeppelin equals Gaussian with lambda_perp
    # perpendicular to mu
    n_perp = np.array([perpendicular_vector(n[0])])
    scheme = acquisition_scheme_from_bvalues(random_bval, n_perp, delta, Delta)
    E_zep_perp = zeppelin(scheme)
    E_check_perp = np.exp(-random_bval * random_lambda_perp)
    assert_almost_equal(E_zep_perp, E_check_perp)
