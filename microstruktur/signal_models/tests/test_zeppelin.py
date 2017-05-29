from numpy.testing import assert_almost_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models, utils
from microstruktur.signal_models.utils import perpendicular_vector
DIFFUSIVITY_SCALING = 1e-9


def test_orienting_zeppelin():
    # test for orienting the axis of the Zeppelin along mu
    # first test to see if Ezeppelin equals Gaussian with lambda_par along mu
    random_mu = np.random.rand(2) * np.pi
    n = np.array([utils.sphere2cart(np.r_[1, random_mu])])
    random_bval = np.r_[np.random.rand() * 1e9]
    random_lambda_par = np.random.rand() * 3
    random_lambda_perp = random_lambda_par / 2.
    shell_index = np.r_[1]

    zeppelin = three_dimensional_models.E4Zeppelin(
        mu=random_mu, lambda_par=random_lambda_par,
        lambda_perp=random_lambda_perp)
    E_zep_par = zeppelin(bvals=random_bval, n=n, shell_indices=shell_index)
    E_check_par = np.exp(-random_bval * random_lambda_par *
                         DIFFUSIVITY_SCALING)
    assert_almost_equal(E_zep_par, E_check_par)

    # second test to see if Ezeppelin equals Gaussian with lambda_perp
    # perpendicular to mu
    n_perp = np.array([perpendicular_vector(n[0])])
    E_zep_perp = zeppelin(bvals=random_bval, n=n_perp,
                          shell_indices=shell_index)
    E_check_perp = np.exp(-random_bval * random_lambda_perp *
                          DIFFUSIVITY_SCALING)
    assert_almost_equal(E_zep_perp, E_check_perp)
