from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
import numpy as np
from microstruktur.signal_models.three_dimensional_models import E4_zeppelin
from microstruktur.signal_models.utils import perpendicular_vector


def test_orienting_zeppelin():
    # test for orienting the axis of the Zeppelin along mu
    # first test to see if Ezeppelin equals Gaussian with lambda_par along mu
    random_n_mu_vector = np.random.rand(3)
    random_n_mu_vector /= np.linalg.norm(random_n_mu_vector)
    random_bval = np.random.rand() * 1000.
    random_lambda_par = np.random.rand() * 3e-3
    random_lambda_perp = random_lambda_par / 2.
    E_zep_par = E4_zeppelin(random_bval, random_n_mu_vector,
                            random_n_mu_vector,
                            random_lambda_par, random_lambda_perp)
    E_check_par = np.exp(-random_bval * random_lambda_par)
    assert_almost_equal(E_zep_par, E_check_par)

    # second test to see if Ezeppelin equals Gaussian with lambda_perp
    # perpendicular to mu
    random_n_mu_vector_perp = perpendicular_vector(random_n_mu_vector)
    E_zep_perp = E4_zeppelin(random_bval, random_n_mu_vector_perp,
                             random_n_mu_vector,
                             random_lambda_par, random_lambda_perp)
    E_check_perp = np.exp(-random_bval * random_lambda_perp)
    assert_almost_equal(E_zep_perp, E_check_perp)


def test_array_zeppelin():
    # test to see if n can also be given as an array.
    random_mu_vector = np.random.rand(3)
    random_mu_vector /= np.linalg.norm(random_mu_vector)
    n_length = 10
    random_n_vector = np.random.rand(n_length, 3)
    random_n_vector /= np.linalg.norm(random_n_vector, axis=0)
    random_bval = np.random.rand(n_length) * 1000.
    random_lambda_par = np.random.rand() * 3e-3
    random_lambda_perp = random_lambda_par / 2.
    E_zeppelin = E4_zeppelin(random_bval, random_n_vector, random_mu_vector,
                             random_lambda_par, random_lambda_perp)
    assert_equal(E_zeppelin.shape[0], n_length)
