from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
import numpy as np
from microstruktur.signal_models.three_dimensional_models import I1_stick


def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            v_perp = np.cross(v, [0, 1, 0])
            v_perp /= np.linalg.norm(v_perp)
            return v_perp
    v_perp = np.cross(v, [1, 0, 0])
    v_perp /= np.linalg.norm(v_perp)
    return v_perp


def test_orienting_stick():
    # test for orienting the axis of the Stick along mu
    # first test to see if Estick equals Gaussian with lambda_par along mu
    random_n_mu_vector = np.random.rand(3)
    random_n_mu_vector /= np.linalg.norm(random_n_mu_vector)
    random_bval = np.random.rand() * 1000.
    random_lambda_par = np.random.rand() * 3e-3
    E_stick = I1_stick(random_bval, random_n_mu_vector,
                       random_n_mu_vector, random_lambda_par)
    E_check = np.exp(-random_bval * random_lambda_par)
    assert_almost_equal(E_stick, E_check)

    # second test to see if perpendicular vector is unity
    random_n_mu_vector_perp = perpendicular_vector(random_n_mu_vector)
    E_stick_perp = I1_stick(random_bval, random_n_mu_vector_perp,
                            random_n_mu_vector, random_lambda_par)
    assert_equal(E_stick_perp, 1.)


def test_array_stick(n_length=10):
    # test to see if n can also be given as an array.
    random_mu_vector = np.random.rand(3)
    random_mu_vector /= np.linalg.norm(random_mu_vector)
    random_n_vector = np.random.rand(n_length, 3)
    random_n_vector /= np.linalg.norm(random_n_vector, axis=0)
    random_bval = np.random.rand() * 1000.
    random_lambda_par = np.random.rand() * 3e-3
    E_stick = I1_stick(random_bval, random_n_vector,
                       random_mu_vector, random_lambda_par)
    assert_equal(E_stick.shape[0], n_length)
