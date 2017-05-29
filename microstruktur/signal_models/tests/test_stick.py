from numpy.testing import assert_almost_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models, utils
from microstruktur.signal_models.utils import perpendicular_vector
DIFFUSIVITY_SCALING = 1e-9


def test_orienting_stick():
    # test for orienting the axis of the Stick along mu
    # first test to see if Estick equals Gaussian with lambda_par along mu
    random_n_mu_vector = np.random.rand(2) * np.pi
    n = utils.sphere2cart(np.r_[1, random_n_mu_vector])
    random_bval = np.r_[np.random.rand() * 1e9]
    random_lambda_par = np.random.rand() * 3

    # initialize model
    stick = three_dimensional_models.I1Stick(mu=random_n_mu_vector,
                                             lambda_par=random_lambda_par)

    # test if parallel direction attenuation as a Gaussian
    E_stick = stick(bvals=random_bval, n=n)
    E_check = np.exp(-random_bval * (random_lambda_par * DIFFUSIVITY_SCALING))
    assert_almost_equal(E_stick, E_check)

    # test if perpendicular direction does not attenuate
    n_perp = perpendicular_vector(n)
    E_stick_perp = stick(bvals=random_bval, n=n_perp)
    assert_almost_equal(E_stick_perp, 1.)
