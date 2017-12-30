from numpy.testing import assert_almost_equal
import numpy as np
from microstruktur.utils import utils
from microstruktur.signal_models import cylinder_models
from microstruktur.utils.utils import perpendicular_vector
from dipy.data import get_sphere

from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)
sphere = get_sphere().subdivide()

delta = 0.01
Delta = 0.03


def test_orienting_stick():
    # test for orienting the axis of the Stick along mu
    # first test to see if Estick equals Gaussian with lambda_par along mu
    random_n_mu_vector = np.random.rand(2) * np.pi
    n = utils.sphere2cart(np.r_[1, random_n_mu_vector])
    random_bval = np.r_[np.random.rand() * 1e9]
    random_lambda_par = np.random.rand() * 3e-9

    scheme = acquisition_scheme_from_bvalues(
        random_bval, np.atleast_2d(n), delta, Delta)
    # initialize model
    stick = cylinder_models.C1Stick(mu=random_n_mu_vector,
                                    lambda_par=random_lambda_par)

    # test if parallel direction attenuation as a Gaussian
    E_stick = stick(scheme)
    E_check = np.exp(-random_bval * (random_lambda_par))
    assert_almost_equal(E_stick, E_check)

    # test if perpendicular direction does not attenuate
    n_perp = perpendicular_vector(n)
    scheme = acquisition_scheme_from_bvalues(
        random_bval, np.atleast_2d(n_perp), delta, Delta)
    E_stick_perp = stick(scheme)
    assert_almost_equal(E_stick_perp, 1.)
