from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from microstruktur.utils import utils
from microstruktur.signal_models import cylinder_models, spherical_mean_models
from microstruktur.utils.utils import perpendicular_vector
from microstruktur.signal_models.dispersed_models import (
    SD2C1BinghamDispersedStick,
    SD1C1WatsonDispersedStick
)
from dipy.data import get_sphere
from microstruktur.signal_models.spherical_mean import (
    estimate_spherical_mean_shell
)
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


def test_watson_dispersed_stick_kappa0(
    lambda_par=1.7e-9, bvalue=1e9, mu=[0, 0], kappa=0
):
    # testing uniformly dispersed bingham stick.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of stick.
    sm_stick = spherical_mean_models.C1StickSphericalMean(
        lambda_par=lambda_par
    )
    E_sm_stick = sm_stick(scheme)

    watson_stick = SD1C1WatsonDispersedStick(mu=mu, kappa=kappa,
                                             lambda_par=lambda_par)
    E_watson_stick = watson_stick(scheme)
    E_unique_watson_stick = np.unique(E_watson_stick)
    # All values are the same:
    assert_equal(len(E_unique_watson_stick), 1)
    # and are equal to the spherical mean:
    assert_almost_equal(E_unique_watson_stick, E_sm_stick)


def test_watson_dispersed_stick_kappa_positive(
    lambda_par=1.7e-9, bvalue=1e9, mu=[0, 0], kappa=10
):
    # testing uniformly dispersed bingham stick.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of stick.
    sm_stick = spherical_mean_models.C1StickSphericalMean(
        lambda_par=lambda_par
    )
    E_sm_stick = sm_stick(scheme)

    watson_stick = SD1C1WatsonDispersedStick(mu=mu, kappa=kappa,
                                             lambda_par=lambda_par)
    E_watson_stick = watson_stick(scheme)
    E_unique_watson_stick = np.unique(E_watson_stick)
    E_sm_watson_stick = estimate_spherical_mean_shell(E_watson_stick, n)
    # Different values for different orientations:
    assert_equal(len(E_unique_watson_stick) > 1, True)
    # but the spherical mean does not change with dispersion:
    assert_almost_equal(E_sm_watson_stick, E_sm_stick, 4)


def test_bingham_dispersed_stick_kappa0(
    lambda_par=1.7e-9, bvalue=1e9, mu=[0, 0], kappa=0, beta=0, psi=0
):
    # testing uniformly dispersed bingham stick.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of stick.
    sm_stick = spherical_mean_models.C1StickSphericalMean(
        lambda_par=lambda_par
    )
    E_sm_stick = sm_stick(scheme)

    bingham_stick = SD2C1BinghamDispersedStick(
        mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par
    )
    E_bingham_stick = bingham_stick(scheme)
    E_unique_bingham_stick = np.unique(E_bingham_stick)
    # All values are the same:
    assert_equal(len(E_unique_bingham_stick), 1)
    # and are equal to the spherical mean:
    assert_almost_equal(E_unique_bingham_stick, E_sm_stick)


def test_bingham_dispersed_stick_kappa_positive(
        lambda_par=1.7e-9, bvalue=1e9, mu=[0, 0], kappa=10, beta=0, psi=0
):
    # testing uniformly dispersed bingham stick.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of stick.
    sm_stick = spherical_mean_models.C1StickSphericalMean(
        lambda_par=lambda_par
    )
    E_sm_stick = sm_stick(scheme)

    bingham_stick = SD2C1BinghamDispersedStick(
        mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par
    )
    E_bingham_stick = bingham_stick(scheme)
    E_sm_bingham_stick = estimate_spherical_mean_shell(E_bingham_stick, n)
    E_unique_bingham_stick = np.unique(E_bingham_stick)
    # Different values for different orientations:
    assert_equal(len(E_unique_bingham_stick) > 1, True)
    # but the spherical mean does not change with dispersion:
    assert_almost_equal(E_sm_bingham_stick, E_sm_stick, 3)
