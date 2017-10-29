from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models, utils
from microstruktur.signal_models.utils import perpendicular_vector
from microstruktur.signal_models.dispersed_models import (
    SD2E4BinghamDispersedZeppelin,
    SD3E4WatsonDispersedZeppelin
)
from microstruktur.signal_models.spherical_mean import (
    estimate_spherical_mean_shell
)
from dipy.data import get_sphere
from microstruktur.acquisition_scheme.acquisition_scheme import (
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

    zeppelin = three_dimensional_models.E4Zeppelin(
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


def test_watson_dispersed_zeppelin_kappa0(
    lambda_par=1.7e-9, lambda_perp=1e-9, bvalue=1e9, mu=[0, 0], kappa=0
):
    # testing uniformly dispersed watson zeppelin.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of zeppelin.
    sm_zeppelin = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    E_sm_zeppelin = sm_zeppelin(scheme)

    watson_zeppelin = SD3E4WatsonDispersedZeppelin(
        mu=mu, kappa=kappa, lambda_par=lambda_par, lambda_perp=lambda_perp)
    E_watson_zeppelin = watson_zeppelin(scheme)
    E_unique_watson_zeppelin = np.unique(E_watson_zeppelin)
    # All values are the same:
    assert_equal(len(E_unique_watson_zeppelin), 1)
    # and are equal to the spherical mean:
    assert_almost_equal(E_unique_watson_zeppelin, E_sm_zeppelin)


def test_watson_dispersed_zeppelin_kappa_positive(
    lambda_par=1.7e-9, lambda_perp=1e-9, bvalue=1e9, mu=[0, 0], kappa=10
):
    # now testing concentrated watson zeppelin with positive kappa.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of zeppelin.
    sm_zeppelin = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    E_sm_zeppelin = sm_zeppelin(scheme)

    watson_zeppelin = SD3E4WatsonDispersedZeppelin(
        mu=mu, kappa=kappa, lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    E_watson_zeppelin = watson_zeppelin(scheme)
    E_unique_watson_zeppelin = np.unique(E_watson_zeppelin)
    E_sm_watson_zeppelin = estimate_spherical_mean_shell(E_watson_zeppelin, n)
    # Different values for different orientations:
    assert_equal(len(E_unique_watson_zeppelin) > 1, True)
    # but the spherical mean does not change with dispersion:
    assert_almost_equal(E_sm_watson_zeppelin, E_sm_zeppelin, 4)


def test_bingham_dispersed_zeppelin_kappa0(
    lambda_par=1.7e-9, lambda_perp=1e-9, bvalue=1e9, mu=[0, 0],
    kappa=0, beta=0, psi=0
):
    # testing uniformly dispersed bingham zeppelin.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)

    # for comparison we do spherical mean of zeppelin.
    sm_zeppelin = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    E_sm_zeppelin = sm_zeppelin(scheme)

    bingham_zeppelin = SD2E4BinghamDispersedZeppelin(
        mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par,
        lambda_perp=lambda_perp
    )
    E_bingham_zeppelin = bingham_zeppelin(scheme)
    E_unique_bingham_zeppelin = np.unique(E_bingham_zeppelin)
    # All values are the same:
    assert_equal(len(E_unique_bingham_zeppelin), 1)
    # and are equal to the spherical mean:
    assert_almost_equal(E_unique_bingham_zeppelin, E_sm_zeppelin)


def test_bingham_dispersed_zeppelin_kappa_positive(
    lambda_par=1.7e-9, lambda_perp=1e-9, bvalue=1e9, mu=[0, 0], kappa=10,
    beta=0, psi=0
):
    # for comparison we do spherical mean of zeppelin.
    sm_zeppelin = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    # testing uniformly dispersed bingham zeppelin.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    scheme = acquisition_scheme_from_bvalues(bvals, n, delta, Delta)
    E_sm_zeppelin = sm_zeppelin(scheme)

    bingham_zeppelin = SD2E4BinghamDispersedZeppelin(
        mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par,
        lambda_perp=lambda_perp
    )
    E_bingham_zeppelin = bingham_zeppelin(scheme)
    E_sm_bingham_zeppelin = estimate_spherical_mean_shell(
        E_bingham_zeppelin, n
    )
    E_unique_bingham_zeppelin = np.unique(E_bingham_zeppelin)
    # Different values for different orientations:
    assert_equal(len(E_unique_bingham_zeppelin) > 1, True)
    # but the spherical mean does not change with dispersion:
    assert_almost_equal(E_sm_bingham_zeppelin, E_sm_zeppelin, 4)
