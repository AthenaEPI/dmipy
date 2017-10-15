from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models, utils
from microstruktur.signal_models.utils import perpendicular_vector
from microstruktur.signal_models.dispersed_models import (
    SD2I1BinghamDispersedStick,
    SD3I1WatsonDispersedStick
)
from dipy.data import get_sphere
from microstruktur.signal_models.spherical_mean import (
    estimate_spherical_mean_shell
)
DIFFUSIVITY_SCALING = 1e-9
sphere = get_sphere().subdivide()


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


def test_watson_dispersed_stick_kappa0(lambda_par=1.7, bvalue=1e9, mu=[0, 0],
                                       kappa=0):
    # for comparison we do spherical mean of stick.
    sm_stick = three_dimensional_models.I1StickSphericalMean(
        lambda_par=lambda_par
    )
    E_sm_stick = sm_stick(np.r_[bvalue])

    # testing uniformly dispersed watson stick.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))

    watson_stick = SD3I1WatsonDispersedStick(mu=mu, kappa=kappa,
                                             lambda_par=lambda_par)
    E_watson_stick = watson_stick(bvals=bvals, n=n,
                                  shell_indices=shell_indices)
    E_unique_watson_stick = np.unique(E_watson_stick)
    # All values are the same:
    assert_equal(len(E_unique_watson_stick), 1)
    # and are equal to the spherical mean:
    assert_almost_equal(E_unique_watson_stick, E_sm_stick)


def test_watson_dispersed_stick_kappa_positive(lambda_par=1.7, bvalue=1e9,
                                               mu=[0, 0], kappa=10):
    # for comparison we do spherical mean of stick.
    sm_stick = three_dimensional_models.I1StickSphericalMean(
        lambda_par=lambda_par
    )
    E_sm_stick = sm_stick(np.r_[bvalue])

    # now testing concentrated watson stick with positive kappa.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))

    watson_stick = SD3I1WatsonDispersedStick(mu=mu, kappa=kappa,
                                             lambda_par=lambda_par)
    E_watson_stick = watson_stick(bvals=bvals, n=n,
                                  shell_indices=shell_indices)
    E_unique_watson_stick = np.unique(E_watson_stick)
    E_sm_watson_stick = estimate_spherical_mean_shell(E_watson_stick, n)
    # Different values for different orientations:
    assert_equal(len(E_unique_watson_stick) > 1, True)
    # but the spherical mean does not change with dispersion:
    assert_almost_equal(E_sm_watson_stick, E_sm_stick, 4)
