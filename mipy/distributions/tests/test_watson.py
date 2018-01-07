from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from mipy.distributions import distributions
from mipy.utils import utils
from dipy.data import get_sphere


def test_watson_integral_unity():
    # test for integral to unity for isotropic concentration (k=0)
    odi = 1.  # isotropic distribution
    # first test for one orientation n
    random_n = np.random.rand(3)
    random_n /= np.linalg.norm(random_n)
    random_mu = np.random.rand(2)
    watson = distributions.SD1Watson(mu=random_mu, odi=odi)
    Wn = watson(n=random_n)  # in random direction
    spherical_integral = Wn * 4 * np.pi  # for isotropic distribution
    assert_equal(spherical_integral, 1.)

    # third test for unity when k>0
    sphere = get_sphere('repulsion724')
    n_sphere = sphere.vertices
    odi = np.random.rand()
    Wn_sphere = watson(n=n_sphere, mu=random_mu, odi=odi)
    spherical_integral = sum(Wn_sphere) / n_sphere.shape[0] * 4 * np.pi
    assert_almost_equal(spherical_integral, 1., 4)


def test_watson_orienting():
    # test for orienting the axis of the Watson distribution mu for k>0
    # first test to see if Wn is highest along mu
    sphere = get_sphere('repulsion724')
    n = sphere.vertices
    indices = np.array(range(n.shape[0]))
    np.random.shuffle(indices)
    mu_index = indices[0]
    mu_cart = n[mu_index]
    mu_sphere = utils.cart2sphere(mu_cart)[1:]
    odi = np.random.rand()
    watson = distributions.SD1Watson(mu=mu_sphere, odi=odi)
    Wn_vector = watson(n=n)
    assert_almost_equal(Wn_vector[mu_index], max(Wn_vector))

    # second test to see if Wn is lowest prependicular to mu
    mu_perp = utils.perpendicular_vector(mu_cart)
    Wn_perp = watson(n=mu_perp)
    assert_equal(np.all(Wn_perp < Wn_vector), True)


def test_watson_kappa():
    # test for Wn(k2) > Wn(k1) when k2>k1 along mu
    random_mu = np.random.rand(2)
    random_mu_cart = utils.sphere2cart(np.r_[1., random_mu])
    odi1 = .8
    odi2 = .6
    watson = distributions.SD1Watson(mu=random_mu)
    Wn1 = watson(n=random_mu_cart, odi=odi1)
    Wn2 = watson(n=random_mu_cart, odi=odi2)
    assert_equal(Wn2 > Wn1, True)
