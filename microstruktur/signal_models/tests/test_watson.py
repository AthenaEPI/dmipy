from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           run_module_suite)
import numpy as np
from microstruktur.signal_models.three_dimensional_models import SD3_watson
from dipy.data import get_sphere

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

def test_watson_integral_unity():
    # test for integral to unity for isotropic concentration (k=0)
    kappa = 0 # isotropic distribution
    # first test for one orientation n
    random_n = np.random.rand(3)
    random_n /= np.linalg.norm(random_n)
    random_mu = np.random.rand(3)
    random_mu /= np.linalg.norm(random_mu)
    Wn = SD3_watson(random_n, random_mu, kappa) # in random direction
    spherical_integral = Wn * 4 * np.pi # for isotropic distribution
    assert_equal(spherical_integral, 1.)
    
    # second test for unity for array of orientations n
    random_n_vector = np.random.rand(10,3)
    random_n_vector /= np.linalg.norm(random_n_vector, axis=0)
    Wn_vector = SD3_watson(random_n_vector, random_mu, kappa) # in random directions
    spherical_integrals = Wn_vector * 4 * np.pi
    assert_equal(np.all(spherical_integrals == 1.), True)
    
    # third test for unity when k>0
    sphere = get_sphere('repulsion724')
    n_sphere = sphere.vertices
    kappa = np.random.rand() + 0.1 # just to be sure kappa>0
    Wn_sphere = SD3_watson(n_sphere, random_mu, kappa)
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
    random_mu = n[mu_index]
    kappa = np.random.rand() + 0.1 # just to be sure kappa>0
    Wn_vector = SD3_watson(n, random_mu, kappa)
    assert_equal(Wn_vector[mu_index] == max(Wn_vector), True)
    
    # second test to see if Wn is lowest prependicular to mu
    random_mu_perp = perpendicular_vector(random_mu)
    Wn_perp = SD3_watson(random_mu_perp, random_mu, kappa)
    assert_equal(np.all(Wn_perp < Wn_vector), True)

def test_watson_kapa():
    # test for Wn(k1) > Wn(k2) when k1>k2 along mu
    random_n_mu_vector = np.random.rand(3)
    random_n_mu_vector /= np.linalg.norm(random_n_mu_vector)
    kappa2 = np.random.rand()
    kappa1 = kappa2 + np.random.rand() + 0.1
    Wn1 = SD3_watson(random_n_mu_vector, random_n_mu_vector, kappa1)
    Wn2 = SD3_watson(random_n_mu_vector, random_n_mu_vector, kappa2)
    assert_equal(Wn1 > Wn2, True)