from numpy.testing import assert_almost_equal, assert_equal
from mipy.utils import utils
import numpy as np
from mipy.utils.utils import (
    rotation_matrix_100_to_theta_phi, rotation_matrix_around_100,
    rotation_matrix_100_to_theta_phi_psi
)
from mipy.distributions import distributions


def test_rotation_100_to_theta_phi():
    # test 1: does R100_to_theta_phi rotate a vector theta_phi?
    theta_ = np.random.rand() * np.pi
    phi_ = (np.random.rand() - .5) * np.pi
    R100_to_theta_pi = rotation_matrix_100_to_theta_phi(theta_, phi_)
    xyz = np.dot(R100_to_theta_pi, np.r_[1, 0, 0])
    _, theta_rec, phi_rec = utils.cart2sphere(xyz)
    assert_almost_equal(theta_, theta_rec)
    assert_almost_equal(phi_, phi_rec)


def test_axis_rotation_does_not_affect_axis():
    # test 2: does R_around_100 not affect 100?
    psi_ = np.random.rand() * np.pi
    R_around_100 = rotation_matrix_around_100(psi_)
    v100 = np.r_[1, 0, 0]
    assert_equal(v100, np.dot(R_around_100, v100))


def test_psi_insensitivity_when_doing_psi_theta_phi_rotation():
    # test 3: does psi still have no influence on main eigenvector when doing
    # both rotations?
    theta_ = np.random.rand() * np.pi
    phi_ = (np.random.rand() - .5) * np.pi
    psi_ = np.random.rand() * np.pi
    R_ = rotation_matrix_100_to_theta_phi_psi(theta_, phi_, psi_)
    xyz = np.dot(R_, np.r_[1, 0, 0])
    _, theta_rec, phi_rec = utils.cart2sphere(xyz)
    assert_almost_equal(theta_, theta_rec)
    assert_almost_equal(phi_, phi_rec)


def test_rotation_around_axis():
    # test 4: does psi really rotate the second vector?
    psi_ = np.pi  # half circle
    R_around_100 = rotation_matrix_around_100(psi_)
    v2 = np.r_[0, 1, 0]
    v2_expected = np.r_[0, -1, 0]
    v2_rot = np.dot(R_around_100, v2)
    assert_equal(np.round(v2_rot), v2_expected)


def test_rotation_on_bingham_tensor():
    # test 5: does combined rotation rotate Bingham well?
    kappa_ = np.random.rand()
    beta_ = kappa_ / 2.  # beta<kappa
    Bdiag_ = np.diag(np.r_[kappa_, beta_, 0])

    theta_ = np.random.rand() * np.pi
    phi_ = (np.random.rand() - .5) * np.pi
    psi_ = np.random.rand() * np.pi * 0
    R_ = rotation_matrix_100_to_theta_phi_psi(theta_, phi_, psi_)

    B_ = R_.dot(Bdiag_).dot(R_.T)
    eigvals, eigvecs = np.linalg.eigh(B_)
    main_evec = eigvecs[:, np.argmax(eigvals)]
    _, theta_rec0, phi_rec0 = utils.cart2sphere(main_evec)

    # checking if the angles are antipodal to each other
    if abs(theta_ - theta_rec0) > 1e-5:
        theta_rec = np.pi - theta_rec0
        if phi_rec0 > 0:
            phi_rec = phi_rec0 - np.pi
        elif phi_rec0 < 0:
            phi_rec = phi_rec0 + np.pi
    else:
        theta_rec = theta_rec0
        phi_rec = phi_rec0
    assert_almost_equal(theta_, theta_rec)
    assert_almost_equal(phi_, phi_rec)
    assert_almost_equal(np.diag(Bdiag_), np.sort(eigvals)[::-1])


def test_bingham_equal_to_watson(beta_fraction=0):
    # test if bingham with beta=0 equals watson distribution
    mu_ = np.random.rand(2)
    n_cart = utils.sphere2cart(np.r_[1., mu_])
    psi_ = np.random.rand() * np.pi
    odi_ = np.random.rand()
    bingham = distributions.SD2Bingham(mu=mu_, psi=psi_,
                                       odi=odi_,
                                       beta_fraction=beta_fraction)
    watson = distributions.SD1Watson(mu=mu_, odi=odi_)
    Bn = bingham(n=n_cart)
    Wn = watson(n=n_cart)
    assert_almost_equal(Bn, Wn, 4)
