from numpy.testing import assert_almost_equal, assert_equal
from dipy.core.geometry import cart2sphere, sphere2cart
import numpy as np
from microstruktur.signal_models.utils import (
    rotation_matrix_100_to_theta_phi, rotation_matrix_around_100,
    rotation_matrix_100_to_theta_phi_psi
)
from microstruktur.signal_models.three_dimensional_models import (
    SD2_bingham_cartesian, SD3_watson
)


def test_rotation_100_to_theta_phi():
    # test 1: does R100_to_theta_phi rotate a vector theta_phi?
    theta_ = np.random.rand() * np.pi
    phi_ = (np.random.rand() - .5) * np.pi
    R100_to_theta_pi = rotation_matrix_100_to_theta_phi(theta_, phi_)
    x_, y_, z_ = np.dot(R100_to_theta_pi, np.r_[1, 0, 0])
    _, theta_rec, phi_rec = cart2sphere(x_, y_, z_)
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
    x_, y_, z_ = np.dot(R_, np.r_[1, 0, 0])
    _, theta_rec, phi_rec = cart2sphere(x_, y_, z_)
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
    _, theta_rec0, phi_rec0 = cart2sphere(main_evec[0], main_evec[1],
                                          main_evec[2])

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


def test_bingham_equal_to_watson(beta_=0):
    # test if bingham with beta=0 equals watson distribution
    n_ = np.random.rand(3)
    n_ /= np.linalg.norm(n_)
    theta_ = np.random.rand() * np.pi
    phi_ = (np.random.rand() - 0.5) * np.pi
    psi_ = np.random.rand() * np.pi
    kappa_ = np.random.rand()
    mu_ = sphere2cart(1., theta_, phi_)
    Bn = SD2_bingham_cartesian(n_, mu_, psi_, kappa_, beta_)
    Wn = SD3_watson(n_, mu_, kappa_)
    assert_almost_equal(Bn, Wn)
