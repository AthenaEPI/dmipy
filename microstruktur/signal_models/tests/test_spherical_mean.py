import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from dipy.data import get_sphere
from microstruktur.signal_models.spherical_mean import (
    estimate_spherical_mean_shell,
    estimate_spherical_mean_multi_shell
)
from microstruktur.signal_models import three_dimensional_models
sphere = get_sphere().subdivide()


def test_spherical_mean_stick_analytic_vs_numerical(bvalue=1e9,
                                                    lambda_par=1.7e-9,
                                                    mu=np.r_[0, 0]):
    stick = three_dimensional_models.I1Stick(mu=mu, lambda_par=lambda_par)
    sm_stick_numerical = np.mean(stick(bvals=bvalue, n=sphere.vertices))
    stick_sm = three_dimensional_models.I1StickSphericalMean(
        lambda_par=lambda_par
    )
    sm_stick_analytic = stick_sm(bvals=np.r_[bvalue])
    assert_almost_equal(sm_stick_analytic, sm_stick_numerical, 3)


def test_spherical_mean_zeppelin_analytic_vs_numerical(bvalue=1e9,
                                                       lambda_par=1.7e-9,
                                                       lambda_perp=0.8e-9,
                                                       mu=np.r_[0, 0]):
    zeppelin_sm = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    sm_zep_analytic = zeppelin_sm(bvals=bvalue)

    zeppelin = three_dimensional_models.E4Zeppelin(
        lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu
    )
    bvals_ = np.tile(bvalue, sphere.vertices.shape[0])
    sm_zep_numerical = np.mean(zeppelin(bvals=bvals_, n=sphere.vertices))
    assert_almost_equal(sm_zep_analytic, sm_zep_numerical, 3)


def test_spherical_mean_stick_analytic_vs_sh(bvalue=1e9, lambda_par=1.7e-9,
                                             mu=np.r_[0, 0]):
    stick_sm = three_dimensional_models.I1StickSphericalMean(
        lambda_par=lambda_par
    )
    sm_stick_analytic = stick_sm(bvals=np.r_[bvalue])

    stick = three_dimensional_models.I1Stick(mu=mu, lambda_par=lambda_par)
    E_stick = stick(bvals=bvalue, n=sphere.vertices)
    sm_zep_sh = estimate_spherical_mean_shell(E_stick, sphere.vertices)
    assert_almost_equal(sm_stick_analytic, sm_zep_sh, 3)


def test_spherical_mean_zeppelin_analytic_vs_sh(bvalue=1e9, lambda_par=1.7e-9,
                                                lambda_perp=0.8e-9,
                                                mu=np.r_[0, 0]):
    zeppelin_sm = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    sm_zep_analytic = zeppelin_sm(bvals=bvalue)

    zeppelin = three_dimensional_models.E4Zeppelin(
        lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu
    )
    bvals_ = np.tile(bvalue, sphere.vertices.shape[0])
    E_zep = zeppelin(bvals=bvals_, n=sphere.vertices)
    sm_zep_sh = estimate_spherical_mean_shell(E_zep, sphere.vertices)
    assert_almost_equal(sm_zep_analytic, sm_zep_sh, 3)


def test_restricted_vs_regular_zeppelin_analytic(
    bvalue=1e9, lambda_par=1.7e-9, lambda_perp=0.8e-9, lambda_inf=0.8e-9,
    A=0.
):
    rest_zeppelin_sm = (
        three_dimensional_models.E5RestrictedZeppelinSphericalMean(
            lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
    )
    delta = 0.01
    Delta = 0.03
    E_rest_zep_analytic = rest_zeppelin_sm(bvalue, delta=delta, Delta=Delta)

    zeppelin_sm = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp)
    E_zep_analytic = zeppelin_sm(bvalue)
    assert_almost_equal(E_rest_zep_analytic, E_zep_analytic)


def test_restricted_spherical_mean_zeppelin_analytic_vs_sh(
    bvalue=1e9, lambda_par=1.7e-9, lambda_inf=0.8e-9, mu=np.r_[0, 0], A=1e-12
):
    rest_zeppelin_sm = (
        three_dimensional_models.E5RestrictedZeppelinSphericalMean(
            lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
    )
    delta = 0.01
    Delta = 0.03
    sm_rest_zep_analytic = rest_zeppelin_sm(bvalue, delta=delta, Delta=Delta)

    zeppelin = three_dimensional_models.E5RestrictedZeppelin(
        mu=mu, lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
    N_samples = len(sphere.vertices)
    bvals_ = np.tile(bvalue, N_samples)
    delta_ = np.tile(delta, N_samples)
    Delta_ = np.tile(Delta, N_samples)
    E_zep = zeppelin(bvals=bvals_, n=sphere.vertices,
                     delta=delta_, Delta=Delta_)
    sm_zep_sh = estimate_spherical_mean_shell(E_zep, sphere.vertices)
    assert_almost_equal(sm_zep_sh, sm_rest_zep_analytic, 3)


def test_estimate_spherical_mean_multi_shell(
    bvalue_1=1e9, bvalue_2=15e9, lambda_par=1.7e-9, lambda_perp=0.8e-9,
    mu=np.r_[0, 0]
):
    zeppelin_sm = three_dimensional_models.E4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    sm_zep_analytic_1 = zeppelin_sm(bvals=bvalue_1)
    sm_zep_analytic_2 = zeppelin_sm(bvals=bvalue_2)

    zeppelin = zeppelin = three_dimensional_models.E4Zeppelin(
        lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu
    )
    bvals_1 = np.tile(bvalue_1, sphere.vertices.shape[0])
    bvals_2 = np.tile(bvalue_2, sphere.vertices.shape[0])
    E_zep_1 = zeppelin(bvals=bvals_1, n=sphere.vertices)
    E_zep_2 = zeppelin(bvals=bvals_2, n=sphere.vertices)

    bvecs_multi_shell = np.tile(sphere.vertices, (2, 1))
    E_zep_multi_shell = np.r_[E_zep_1, E_zep_2]
    shell_numbers = np.r_[np.tile(1, sphere.vertices.shape[0]),
                          np.tile(2, sphere.vertices.shape[0])]

    sm_multi_shell = estimate_spherical_mean_multi_shell(
        E_zep_multi_shell, bvecs_multi_shell, shell_numbers
    )

    assert_array_almost_equal(sm_multi_shell,
                              np.r_[sm_zep_analytic_1, sm_zep_analytic_2])
