import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from dipy.data import get_sphere
from microstruktur.utils.spherical_mean import (
    estimate_spherical_mean_shell,
    estimate_spherical_mean_multi_shell
)
from microstruktur.signal_models import cylinder_models, spherical_mean_models, gaussian_models
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)
sphere = get_sphere().subdivide()
delta = 0.01
Delta = 0.03


def test_spherical_mean_stick_analytic_vs_numerical(
        bvalue=1e9, lambda_par=1.7e-9, mu=np.r_[0, 0]):
    bvals = np.tile(bvalue, len(sphere.vertices))
    scheme = acquisition_scheme_from_bvalues(
        bvals, sphere.vertices, delta, Delta)
    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    sm_stick_numerical = np.mean(stick(scheme))
    stick_sm = spherical_mean_models.C1StickSphericalMean(
        lambda_par=lambda_par)
    sm_stick_analytic = stick_sm(scheme)
    assert_almost_equal(sm_stick_analytic, sm_stick_numerical, 3)


def test_spherical_mean_zeppelin_analytic_vs_numerical(
        bvalue=1e9, lambda_par=1.7e-9, lambda_perp=0.8e-9, mu=np.r_[0, 0]):
    bvals = np.tile(bvalue, len(sphere.vertices))
    scheme = acquisition_scheme_from_bvalues(
        bvals, sphere.vertices, delta, Delta)

    zeppelin_sm = spherical_mean_models.G4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp)
    sm_zep_analytic = zeppelin_sm(scheme)

    zeppelin = gaussian_models.G4Zeppelin(
        lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu)
    sm_zep_numerical = np.mean(zeppelin(scheme))
    assert_almost_equal(sm_zep_analytic, sm_zep_numerical, 3)


def test_spherical_mean_stick_analytic_vs_sh(bvalue=1e9, lambda_par=1.7e-9,
                                             mu=np.r_[0, 0]):
    bvals = np.tile(bvalue, len(sphere.vertices))
    scheme = acquisition_scheme_from_bvalues(
        bvals, sphere.vertices, delta, Delta)
    stick_sm = spherical_mean_models.C1StickSphericalMean(
        lambda_par=lambda_par)
    sm_stick_analytic = stick_sm(scheme)

    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    E_stick = stick(scheme)
    sm_zep_sh = estimate_spherical_mean_shell(E_stick, sphere.vertices)
    assert_almost_equal(sm_stick_analytic, sm_zep_sh, 3)


def test_spherical_mean_zeppelin_analytic_vs_sh(
        bvalue=1e9, lambda_par=1.7e-9, lambda_perp=0.8e-9, mu=np.r_[0, 0]):
    bvals = np.tile(bvalue, len(sphere.vertices))
    scheme = acquisition_scheme_from_bvalues(
        bvals, sphere.vertices, delta, Delta)
    zeppelin_sm = spherical_mean_models.G4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    sm_zep_analytic = zeppelin_sm(scheme)

    zeppelin = gaussian_models.G4Zeppelin(
        lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu
    )
    E_zep = zeppelin(scheme)
    sm_zep_sh = estimate_spherical_mean_shell(E_zep, sphere.vertices)
    assert_almost_equal(sm_zep_analytic, sm_zep_sh, 3)


def test_restricted_vs_regular_zeppelin_analytic(
    bvalue=1e9, lambda_par=1.7e-9, lambda_perp=0.8e-9, lambda_inf=0.8e-9,
    A=0.
):
    scheme = acquisition_scheme_from_bvalues(
        np.r_[bvalue], np.array([[1., 0., 0.]]), delta, Delta)

    rest_zeppelin_sm = (
        spherical_mean_models.G5RestrictedZeppelinSphericalMean(
            lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
    )
    E_rest_zep_analytic = rest_zeppelin_sm(scheme)

    zeppelin_sm = spherical_mean_models.G4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp)
    E_zep_analytic = zeppelin_sm(scheme)
    assert_almost_equal(E_rest_zep_analytic, E_zep_analytic)


def test_restricted_spherical_mean_zeppelin_analytic_vs_sh(
    bvalue=1e9, lambda_par=1.7e-9, lambda_inf=0.8e-9, mu=np.r_[0, 0], A=1e-12
):
    N_samples = len(sphere.vertices)
    bvals = np.tile(bvalue, N_samples)
    n = sphere.vertices
    scheme = acquisition_scheme_from_bvalues(
        bvals, n, delta, Delta)

    rest_zeppelin_sm = (
        spherical_mean_models.G5RestrictedZeppelinSphericalMean(
            lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)
    )
    sm_rest_zep_analytic = rest_zeppelin_sm(scheme)

    zeppelin = gaussian_models.G5RestrictedZeppelin(
        mu=mu, lambda_par=lambda_par, lambda_inf=lambda_inf, A=A)

    E_zep = zeppelin(scheme)
    sm_zep_sh = estimate_spherical_mean_shell(E_zep, n)
    assert_almost_equal(sm_zep_sh, sm_rest_zep_analytic, 3)


def test_estimate_spherical_mean_multi_shell(
    bvalue_1=1e9, bvalue_2=15e9, lambda_par=1.7e-9, lambda_perp=0.8e-9,
    mu=np.r_[0, 0]
):
    bvals_1 = np.tile(bvalue_1, sphere.vertices.shape[0])
    bvals_2 = np.tile(bvalue_2, sphere.vertices.shape[0])
    bvals = np.hstack([bvals_1, bvals_2])
    bvecs_multi_shell = np.tile(sphere.vertices, (2, 1))
    scheme = acquisition_scheme_from_bvalues(
        bvals, bvecs_multi_shell, delta, Delta)
    zeppelin_sm = spherical_mean_models.G4ZeppelinSphericalMean(
        lambda_par=lambda_par, lambda_perp=lambda_perp
    )
    sm_zep_analytic = zeppelin_sm(scheme)

    zeppelin = zeppelin = gaussian_models.G4Zeppelin(
        lambda_par=lambda_par, lambda_perp=lambda_perp, mu=mu
    )

    E_zep = zeppelin(scheme)

    sm_multi_shell = estimate_spherical_mean_multi_shell(
        E_zep, scheme
    )

    assert_array_almost_equal(sm_multi_shell, sm_zep_analytic)
