import numpy as np
from numpy.testing import assert_almost_equal
from dipy.data import get_sphere
from microstruktur.signal_models.spherical_mean import (
  spherical_mean_stick, spherical_mean_zeppelin, estimate_spherical_mean_shell)
from microstruktur.signal_models.three_dimensional_models import (I1_stick,
                                                                  E4_zeppelin)

sphere = get_sphere().subdivide()


def test_spherical_mean_stick_analytic_vs_numerical(bvalue=1e3,
                                                    lambda_par=1.7e-3,
                                                    mu=np.r_[0, 0, 1]):
    sm_stick_analytic = spherical_mean_stick(bvalue, lambda_par)
    sm_stick_numerical = np.mean(I1_stick(bvalue, sphere.vertices, mu,
                                          lambda_par))
    assert_almost_equal(sm_stick_analytic, sm_stick_numerical, 3)


def test_spherical_mean_zeppelin_analytic_vs_numerical(bvalue=1e3,
                                                       lambda_par=1.7e-3,
                                                       lambda_perp=0.8e-3,
                                                       mu=np.r_[0, 0, 1]):
    sm_zep_analytic = spherical_mean_zeppelin(bvalue, lambda_par, lambda_perp)
    sm_zep_numerical = np.mean(E4_zeppelin(bvalue, sphere.vertices, mu,
                                           lambda_par, lambda_perp))
    assert_almost_equal(sm_zep_analytic, sm_zep_numerical, 3)


def test_spherical_mean_stick_analytic_vs_sh(bvalue=1e3, lambda_par=1.7e-3,
                                             mu=np.r_[0, 0, 1]):
    sm_stick_analytic = spherical_mean_stick(bvalue, lambda_par)
    E_stick = I1_stick(bvalue, sphere.vertices, mu, lambda_par)
    sm_zep_sh = estimate_spherical_mean_shell(E_stick, sphere.vertices)
    assert_almost_equal(sm_stick_analytic, sm_zep_sh, 3)


def test_spherical_mean_zeppelin_analytic_vs_sh(bvalue=1e3, lambda_par=1.7e-3,
                                                lambda_perp=0.8e-3,
                                                mu=np.r_[0, 0, 1]):
    sm_zep_analytic = spherical_mean_zeppelin(bvalue, lambda_par, lambda_perp)
    E_zep = E4_zeppelin(bvalue, sphere.vertices, mu, lambda_par, lambda_perp)
    sm_zep_sh = estimate_spherical_mean_shell(E_zep, sphere.vertices)
    sm_zep_numerical = np.mean(E4_zeppelin(bvalue, sphere.vertices, mu,
                                           lambda_par, lambda_perp))
    assert_almost_equal(sm_zep_analytic, sm_zep_sh, 3)
