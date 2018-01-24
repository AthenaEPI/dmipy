from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_array_almost_equal)
import numpy as np
from mipy.signal_models import cylinder_models, plane_models
from mipy.core.acquisition_scheme import (
    acquisition_scheme_from_qvalues)


def test_RTAP_to_diameter_callaghan(samples=10000):
    mu = [0, 0]
    lambda_par = 1.7
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    callaghan = cylinder_models.C3CylinderCallaghanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)

    E_callaghan = callaghan(scheme)

    rtap_callaghan = 2 * np.pi * np.trapz(
        E_callaghan * qvals_perp, x=qvals_perp
    )

    diameter_callaghan = 2 / np.sqrt(np.pi * rtap_callaghan)
    assert_almost_equal(diameter_callaghan, diameter, 7)


def test_callaghan_profile_narrow_pulse_not_restricted(samples=100):
    # in short diffusion times the model should approach a Gaussian
    # profile as np.exp(-b * lambda_perp)
    mu = [0, 0]
    lambda_par = .1e-9
    diameter = 10e-5
    diffusion_perpendicular = 1.7e-09

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(.15e-2, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 3e5, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    # needed to increase the number of roots and functions to approximate
    # the gaussian function.
    callaghan = cylinder_models.C3CylinderCallaghanApproximation(
        number_of_roots=20, number_of_functions=50,
        mu=mu, lambda_par=lambda_par, diameter=diameter,
        diffusion_perpendicular=diffusion_perpendicular)

    E_callaghan = callaghan(scheme)
    E_free_diffusion = np.exp(-scheme.bvalues * diffusion_perpendicular)
    assert_equal(np.max(np.abs(E_callaghan - E_free_diffusion)) < 0.01, True)


def test_soderman_equivalent_to_callaghan_with_one_root_and_function(
        samples=100):
    mu = [0, 0]
    lambda_par = .1
    diameter = 10e-5
    diffusion_perpendicular = 1.7e-09

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(.15e-2, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 1e5, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    soderman = cylinder_models.C2CylinderSodermanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)
    callaghan = cylinder_models.C3CylinderCallaghanApproximation(
        number_of_roots=1, number_of_functions=1,
        mu=mu, lambda_par=lambda_par, diameter=diameter,
        diffusion_perpendicular=diffusion_perpendicular)

    E_soderman = soderman(scheme)
    E_callaghan = callaghan(scheme)
    assert_array_almost_equal(E_soderman, E_callaghan)


def test_RTPP_to_length_callaghan(samples=1000):
    length = 10e-6

    delta = 1e-10  # delta towards zero
    Delta = 1e10  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    plane = plane_models.P3PlaneCallaghanApproximation(length=length)
    E_callaghan = plane(scheme)

    rtpp_callaghan = 2 * np.trapz(E_callaghan, x=qvals_perp)

    length_callaghan = 1 / rtpp_callaghan
    assert_almost_equal(length_callaghan, length, 7)


def test_callaghan_plane_profile_narrow_pulse_not_restricted(samples=100):
    # in short diffusion times the model should approach a Gaussian
    # profile as np.exp(-b * lambda_perp)
    length = 10e-5
    diffusion_perpendicular = 1.7e-09

    delta = 1e-4  # delta towards zero
    Delta = 2e-4  # Delta also small
    qvals_perp = np.linspace(0, 3.0001e5, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    # needed to increase the number of roots and functions to approximate
    # the gaussian function.
    callaghan = plane_models.P3PlaneCallaghanApproximation(
        length=length, n_roots=50)

    E_callaghan = callaghan(scheme)
    E_free_diffusion = np.exp(-scheme.bvalues * diffusion_perpendicular)
    assert_equal(np.max(np.abs(E_callaghan - E_free_diffusion)) < 0.01, True)
