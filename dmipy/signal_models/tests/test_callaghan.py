from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_array_almost_equal)
import numpy as np
from dmipy.signal_models import cylinder_models, plane_models
from dmipy.core.acquisition_scheme import (
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

    soderman = cylinder_models.C2CylinderStejskalTannerApproximation(
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

    plane = plane_models.P3PlaneCallaghanApproximation(diameter=length)
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
        diameter=length, number_of_roots=50)

    E_callaghan = callaghan(scheme)
    E_free_diffusion = np.exp(-scheme.bvalues * diffusion_perpendicular)
    assert_equal(np.max(np.abs(E_callaghan - E_free_diffusion)) < 0.01, True)


def test_callaghan_equivalent_to_StejskalTanner_with_one_root_and_function(
        samples=100):
    diameter = 10e-6
    diffusion_constant = 1.7e-09

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(1, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 1e5, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    stejskaltanner = sphere_models.S2SphereStejskalTannerApproximation(
        diameter=diameter)
    callaghan = sphere_models._S3SphereCallaghanApproximation(
        number_of_roots=1, number_of_functions=1,
        diameter=diameter, diffusion_constant=diffusion_constant)

    E_stejskaltanner = stejskaltanner(scheme)
    E_callaghan = callaghan(scheme)
    assert_array_almost_equal(E_stejskaltanner, E_callaghan)


def test_RT0P_to_diameter_callaghan(samples=10000):
    diameter = 1e-5
    diffusion_constant = 1.7e-09

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(1, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 2e6, samples)
    k = 2 * np.pi * qvals_perp
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    callaghan = sphere_models._S3SphereCallaghanApproximation(
        diameter=diameter, diffusion_constant=diffusion_constant,
        number_of_roots=20, number_of_functions=50
    )
    E_callaghan = callaghan(scheme)

    rtop_callaghan = (1 / (2 * np.pi**2)) * np.trapz(
        E_callaghan * k**2, x=k
    )
    # Equivalent to Eq. 54 in Simple Harmonic Oscillator Based Reconstruction
    # and Estimation for One-Dimensional q-Space Magnetic Resonance (1D-SHORE)
    # by E. Özarslan et al
    # rtop_callaghan = 4 * np.pi * np.trapz(
    #     E_callaghan * qvals_perp ** 2, x=qvals_perp
    # )

    diameter_callaghan = 2 * (3 / (4 * np.pi * rtop_callaghan)) ** (1/3)
    assert_almost_equal(diameter_callaghan, diameter, 7)


def test_callaghan_sphere_profile_narrow_pulse_not_restricted(samples=100):
    # in short diffusion times the model should approach a Gaussian
    # profile as np.exp(-b * lambda_perp)
    diffusion_constant = 1.7e-09

    delta = np.tile(1e-6, samples)
    Delta = np.tile(2e-6, samples)
    qvals_perp = np.linspace(1e-20, 20e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    # Need to fulfill the condition Dt << (4 * (diameter/2) ** 2) / 9 * np.pi
    diameter = 3/2 * (np.pi * diffusion_constant * scheme.tau[0])  ** (1/2)
    diameter *= 10

    # needed to increase the number of roots and functions to approximate
    # the gaussian function.
    callaghan = sphere_models._S3SphereCallaghanApproximation(
        number_of_roots=20, number_of_functions=50,
        diameter=diameter, diffusion_constant=diffusion_constant
    )
    E_callaghan = callaghan(scheme)

    E_free_diffusion = np.exp(-scheme.bvalues * diffusion_constant)
    assert_equal(np.max(np.abs(E_callaghan - E_free_diffusion)) < 0.01, True)