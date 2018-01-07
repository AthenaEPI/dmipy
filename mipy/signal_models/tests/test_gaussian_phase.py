from numpy.testing import assert_equal
import numpy as np
from mipy.signal_models import cylinder_models, sphere_models
from mipy.core.acquisition_scheme import (
    acquisition_scheme_from_qvalues)


def test_cylinder_gaussian_phase_profile_narrow_pulse_not_restricted(samples=100):
    # the Gaussian Phase model approaches the following equation
    # when delta << tau according to Eq. (13) in VanGelderen et al:
    # np.exp(-2 * (gamma * G * delta * lambda_perp) ** 2)
    mu = [0, 0]
    lambda_par = .1
    diameter = 1e-4
    diffusion_perpendicular = 1.7e-09

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(1e-3, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 3e5, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(qvals_perp, n_perp, delta, Delta)

    vangelderen = (
        cylinder_models.C4CylinderGaussianPhaseApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter)
    )
    E_vangelderen = vangelderen(scheme)
    E_free_diffusion = np.exp(-scheme.bvalues * diffusion_perpendicular)
    assert_equal(np.max(np.abs(E_vangelderen - E_free_diffusion)) < 0.01, True)


def test_cylinder_gaussian_phase_profile_narrow_pulse_restricted():
    # given narrow pulses and long diffusion time the model
    # approaches according to Eq. (14) in VanGelderen et al:
    # np.exp(-(gamma * G * delta * R) ** 2).R
    # But... how can it be Gaussian?
    return None


def test_sphere_gaussian_phase_profile_narrow_pulse_not_restricted():
    "balinov"
    diameter = 1e-4
    sphere_GFA = (
        sphere_models.S4SphereGaussianPhaseApproximation(
            diameter=diameter)
    )
    pass


def test_cylinder_gaussian_phase_profile_narrow_pulse_restricted():
    "balinov"
    diameter = 1e-4
    sphere_GFA = (
        sphere_models.S4SphereGaussianPhaseApproximation(
            diameter=diameter)
    )
    pass
