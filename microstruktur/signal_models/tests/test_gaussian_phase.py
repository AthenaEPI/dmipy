from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models
from microstruktur.signal_models import dispersed_models
from microstruktur.signal_models.gradient_conversions import b_from_q
from dipy.data import get_sphere
sphere = get_sphere().subdivide()
DIFFUSIVITY_SCALING = 1e-9


def test_gaussian_phase_profile_narrow_pulse_not_restricted(samples=100):
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
    bvals_perp = b_from_q(qvals_perp, delta, Delta)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))

    vangelderen = (
        three_dimensional_models.I4CylinderGaussianPhaseApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter)
    )
    E_vangelderen = vangelderen(bvals_perp, n_perp, delta=delta, Delta=Delta)
    E_free_diffusion = np.exp(-bvals_perp * diffusion_perpendicular)
    assert_equal(np.max(np.abs(E_vangelderen - E_free_diffusion)) < 0.01, True)


def test_gaussian_phase_profile_narrow_pulse_restricted():
    # given narrow pulses and long diffusion time the model
    # approaches according to Eq. (14) in VanGelderen et al:
    # np.exp(-(gamma * G * delta * R) ** 2)
    return None


def test_watson_dispersed_gaussian_phase_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0
):

    # testing uniformly dispersed watson zeppelin.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))

    watson_gaussian_phase = (
        dispersed_models.SD3I4WatsonDispersedGaussianPhaseCylinder(
            mu=mu, kappa=kappa, lambda_par=lambda_par, diameter=diameter)
    )
    E_watson_gaussian_phase = watson_gaussian_phase(
        bvals=bvals, n=n, shell_indices=shell_indices, delta=delta,
        Delta=Delta)
    E_unique_watson_gaussian_phase = np.unique(E_watson_gaussian_phase)
    # All values are the same:
    assert_equal(len(E_unique_watson_gaussian_phase), 1)


def test_bingham_dispersed_gaussian_phase_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0, beta=0,
    psi=0
):
    # testing uniformly dispersed bingham zeppelin.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))

    bingham_gaussian_phase = (
        dispersed_models.SD2I4BinghamDispersedGaussianPhaseCylinder(
            mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par,
            diameter=diameter)
    )
    E_bingham_gaussian_phase = bingham_gaussian_phase(
        bvals=bvals, n=n, shell_indices=shell_indices,
        delta=delta, Delta=Delta)
    E_unique_bingham_gaussian_phase = np.unique(E_bingham_gaussian_phase)
    # All values are the same:
    assert_equal(len(E_unique_bingham_gaussian_phase), 1)
