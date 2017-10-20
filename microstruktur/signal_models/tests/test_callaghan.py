from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models
from microstruktur.signal_models import dispersed_models
from microstruktur.signal_models.gradient_conversions import b_from_q
from dipy.data import get_sphere
sphere = get_sphere().subdivide()
DIFFUSIVITY_SCALING = 1e-9


def test_RTAP_to_diameter_callaghan(samples=10000):
    mu = [0, 0]
    lambda_par = 1.7
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    bvals_perp = b_from_q(qvals_perp, delta, Delta)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))

    callaghan = three_dimensional_models.I3CylinderCallaghanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)

    E_callaghan = callaghan(bvals_perp, n_perp, delta=delta, Delta=Delta)

    rtap_callaghan = 2 * np.pi * np.trapz(
        E_callaghan * qvals_perp, x=qvals_perp
    )

    diameter_callaghan = 2 / np.sqrt(np.pi * rtap_callaghan)
    assert_almost_equal(diameter_callaghan, diameter, 7)


def test_callaghan_profile_narrow_pulse_not_restricted(samples=100):
    # in short diffusion times the model should approach a Gaussian
    # profile as np.exp(-b * lambda_perp)
    mu = [0, 0]
    lambda_par = .1
    diameter = 10e-5
    diffusion_perpendicular = 1.7e-09

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(.15e-2, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 3e5, samples)
    bvals_perp = b_from_q(qvals_perp, delta, Delta)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))

    # needed to increase the number of roots and functions to approximate
    # the gaussian function.
    callaghan = three_dimensional_models.I3CylinderCallaghanApproximation(
        number_of_roots=20, number_of_functions=50,
        mu=mu, lambda_par=lambda_par, diameter=diameter,
        diffusion_perpendicular=diffusion_perpendicular)

    E_callaghan = callaghan(bvals_perp, n_perp, delta=delta, Delta=Delta)
    E_free_diffusion = np.exp(-bvals_perp * diffusion_perpendicular)
    assert_equal(np.max(np.abs(E_callaghan - E_free_diffusion)) < 0.01, True)


def test_watson_dispersed_callaghan_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0
):

    # testing uniformly dispersed watson zeppelin.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))

    watson_callaghan = dispersed_models.SD3I3WatsonDispersedCallaghanCylinder(
        mu=mu, kappa=kappa, lambda_par=lambda_par, diameter=diameter)
    E_watson_callaghan = watson_callaghan(bvals=bvals, n=n,
                                          shell_indices=shell_indices,
                                          delta=delta, Delta=Delta)
    E_unique_watson_callaghan = np.unique(E_watson_callaghan)
    # All values are the same:
    assert_equal(len(E_unique_watson_callaghan), 1)


def test_bingham_dispersed_callaghan_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0, beta=0,
    psi=0
):
    # testing uniformly dispersed bingham zeppelin.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))

    bingham_callaghan = (
        dispersed_models.SD2I3BinghamDispersedCallaghanCylinder(
            mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par,
            diameter=diameter)
    )
    E_bingham_callaghan = bingham_callaghan(bvals=bvals, n=n,
                                            shell_indices=shell_indices,
                                            delta=delta, Delta=Delta)
    E_unique_bingham_callaghan = np.unique(E_bingham_callaghan)
    # All values are the same:
    assert_equal(len(E_unique_bingham_callaghan), 1)
