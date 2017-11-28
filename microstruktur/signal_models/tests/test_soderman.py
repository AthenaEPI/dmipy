from numpy.testing import (
    assert_almost_equal, assert_equal, assert_array_almost_equal)
import numpy as np
from scipy import stats
from microstruktur.signal_models import cylinder_models, distributions
from microstruktur.signal_models import dispersed_models
from dipy.data import get_sphere
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_qvalues,
    acquisition_scheme_from_bvalues)
sphere = get_sphere().subdivide()
DIFFUSIVITY_SCALING = 1e-9


def test_RTAP_to_diameter_soderman(samples=10000):
    mu = [0, 0]
    lambda_par = 1.7
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(
        qvals_perp, n_perp, delta, Delta)

    soderman = cylinder_models.C2CylinderSodermanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)

    E_soderman = soderman(scheme)

    rtap_soderman = 2 * np.pi * np.trapz(
        E_soderman * qvals_perp, x=qvals_perp
    )

    diameter_soderman = 2 / np.sqrt(np.pi * rtap_soderman)

    assert_almost_equal(diameter_soderman, diameter, 7)


def test_watson_dispersed_soderman_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0
):

    # testing uniformly dispersed watson zeppelin.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))
    scheme = acquisition_scheme_from_bvalues(
        bvals, n, delta, Delta)

    watson_soderman = dispersed_models.SD1C2WatsonDispersedSodermanCylinder(
        mu=mu, kappa=kappa, lambda_par=lambda_par, diameter=diameter)
    E_watson_soderman = watson_soderman(scheme)
    # All values are the same:
    assert_almost_equal(E_watson_soderman - E_watson_soderman[0], 0)


def test_bingham_dispersed_soderman_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0, beta=0,
    psi=0
):
    # testing uniformly dispersed bingham zeppelin.
    n = sphere.vertices
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))
    scheme = acquisition_scheme_from_bvalues(
        bvals, n, delta, Delta)

    bingham_soderman = dispersed_models.SD2C2BinghamDispersedSodermanCylinder(
        mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par,
        diameter=diameter
    )
    E_bingham_soderman = bingham_soderman(scheme)
    # All values are the same:
    assert_almost_equal(E_bingham_soderman - E_bingham_soderman[0], 0)


def test_gamma_distributed_soderman(alpha=.1, beta=1e-5,
                                    radius_integral_steps=35,
                                    samples=100,
                                    mu=[0, 0],
                                    lambda_par=.1):

    delta = np.tile(1e-3, samples)  # delta towards zero
    Delta = np.tile(20e-3, samples)  # Delta towards infinity
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    qvals_perp = np.linspace(0, 3e5, samples)
    scheme = acquisition_scheme_from_qvalues(
        qvals_perp, n_perp, delta, Delta)

    DD1 = distributions.DD1GammaDistribution(alpha=alpha, beta=beta)
    soderman = cylinder_models.C2CylinderSodermanApproximation(
        mu=mu, lambda_par=lambda_par
    )
    DD1I2 = dispersed_models.DD1C2GammaDistributedSodermanCylinder(
        mu=mu, lambda_par=lambda_par, alpha=.1, beta=1e-5)

    gamma_dist = stats.gamma(alpha, scale=beta)
    radius_max = gamma_dist.mean() + 6 * gamma_dist.std()

    radii = np.linspace(1e-50, radius_max, radius_integral_steps)
    area = np.pi * radii ** 2

    radii_pdf = DD1(radii * 2)
    radii_pdf_area = radii_pdf * area
    radii_pdf_normalized = (
        radii_pdf_area /
        np.trapz(x=radii, y=radii_pdf_area)
    )

    E = np.empty(
        (radius_integral_steps, scheme.number_of_measurements),
        dtype=float
    )
    for i, radius in enumerate(radii):
        E[i] = (
            radii_pdf_normalized[i] *
            soderman(scheme, diameter=radius * 2)
        )

    E_manual = np.trapz(E, x=radii, axis=0)
    E_func = DD1I2(scheme)

    assert_array_almost_equal(E_manual, E_func)
