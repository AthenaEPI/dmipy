from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models
from microstruktur.signal_models import dispersed_models
from microstruktur.signal_models.gradient_conversions import b_from_q
from dipy.data import get_sphere
sphere = get_sphere().subdivide()
DIFFUSIVITY_SCALING = 1e-9


def test_RTAP_to_diameter_soderman(samples=10000):
    mu = [0, 0]
    lambda_par = 1.7
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    bvals_perp = b_from_q(qvals_perp, delta, Delta)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))

    soderman = three_dimensional_models.I2CylinderSodermanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)

    E_soderman = soderman(bvals_perp, n_perp, delta=delta, Delta=Delta)

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
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))

    watson_soderman = dispersed_models.SD3I2WatsonDispersedSodermanCylinder(
        mu=mu, kappa=kappa, lambda_par=lambda_par, diameter=diameter)
    E_watson_soderman = watson_soderman(bvals=bvals, n=n,
                                        shell_indices=shell_indices,
                                        delta=delta, Delta=Delta)
    E_unique_watson_soderman = np.unique(E_watson_soderman)
    # All values are the same:
    assert_equal(len(E_unique_watson_soderman), 1)


def test_bingham_dispersed_soderman_kappa0(
    lambda_par=1.7, diameter=8e-6, bvalue=1e9, mu=[0, 0], kappa=0, beta=0,
    psi=0
):
    # testing uniformly dispersed bingham zeppelin.
    n = sphere.vertices
    shell_indices = np.ones(len(n))
    bvals = np.tile(bvalue, len(n))
    delta = np.tile(1e-2, len(bvals))
    Delta = np.tile(3e-2, len(bvals))

    bingham_soderman = dispersed_models.SD2I2BinghamDispersedSodermanCylinder(
        mu=mu, kappa=kappa, beta=beta, psi=psi, lambda_par=lambda_par,
        diameter=diameter
    )
    E_bingham_soderman = bingham_soderman(bvals=bvals, n=n,
                                          shell_indices=shell_indices,
                                          delta=delta, Delta=Delta)
    E_unique_bingham_soderman = np.unique(E_bingham_soderman)
    # All values are the same:
    assert_equal(len(E_unique_bingham_soderman), 1)
