from numpy.testing import assert_almost_equal
import numpy as np
from microstruktur.signal_models import cylinder_models
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_qvalues,)


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
