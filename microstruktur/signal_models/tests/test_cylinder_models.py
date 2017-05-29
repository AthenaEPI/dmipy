from numpy.testing import assert_almost_equal
import numpy as np
from microstruktur.signal_models import three_dimensional_models
from microstruktur.signal_models.gradient_conversions import b_from_q
DIFFUSIVITY_SCALING = 1e-9

def test_RTAP_to_diameter_soderman_callaghan(samples = 10000):
    mu=[0,0]
    lambda_par=1.7
    diameter=10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    bvals_perp = b_from_q(qvals_perp, delta, Delta)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))

    soderman = three_dimensional_models.I2CylinderSodermanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)
    callaghan = three_dimensional_models.I3CylinderCallaghanApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)

    E_soderman = soderman(bvals_perp, n_perp, delta=delta, Delta=Delta)
    E_callaghan = callaghan(bvals_perp, n_perp, delta=delta, Delta=Delta)

    rtap_soderman = 2 * np.pi * np.trapz(
        E_soderman * qvals_perp, x=qvals_perp
    )
    rtap_callaghan = 2 * np.pi * np.trapz(
        E_callaghan * qvals_perp, x=qvals_perp
    )

    diameter_soderman = 2 / np.sqrt(np.pi * rtap_soderman)
    diameter_callaghan = 2 / np.sqrt(np.pi * rtap_callaghan)

    assert_almost_equal(diameter_soderman, diameter, 7)
    assert_almost_equal(diameter_callaghan, diameter, 7)
