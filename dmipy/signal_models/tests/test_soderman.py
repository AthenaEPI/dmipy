from numpy.testing import assert_almost_equal
import numpy as np
from dmipy.signal_models import plane_models, cylinder_models, sphere_models
from dmipy.core.acquisition_scheme import (
    acquisition_scheme_from_qvalues,)


def test_RTPP_to_diameter_soderman(samples=1000):
    """This tests if the RTPP of the plane relates correctly to the diameter
    of the plane."""
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(
        qvals_perp, n_perp, delta, Delta)

    soderman = plane_models.P2PlaneStejskalTannerApproximation(
        diameter=diameter)

    E_soderman = soderman(scheme)

    rtpp_soderman = 2 * np.trapz(
        E_soderman, x=qvals_perp
    )

    diameter_soderman = 1 / rtpp_soderman

    assert_almost_equal(diameter_soderman, diameter, 7)


def test_RTAP_to_diameter_soderman(samples=1000):
    """This tests if the RTAP of the cylinder relates correctly to the diameter
    of the cylinder."""
    mu = [0, 0]
    lambda_par = 1.7
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(
        qvals_perp, n_perp, delta, Delta)

    soderman = cylinder_models.C2CylinderStejskalTannerApproximation(
        mu=mu, lambda_par=lambda_par, diameter=diameter)

    E_soderman = soderman(scheme)

    rtap_soderman = 2 * np.pi * np.trapz(
        E_soderman * qvals_perp, x=qvals_perp
    )

    diameter_soderman = 2 / np.sqrt(np.pi * rtap_soderman)

    assert_almost_equal(diameter_soderman, diameter, 7)


def test_RTOP_to_diameter_soderman(samples=1000):
    """This tests if the RTAP of the sphere relates correctly to the diameter
    of the sphere."""
    diameter = 10e-6

    delta = np.tile(1e-10, samples)  # delta towards zero
    Delta = np.tile(1e10, samples)  # Delta towards infinity
    qvals_perp = np.linspace(0, 10e6, samples)
    n_perp = np.tile(np.r_[1., 0., 0.], (samples, 1))
    scheme = acquisition_scheme_from_qvalues(
        qvals_perp, n_perp, delta, Delta)

    soderman = sphere_models.S2SphereStejskalTannerApproximation(
        diameter=diameter)

    E_soderman = soderman(scheme)

    rtop_soderman = 4 * np.pi * np.trapz(
        E_soderman * qvals_perp ** 2, x=qvals_perp
    )

    sphere_volume = 1. / rtop_soderman
    diameter_soderman = 2 * (sphere_volume / ((4. / 3.) * np.pi)) ** (1. / 3.)

    assert_almost_equal(diameter_soderman, diameter, 7)
