from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy.signal_models.tissue_response_models import (
    estimate_TR1_isotropic_tissue_response_model,
    estimate_TR2_anisotropic_tissue_response_model)
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme,)
import numpy as np
scheme = wu_minn_hcp_acquisition_scheme()


def test_fit_S0_response(S0_iso=10., S0_aniso=1.):
    ball = G1Ball(lambda_iso=3e-9)
    data_iso = S0_iso * ball(scheme)
    S0_iso, iso_model = estimate_TR1_isotropic_tissue_response_model(
        scheme, np.atleast_2d(data_iso))

    zeppelin = G2Zeppelin(
        lambda_par=2.2e-9, lambda_perp=1e-9, mu=[np.pi / 2, np.pi / 2])
    data_aniso = S0_aniso * zeppelin(scheme)
    S0_aniso, aniso_model = estimate_TR2_anisotropic_tissue_response_model(
        scheme, np.atleast_2d(data_aniso))

    mccsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model])
    mtcsd = MultiCompartmentSphericalHarmonicsModel(
        models=[iso_model, aniso_model],
        S0_tissue_responses=[S0_iso, S0_aniso])

    data_to_fit = 0.3 * data_iso + 0.7 * data_aniso

    csd_fit_no_S0 = mccsd.fit(scheme, data_to_fit)
    csd_fit_S0 = mtcsd.fit(scheme, data_to_fit)
    np.testing.assert_almost_equal(
        0.3, csd_fit_S0.fitted_parameters['partial_volume_0'], 1)
    np.testing.assert_almost_equal(
        0.7, csd_fit_S0.fitted_parameters['partial_volume_1'], 1)

    # test iso volume fraction overestimated without S0 response
    np.testing.assert_(
        csd_fit_no_S0.fitted_parameters['partial_volume_0'] > 0.3)
