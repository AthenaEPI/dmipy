from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy.signal_models.tissue_response_models import (
    IsotropicTissueResponseModel,
    AnisotropicTissueResponseModel)
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme,)
import numpy as np
scheme = wu_minn_hcp_acquisition_scheme()


def test_fit_S0_response():
    ball = G1Ball(lambda_iso=3e-9)
    data_iso = 10. * ball(scheme)
    iso_model = IsotropicTissueResponseModel(scheme, np.atleast_2d(data_iso))

    zeppelin = G2Zeppelin(
        lambda_par=2.2e-9, lambda_perp=1e-9, mu=[np.pi / 2, np.pi / 2])
    data_aniso = zeppelin(scheme)
    aniso_model = AnisotropicTissueResponseModel(
        scheme, np.atleast_2d(data_aniso))

    mtcsd = MultiCompartmentSphericalHarmonicsModel([iso_model, aniso_model])

    data_to_fit = 0.3 * data_iso + 0.7 * data_aniso

    csd_fit_no_S0 = mtcsd.fit(scheme, data_to_fit, fit_S0_response=False)
    csd_fit_S0 = mtcsd.fit(scheme, data_to_fit, fit_S0_response=True)
    np.testing.assert_almost_equal(
    	0.3, csd_fit_S0.fitted_parameters['partial_volume_0'], 4)
    np.testing.assert_almost_equal(
    	0.7, csd_fit_S0.fitted_parameters['partial_volume_1'], 4)
    
    # test iso volume fraction underestimated
    np.testing.assert_(csd_fit_no_S0 < 0.3)