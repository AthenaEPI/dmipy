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


def test_free_water_elimination(S0_iso=10., S0_aniso=1.):
    ball = G1Ball(lambda_iso=3e-9)
    data_iso = S0_iso * ball(scheme)
    iso_model = IsotropicTissueResponseModel(scheme, np.atleast_2d(data_iso))

    zeppelin = G2Zeppelin(
        lambda_par=2.2e-9, lambda_perp=1e-9, mu=[np.pi / 2, np.pi / 2])
    data_aniso = S0_aniso * zeppelin(scheme)
    aniso_model = AnisotropicTissueResponseModel(
        scheme, np.atleast_2d(data_aniso))

    mtcsd = MultiCompartmentSphericalHarmonicsModel([iso_model, aniso_model])

    data_to_fit = data_iso + data_aniso

    csd_fit_S0 = mtcsd.fit(scheme, data_to_fit, fit_S0_response=True)
    data_recovered_aniso = csd_fit_S0.return_filtered_signal(
        ['partial_volume_0'])
    np.testing.assert_array_almost_equal(
        data_aniso, data_recovered_aniso[0], 1)
