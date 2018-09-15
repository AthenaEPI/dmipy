from dmipy.signal_models.gaussian_models import G2Zeppelin
from dmipy.distributions import distribute_models
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy.signal_models.tissue_response_models import (
    AnisotropicTissueResponseModel)
from dmipy.tissue_response.white_matter_response import (
    white_matter_response_tournier07,
    white_matter_response_tournier13)
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
scheme = wu_minn_hcp_acquisition_scheme()


def _make_single_and_crossing():
    zeppelin = G2Zeppelin(
        lambda_par=1.7e-9, lambda_perp=1e-9, mu=[0., 0.])
    data_aniso = zeppelin(scheme)
    aniso_model = AnisotropicTissueResponseModel(
        scheme, np.atleast_2d(data_aniso))

    watson_mod = distribute_models.SD1WatsonDistributed(
        [aniso_model])
    watson_params_par = {
        'SD1Watson_1_mu': np.array(
            [0., 0.]),
        'SD1Watson_1_odi': .2
    }
    watson_params_perp = {
        'SD1Watson_1_mu': np.array(
            [np.pi / 2., np.pi / 2.]),
        'SD1Watson_1_odi': .2
    }
    data_watson_par = watson_mod(scheme, **watson_params_par)
    data_watson_perp = watson_mod(scheme, **watson_params_perp)

    data_cross = np.array(
        [data_watson_par, data_watson_par + data_watson_perp])
    return data_cross


def test_tournier13_picks_single_peak():
    data_cross = _make_single_and_crossing()

    assert_raises(ValueError,
                  white_matter_response_tournier13,
                  scheme, data_cross, peak_ratio_setting='bla')

    wm, _ = white_matter_response_tournier13(
        scheme, data_cross, peak_ratio_setting='mrtrix',
        N_candidate_voxels=1)
    assert_array_almost_equal(
        wm(scheme, mu=[0., 0.]),
        data_cross[0], 4)


def test_tournier07_picks_single_peak():
    data_cross = _make_single_and_crossing()
    wm, _ = white_matter_response_tournier07(
        scheme, data_cross, peak_ratio_setting='mrtrix',
        N_candidate_voxels=1)
    assert_array_almost_equal(
        wm(scheme, mu=[0., 0.]),
        data_cross[0], 4)
