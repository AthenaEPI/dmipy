from dmipy.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy.signal_models.tissue_response_models import (
    IsotropicTissueResponseModel,
    AnisotropicTissueResponseModel)
import numpy as np
from numpy.testing import assert_array_almost_equal
from dmipy.optimizers_fod.construct_observation_matrix import (
    construct_model_based_A_matrix)
scheme = wu_minn_hcp_acquisition_scheme()


def test_isotropic_response():
    ball = G1Ball(lambda_iso=2.5e-9)
    data = ball(scheme)
    iso_model = IsotropicTissueResponseModel(scheme, np.atleast_2d(data))

    assert_array_almost_equal(
        iso_model.spherical_mean(),
        ball.spherical_mean(scheme))
    assert_array_almost_equal(
        iso_model.rotational_harmonics_representation()[1:],
        ball.rotational_harmonics_representation(scheme))


def test_anisotropic_response():
    zeppelin = G2Zeppelin(lambda_par=1.7e-9, lambda_perp=1e-9, mu=[.2, .7])
    data = zeppelin(scheme)
    aniso_model = AnisotropicTissueResponseModel(scheme, np.atleast_2d(data))

    assert_array_almost_equal(
        aniso_model.spherical_mean(),
        zeppelin.spherical_mean(scheme), 3)
    assert_array_almost_equal(
        aniso_model.rotational_harmonics_representation()[1:],
        zeppelin.rotational_harmonics_representation(scheme), 3)


def test_isotropic_convolution_kernel():
    ball = G1Ball(lambda_iso=2.5e-9)
    data = ball(scheme)
    iso_model = IsotropicTissueResponseModel(scheme, np.atleast_2d(data))
    model_rh = iso_model.rotational_harmonics_representation()
    A = construct_model_based_A_matrix(scheme, model_rh, 0)
    sh_coef = 1 / (2 * np.sqrt(np.pi))
    data_pred = np.dot(A, np.r_[sh_coef])
    assert_array_almost_equal(data, data_pred)
