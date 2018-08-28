from dmipy.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy.distributions import distribute_models
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dmipy.core.modeling_framework import (
    MultiCompartmentModel,
    MultiCompartmentSphericalMeanModel,
    MultiCompartmentSphericalHarmonicsModel)
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
        iso_model.rotational_harmonics_representation(),
        ball.rotational_harmonics_representation(scheme))
    assert_array_almost_equal(
        iso_model(scheme),
        ball(scheme))


def test_anisotropic_response_rh_coef_attenuation(mu=[np.pi / 2, np.pi / 2]):
    zeppelin = G2Zeppelin(
        lambda_par=1.7e-9, lambda_perp=1e-9, mu=mu)
    data = zeppelin(scheme)
    aniso_model = AnisotropicTissueResponseModel(scheme, np.atleast_2d(data))

    assert_array_almost_equal(
        aniso_model.spherical_mean(),
        zeppelin.spherical_mean(scheme), 3)
    assert_array_almost_equal(
        aniso_model.rotational_harmonics_representation(),
        zeppelin.rotational_harmonics_representation(scheme), 3)
    assert_array_almost_equal(
        aniso_model(scheme, mu=mu), data, 3)


def test_anisotropic_response_rh_coef_signal(
        S0=100., mu=[np.pi / 2, np.pi / 2]):
    zeppelin = G2Zeppelin(
        lambda_par=1.7e-9, lambda_perp=1e-9, mu=mu)
    data = zeppelin(scheme) * S0
    aniso_model = AnisotropicTissueResponseModel(scheme, np.atleast_2d(data))

    assert_array_almost_equal(
        aniso_model.spherical_mean(),
        zeppelin.spherical_mean(scheme), 3)
    assert_array_almost_equal(
        aniso_model.rotational_harmonics_representation(),
        zeppelin.rotational_harmonics_representation(scheme), 3)
    assert_array_almost_equal(
        aniso_model(scheme, mu=mu), data / S0, 3)


def test_isotropic_convolution_kernel():
    ball = G1Ball(lambda_iso=2.5e-9)
    data = ball(scheme)
    iso_model = IsotropicTissueResponseModel(scheme, np.atleast_2d(data))
    model_rh = iso_model.rotational_harmonics_representation()
    A = construct_model_based_A_matrix(scheme, model_rh, 0)
    sh_coef = 1 / (2 * np.sqrt(np.pi))
    data_pred = np.dot(A, np.r_[sh_coef])
    assert_array_almost_equal(data, data_pred)


def test_tissue_response_model_multi_compartment_models():
    ball = G1Ball(lambda_iso=2.5e-9)
    data_iso = ball(scheme)
    data_iso_sm = ball.spherical_mean(scheme)
    iso_model = IsotropicTissueResponseModel(scheme, np.atleast_2d(data_iso))

    zeppelin = G2Zeppelin(
        lambda_par=1.7e-9, lambda_perp=1e-9, mu=[np.pi / 2, np.pi / 2])
    data_aniso = zeppelin(scheme)
    data_aniso_sm = zeppelin.spherical_mean(scheme)
    aniso_model = AnisotropicTissueResponseModel(
        scheme, np.atleast_2d(data_aniso))
    models = [iso_model, aniso_model]

    mc = MultiCompartmentModel(models)
    mc_smt = MultiCompartmentSphericalMeanModel(models)

    test_mc_data = 0.5 * data_iso + 0.5 * data_aniso
    test_mc_data_sm = 0.5 * data_iso_sm + 0.5 * data_aniso_sm
    test_data = [test_mc_data, test_mc_data_sm]

    params = {
        'partial_volume_0': [0.5],
        'partial_volume_1': [0.5],
        'AnisotropicTissueResponseModel_1_mu': np.array(
            [np.pi / 2, np.pi / 2])
    }

    mc_models = [mc, mc_smt]
    for model, data in zip(mc_models, test_data):
        data_mc = model(scheme, **params)
        assert_array_almost_equal(data, data_mc, 3)

    # csd model with single models
    mc_csd = MultiCompartmentSphericalHarmonicsModel([aniso_model])
    watson_mod = distribute_models.SD1WatsonDistributed(
        [aniso_model])
    watson_params = {
        'SD1Watson_1_mu': np.array(
            [np.pi / 2, np.pi / 2]),
        'SD1Watson_1_odi': .3
    }
    data_watson = watson_mod(scheme, **watson_params)
    mc_csd_fit = mc_csd.fit(scheme, data_watson)
    assert_array_almost_equal(mc_csd_fit.predict()[0], data_watson, 2)

    # csd model with multiple models
    mc_csd = MultiCompartmentSphericalHarmonicsModel(models)
    watson_mod = distribute_models.SD1WatsonDistributed(
        [iso_model, aniso_model])
    watson_params = {
        'SD1Watson_1_mu': np.array(
            [np.pi / 2, np.pi / 2]),
        'SD1Watson_1_odi': .3,
        'partial_volume_0': 0.5
    }
    data_watson = watson_mod(scheme, **watson_params)
    mc_csd_fit = mc_csd.fit(scheme, data_watson)
    assert_array_almost_equal(mc_csd_fit.predict()[0], data_watson, 2)
