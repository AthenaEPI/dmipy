from dmipy.data import saved_data
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core import modeling_framework
from dmipy.distributions import distribute_models
from numpy.testing import assert_equal
import numpy as np


scheme, camino_parallel = saved_data.synthetic_camino_data_parallel()
scheme, camino_dispersed = saved_data.synthetic_camino_data_dispersed()


def test_spherical_mean_stick_tortuous_zeppelin():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    mc_mdi = modeling_framework.MultiCompartmentSphericalMeanModel(
        models=[stick, zeppelin])

    mc_mdi.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                  'C1Stick_1_lambda_par',
                                  'partial_volume_0',
                                  'partial_volume_1')
    mc_mdi.set_equal_parameter('G2Zeppelin_1_lambda_par',
                               'C1Stick_1_lambda_par')

    fitted_params_par = (
        mc_mdi.fit(
            scheme,
            camino_parallel.signal_attenuation[::20]
        ).fitted_parameters
    )
    fitted_params_disp = (
        mc_mdi.fit(
            scheme,
            camino_dispersed.signal_attenuation[::40]
        ).fitted_parameters
    )

    mean_abs_error_par = np.mean(
        abs(fitted_params_par['partial_volume_0'].squeeze(
        ) - camino_parallel.fractions[::20]))

    mean_abs_error_disp = np.mean(
        abs(fitted_params_disp['partial_volume_0'].squeeze(
        ) - camino_dispersed.fractions[::40]))
    assert_equal(mean_abs_error_par < 0.02, True)
    assert_equal(mean_abs_error_disp < 0.02, True)


def test_stick_tortuous_zeppelin():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    stick_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin])
    )

    stick_and_zeppelin.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1'
    )
    stick_and_zeppelin.set_equal_parameter(
        'C1Stick_1_mu',
        'G2Zeppelin_1_mu'
    )

    stick_and_zeppelin.set_equal_parameter(
        'G2Zeppelin_1_lambda_par',
        'C1Stick_1_lambda_par'
    )

    fitted_params = (stick_and_zeppelin.fit(
        scheme,
        camino_parallel.signal_attenuation[::20],
    ).fitted_parameters)

    mean_abs_error = np.mean(
        abs(fitted_params['partial_volume_0'].squeeze(
        ) - camino_parallel.fractions[::20]))
    assert_equal(mean_abs_error < 0.02, True)


def test_watson_dispersed_stick_tortuous_zeppelin():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    watson_bundle = distribute_models.SD1WatsonDistributed(
        models=[stick, zeppelin])

    watson_bundle.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'G2Zeppelin_1_lambda_par',
        'partial_volume_0'
    )

    watson_bundle.set_equal_parameter(
        'G2Zeppelin_1_lambda_par',
        'C1Stick_1_lambda_par')

    watson_bundle.set_fixed_parameter(
        'G2Zeppelin_1_lambda_par', 1.7e-9)

    mc_watson = (
        modeling_framework.MultiCompartmentModel(
            models=[watson_bundle])
    )

    beta0 = camino_dispersed.beta == 0.
    diff17 = camino_dispersed.diffusivities == 1.7e-9
    mask = np.all([beta0, diff17], axis=0)
    E_watson = camino_dispersed.signal_attenuation[mask]
    fractions_watson = camino_dispersed.fractions[mask]

    fitted_params = (
        mc_watson.fit(scheme, E_watson[::20]).fitted_parameters
    )

    mean_abs_error = np.mean(
        abs(fitted_params['SD1WatsonDistributed_1_partial_volume_0'].squeeze(
        ) - fractions_watson[::20]))
    assert_equal(mean_abs_error < 0.03, True)


def test_bingham_dispersed_stick_tortuous_zeppelin():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()

    bingham_bundle = distribute_models.SD2BinghamDistributed(
        models=[stick, zeppelin])

    bingham_bundle.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'G2Zeppelin_1_lambda_par',
        'partial_volume_0'
    )

    bingham_bundle.set_equal_parameter(
        'G2Zeppelin_1_lambda_par',
        'C1Stick_1_lambda_par')

    bingham_bundle.set_fixed_parameter(
        'G2Zeppelin_1_lambda_par', 1.7e-9)

    bingham_bundle.set_fixed_parameter(
        'SD2Bingham_1_mu', [0., 0.])

    mc_bingham = (
        modeling_framework.MultiCompartmentModel(
            models=[bingham_bundle])
    )

    beta0 = camino_dispersed.beta > 0
    diff17 = camino_dispersed.diffusivities == 1.7e-9
    mask = np.all([beta0, diff17], axis=0)
    E_watson = camino_dispersed.signal_attenuation[mask]
    fractions_watson = camino_dispersed.fractions[mask]

    fitted_params = (mc_bingham.fit(scheme,
                                    E_watson[::200]).fitted_parameters
                     )

    mean_abs_error = np.mean(
        abs(fitted_params['SD2BinghamDistributed_1_partial_volume_0'].squeeze(
        ) - fractions_watson[::200]))
    assert_equal(mean_abs_error < 0.035, True)
