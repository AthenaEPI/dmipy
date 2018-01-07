from mipy.data import saved_data, saved_acquisition_schemes
from mipy.signal_models import (
    spherical_mean_models, cylinder_models, gaussian_models, dispersed_models)
from mipy.utils.utils import parameter_equality, T1_tortuosity
from mipy.core import modeling_framework
from numpy.testing import assert_equal
import numpy as np


scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
camino_parallel = saved_data.synthetic_camino_data_parallel()
camino_dispersed = saved_data.synthetic_camino_data_dispersed()


def test_spherical_mean_stick_tortuous_zeppelin():
    stick_sm = spherical_mean_models.C1StickSphericalMean()
    zeppelin_sm = spherical_mean_models.G4ZeppelinSphericalMean()

    parameter_links = [
        [zeppelin_sm, 'lambda_perp', T1_tortuosity, [
            (None, 'partial_volume_0'), (None, 'partial_volume_1'), (stick_sm, 'lambda_par')]],
        [zeppelin_sm, 'lambda_par', parameter_equality,
         [(stick_sm, 'lambda_par')]]]

    mc_smt = modeling_framework.MultiCompartmentModel(
        acquisition_scheme=scheme,
        models=[stick_sm, zeppelin_sm], parameter_links=parameter_links)

    fitted_params_par = (mc_smt.fit(
        camino_parallel.signal_attenuation[::20]).fitted_parameters)
    fitted_params_disp = (mc_smt.fit(
        camino_dispersed.signal_attenuation[::40]).fitted_parameters)

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
    zeppelin = gaussian_models.G4Zeppelin()

    parameter_links = [
        [zeppelin, 'lambda_perp', T1_tortuosity, [
            (None, 'partial_volume_0'), (None, 'partial_volume_1'), (stick, 'lambda_par')]],
        [zeppelin, 'lambda_par', parameter_equality, [(stick, 'lambda_par')]],
        [zeppelin, 'mu', parameter_equality, [(stick, 'mu')]]]

    stick_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            acquisition_scheme=scheme,
            models=[stick, zeppelin],
            parameter_links=parameter_links)
    )

    parameter_guess = (
        stick_and_zeppelin.parameter_initial_guess_to_parameter_vector(
            C1Stick_1_mu=np.r_[0, 0])
    )
    fitted_params = (stick_and_zeppelin.fit(
        camino_parallel.signal_attenuation[::20],
        parameter_initial_guess=parameter_guess
    ).fitted_parameters)

    mean_abs_error = np.mean(
        abs(fitted_params['partial_volume_0'].squeeze(
        ) - camino_parallel.fractions[::20]))
    assert_equal(mean_abs_error < 0.02, True)


def test_watson_dispersed_stick_tortuous_zeppelin():
    disp_stick = dispersed_models.SD1C1WatsonDispersedStick()
    disp_zeppelin = dispersed_models.SD1G4WatsonDispersedZeppelin()

    parameter_links = [
        [disp_zeppelin, 'lambda_perp', T1_tortuosity, [
            (None, 'partial_volume_0'), (None, 'partial_volume_1'), (disp_stick, 'lambda_par')]],
        [disp_zeppelin, 'lambda_par', parameter_equality,
            [(disp_stick, 'lambda_par')]],
        [disp_zeppelin, 'mu', parameter_equality, [(disp_stick, 'mu')]],
        [disp_zeppelin, 'kappa', parameter_equality, [(disp_stick, 'kappa')]]]

    disp_stick_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            acquisition_scheme=scheme,
            models=[disp_stick, disp_zeppelin],
            parameter_links=parameter_links)
    )

    parameter_guess = (
        disp_stick_and_zeppelin.parameter_initial_guess_to_parameter_vector(
            SD1C1WatsonDispersedStick_1_mu=np.r_[0, 0])
    )

    beta0 = camino_dispersed.beta == 0.
    diff17 = camino_dispersed.diffusivities == 1.7e-9
    mask = np.all([beta0, diff17], axis=0)
    E_watson = camino_dispersed.signal_attenuation[mask]
    fractions_watson = camino_dispersed.fractions[mask]

    fitted_params = (disp_stick_and_zeppelin.fit(
        E_watson[::20],
        parameter_initial_guess=parameter_guess).fitted_parameters
    )

    mean_abs_error = np.mean(
        abs(fitted_params['partial_volume_0'].squeeze(
        ) - fractions_watson[::20]))
    assert_equal(mean_abs_error < 0.02, True)


def test_bingham_dispersed_stick_tortuous_zeppelin():
    disp_stick = dispersed_models.SD2C1BinghamDispersedStick()
    disp_zeppelin = dispersed_models.SD2G4BinghamDispersedZeppelin()

    parameter_links = [
        [disp_zeppelin, 'lambda_perp', T1_tortuosity, [
            (None, 'partial_volume_0'), (None, 'partial_volume_1'), (disp_stick, 'lambda_par')]],
        [disp_zeppelin, 'lambda_par', parameter_equality,
            [(disp_stick, 'lambda_par')]],
        [disp_zeppelin, 'mu', parameter_equality, [(disp_stick, 'mu')]],
        [disp_zeppelin, 'kappa', parameter_equality, [(disp_stick, 'kappa')]],
        [disp_zeppelin, 'beta', parameter_equality, [(disp_stick, 'beta')]],
        [disp_zeppelin, 'psi', parameter_equality, [(disp_stick, 'psi')]]]

    disp_stick_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            acquisition_scheme=scheme,
            models=[disp_stick, disp_zeppelin],
            parameter_links=parameter_links)
    )

    parameter_guess = (
        disp_stick_and_zeppelin.parameter_initial_guess_to_parameter_vector(
            SD2C1BinghamDispersedStick_1_mu=np.r_[0, 0],
            SD2C1BinghamDispersedStick_1_psi=0.,
            SD2C1BinghamDispersedStick_1_beta=0.)
    )

    beta0 = camino_dispersed.beta > 0
    diff17 = camino_dispersed.diffusivities == 1.7e-9
    mask = np.all([beta0, diff17], axis=0)
    E_watson = camino_dispersed.signal_attenuation[mask]
    fractions_watson = camino_dispersed.fractions[mask]

    fitted_params = (disp_stick_and_zeppelin.fit(
        E_watson[::200],
        parameter_initial_guess=parameter_guess).fitted_parameters
    )

    mean_abs_error = np.mean(
        abs(fitted_params['partial_volume_0'].squeeze(
        ) - fractions_watson[::200]))
    assert_equal(mean_abs_error < 0.02, True)
