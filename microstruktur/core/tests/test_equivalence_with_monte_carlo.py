from microstruktur.data import saved_data, saved_acquisition_schemes
from microstruktur.signal_models import (
    spherical_mean_models, cylinder_models, gaussian_models)
from microstruktur.utils.utils import parameter_equality, T1_tortuosity
from microstruktur.core import modeling_framework
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
            (None, 'partial_volume_0'), (stick_sm, 'lambda_par')]],
        [zeppelin_sm, 'lambda_par', parameter_equality,
         [(stick_sm, 'lambda_par')]]]

    mc_smt = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[stick_sm, zeppelin_sm], parameter_links=parameter_links)

    fitted_params_par = mc_smt.parameter_vector_to_parameters(
        mc_smt.fit(camino_parallel.signal_attenuation[::20]))
    fitted_params_disp = mc_smt.parameter_vector_to_parameters(
        mc_smt.fit(camino_dispersed.signal_attenuation[::40]))

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
            (None, 'partial_volume_0'), (stick, 'lambda_par')]],
        [zeppelin, 'lambda_par', parameter_equality, [(stick, 'lambda_par')]],
        [zeppelin, 'mu', parameter_equality, [(stick, 'mu')]]]

    stick_and_zeppelin = (
        modeling_framework.MultiCompartmentMicrostructureModel(
            acquisition_scheme=scheme,
            models=[stick, zeppelin],
            parameter_links=parameter_links)
    )

    parameter_guess = (
        stick_and_zeppelin.parameter_initial_guess_to_parameter_vector(
            C1Stick_1_mu=np.r_[0, 0])
    )
    fitted_params = stick_and_zeppelin.parameter_vector_to_parameters(
        stick_and_zeppelin.fit(camino_parallel.signal_attenuation[::20],
                               parameter_initial_guess=parameter_guess)
    )

    mean_abs_error = np.mean(
        abs(fitted_params['partial_volume_0'].squeeze(
        ) - camino_parallel.fractions[::20]))
    assert_equal(mean_abs_error < 0.02, True)
