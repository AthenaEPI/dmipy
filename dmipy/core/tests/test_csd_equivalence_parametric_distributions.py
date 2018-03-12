from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions import distribute_models, distributions
from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dipy.data import get_sphere
from numpy.testing import assert_array_almost_equal, assert_almost_equal

scheme = wu_minn_hcp_acquisition_scheme()
sphere = get_sphere('symmetric724')


def test_equivalence_csd_and_parametric_fod(
        odi=0.15, mu=[0., 0.], lambda_par=1.7e-9):
    stick = cylinder_models.C1Stick()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])

    params = {'SD1Watson_1_odi': odi,
              'SD1Watson_1_mu': mu,
              'C1Stick_1_lambda_par': lambda_par}

    data = watsonstick(scheme, **params)

    sh_mod = modeling_framework.MultiCompartmentSphericalHarmonicsModel(
        [stick])
    sh_mod.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)

    sh_fit = sh_mod.fit(scheme, data)
    fod = sh_fit.fod(sphere.vertices)

    watson = distributions.SD1Watson(mu=[0., 0.], odi=0.15)
    sf = watson(sphere.vertices)
    assert_array_almost_equal(fod[0], sf, 2)

    fitted_signal = sh_fit.predict()
    assert_array_almost_equal(data, fitted_signal[0], 4)


def test_multi_compartment_fod_with_parametric_model(
        odi=0.15, mu=[0., 0.], lambda_iso=3e-9, lambda_par=1.7e-9,
        vf_intra=0.7):
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    mc_mod = modeling_framework.MultiCompartmentModel([watsonstick, ball])

    sh_mod = modeling_framework.MultiCompartmentSphericalHarmonicsModel(
        [stick, ball])
    sh_mod.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso)
    sh_mod.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)

    simulation_parameters = mc_mod.parameters_to_parameter_vector(
        G1Ball_1_lambda_iso=lambda_iso,
        SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
        SD1WatsonDistributed_1_C1Stick_1_lambda_par=lambda_par,
        SD1WatsonDistributed_1_SD1Watson_1_odi=odi,
        partial_volume_0=vf_intra,
        partial_volume_1=1 - vf_intra)
    data = mc_mod.simulate_signal(scheme, simulation_parameters)

    sh_fit = sh_mod.fit(scheme, data)

    vf_intra_estimated = sh_fit.fitted_parameters['partial_volume_0']
    assert_almost_equal(vf_intra, vf_intra_estimated)

    predicted_signal = sh_fit.predict()

    assert_array_almost_equal(data, predicted_signal[0], 4)
