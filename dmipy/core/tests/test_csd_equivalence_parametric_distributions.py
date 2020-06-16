from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions import distribute_models, distributions
from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
from dipy.data import get_sphere
import numpy as np
from numpy.testing import (
    assert_array_almost_equal, assert_almost_equal, assert_raises, assert_)
from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy")

scheme = wu_minn_hcp_acquisition_scheme()
sphere = get_sphere('symmetric724')


def test_equivalence_csd_and_parametric_fod_tournier07(
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

    sh_fit = sh_mod.fit(scheme, data, solver='csd_tournier07')
    fod = sh_fit.fod(sphere.vertices)

    watson = distributions.SD1Watson(mu=[0., 0.], odi=0.15)
    sf = watson(sphere.vertices)
    assert_array_almost_equal(fod[0], sf, 1)

    fitted_signal = sh_fit.predict()
    assert_array_almost_equal(data, fitted_signal[0], 2)


@np.testing.dec.skipif(not have_cvxpy)
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

    sh_fit = sh_mod.fit(scheme, data, solver='csd_cvxpy', lambda_lb=0.)
    fod = sh_fit.fod(sphere.vertices)

    watson = distributions.SD1Watson(mu=[0., 0.], odi=0.15)
    sf = watson(sphere.vertices)
    assert_array_almost_equal(fod[0], sf, 2)

    fitted_signal = sh_fit.predict()
    assert_array_almost_equal(data, fitted_signal[0], 4)


@np.testing.dec.skipif(not have_cvxpy)
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

    sh_fit = sh_mod.fit(scheme, data, solver='csd_cvxpy', lambda_lb=0.)

    vf_intra_estimated = sh_fit.fitted_parameters['partial_volume_0']
    assert_almost_equal(vf_intra, vf_intra_estimated)

    predicted_signal = sh_fit.predict()

    assert_array_almost_equal(data, predicted_signal[0], 4)


def test_multi_voxel_parametric_to_sm_to_sh_fod_watson():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick, zeppelin])

    watsonstick.set_equal_parameter(
        'G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    watsonstick.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                       'G2Zeppelin_1_lambda_par',
                                       'partial_volume_0')
    mc_mod = modeling_framework.MultiCompartmentModel([watsonstick])

    parameter_dict = {
        'SD1WatsonDistributed_1_SD1Watson_1_mu': np.random.rand(10, 2),
        'SD1WatsonDistributed_1_partial_volume_0': np.linspace(0.1, 0.9, 10),
        'SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par':
        np.linspace(1.5, 2.5, 10) * 1e-9,
        'SD1WatsonDistributed_1_SD1Watson_1_odi': np.linspace(0.3, 0.7, 10)
    }

    data = mc_mod.simulate_signal(scheme, parameter_dict)

    sm_mod = modeling_framework.MultiCompartmentSphericalMeanModel(
        [stick, zeppelin])
    sm_mod.set_equal_parameter(
        'G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    sm_mod.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp', 'G2Zeppelin_1_lambda_par',
        'partial_volume_0', 'partial_volume_1')

    sf_watson = []
    for mu, odi in zip(
            parameter_dict['SD1WatsonDistributed_1_SD1Watson_1_mu'],
            parameter_dict['SD1WatsonDistributed_1_SD1Watson_1_odi']):
        watson = distributions.SD1Watson(mu=mu, odi=odi)
        sf_watson.append(watson(sphere.vertices))
    sf_watson = np.array(sf_watson)

    sm_fit = sm_mod.fit(scheme, data)
    sh_mod = sm_fit.return_spherical_harmonics_fod_model()

    sh_fit_auto = sh_mod.fit(scheme, data)  # will pick tournier
    fod_tournier = sh_fit_auto.fod(sphere.vertices)
    assert_array_almost_equal(fod_tournier, sf_watson, 1)

    sh_fit_tournier = sh_mod.fit(
        scheme, data, solver='csd_tournier07', unity_constraint=False)
    fod_tournier = sh_fit_tournier.fod(sphere.vertices)
    assert_array_almost_equal(fod_tournier, sf_watson, 1)

    sh_fit_cvxpy = sh_mod.fit(
        scheme, data, solver='csd_cvxpy', unity_constraint=True, lambda_lb=0.)
    fod_cvxpy = sh_fit_cvxpy.fod(sphere.vertices)
    assert_array_almost_equal(fod_cvxpy, sf_watson, 2)

    sh_fit_cvxpy = sh_mod.fit(
        scheme, data, solver='csd_cvxpy', unity_constraint=False, lambda_lb=0.)
    fod_cvxpy = sh_fit_cvxpy.fod(sphere.vertices)
    assert_array_almost_equal(fod_cvxpy, sf_watson, 2)


@np.testing.dec.skipif(not have_cvxpy)
def test_laplacian_and_AI_with_regularization(
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

    for solver in ['csd_tournier07', 'csd_cvxpy']:
        sh_fit = sh_mod.fit(scheme, data, solver=solver, lambda_lb=0.)
        sh_fit_reg = sh_mod.fit(scheme, data, solver=solver, lambda_lb=1e-3)
        ai = sh_fit.anisotropy_index()
        lb = sh_fit.norm_of_laplacian_fod()
        ai_reg = sh_fit_reg.anisotropy_index()
        lb_reg = sh_fit_reg.norm_of_laplacian_fod()
        assert_(ai > ai_reg)
        assert_(lb > lb_reg)


@np.testing.dec.skipif(not have_cvxpy)
def test_spherical_harmonics_model_raises(
        odi=0.15, mu=[0., 0.], lambda_par=1.7e-9):
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])

    params = {'SD1Watson_1_odi': odi,
              'SD1Watson_1_mu': mu,
              'C1Stick_1_lambda_par': lambda_par}
    data = watsonstick(scheme, **params)

    assert_raises(
        ValueError,
        modeling_framework.MultiCompartmentSphericalHarmonicsModel,
        [ball])

    sh_mod = modeling_framework.MultiCompartmentSphericalHarmonicsModel(
        [stick])
    assert_raises(ValueError, sh_mod.fit, scheme, data, solver='csd_cvxpy')

    sh_mod = modeling_framework.MultiCompartmentSphericalHarmonicsModel(
        [stick, ball])
    sh_mod.set_fixed_parameter('C1Stick_1_lambda_par', lambda_par)
    sh_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    assert_raises(
        ValueError, sh_mod.fit, scheme, data, solver='csd_tournier07')


def test_equivalence_sh_distributed_mc_with_mcsh():
    """
    We test if we can input a Watson-distributed zeppelin and stick into an
    SD3SphericalHarmonicsDistributedModel in an MC-model, and compare it with
    an MCSH model with the same watson distribution as a kernel.
    """
    stick = cylinder_models.C1Stick()
    zep = gaussian_models.G2Zeppelin()

    mck_dist = distribute_models.SD1WatsonDistributed([stick, zep])
    mck_dist.set_equal_parameter(
        'G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
    mck_dist.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp', 'G2Zeppelin_1_lambda_par',
        'partial_volume_0')

    mcsh = modeling_framework.MultiCompartmentSphericalHarmonicsModel(
        models=[mck_dist], sh_order=8)
    mc = modeling_framework.MultiCompartmentModel(
        [distribute_models.SD3SphericalHarmonicsDistributed(
            [mck_dist], sh_order=8)])

    lambda_par = 0.
    odi = .02
    sh_coeff = np.ones(45)
    sh_coeff[0] = 1 / (2 * np.sqrt(np.pi))
    pv0 = .3

    params_mcsh = {
        'SD1WatsonDistributed_1_partial_volume_0': pv0,
        'SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par': lambda_par,
        'SD1WatsonDistributed_1_SD1Watson_1_odi': odi,
        'sh_coeff': sh_coeff
    }

    basemod = 'SD3SphericalHarmonicsDistributed_1_'
    params_mc = {
        basemod + 'SD1WatsonDistributed_1_partial_volume_0': pv0,
        basemod + 'SD1WatsonDistributed_1_G2Zeppelin_1_lambda_par': lambda_par,
        basemod + 'SD1WatsonDistributed_1_SD1Watson_1_odi': odi,
        basemod + 'SD3SphericalHarmonics_1_sh_coeff': sh_coeff
    }

    E_mcsh = mcsh.simulate_signal(scheme, params_mcsh)
    E_mc = mc.simulate_signal(scheme, params_mc)

    np.testing.assert_array_almost_equal(E_mcsh, E_mc)
