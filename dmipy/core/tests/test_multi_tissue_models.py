from numpy.testing import assert_almost_equal, assert_

from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy.signal_models import cylinder_models, gaussian_models

scheme = wu_minn_hcp_acquisition_scheme()


def test_multi_tissue_mc_model():
    scheme = wu_minn_hcp_acquisition_scheme()
    ball = gaussian_models.G1Ball()
    cyl = cylinder_models.C1Stick()
    models = [ball, cyl]
    S0_responses = [1., 2.]
    mt_mc = modeling_framework.MultiCompartmentModel(
        models=models, S0_tissue_responses=S0_responses)

    mt_mc.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
    mt_mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    mt_mc.set_fixed_parameter('C1Stick_1_mu', [0., 0.])
    param_dict = {'partial_volume_0': .5, 'partial_volume_1': .5}
    E = mt_mc.simulate_signal(scheme, param_dict)

    mt_mc_fit = mt_mc.fit(scheme, E)
    sig_fracts = mt_mc_fit.fitted_parameters
    vol_fracts = mt_mc_fit.fitted_multi_tissue_fractions
    vol_fracts_norm = mt_mc_fit.fitted_multi_tissue_fractions_normalized

    assert_almost_equal(
        sig_fracts['partial_volume_0'], vol_fracts['partial_volume_0'], 2)
    assert_almost_equal(
        sig_fracts['partial_volume_1'], vol_fracts['partial_volume_1'] * 2., 2)
    assert_almost_equal(
        vol_fracts_norm['partial_volume_0'], 2 / 3., 2)
    assert_almost_equal(
        vol_fracts_norm['partial_volume_1'], 1 / 3., 2)


def test_multi_tissue_mc_sm_model():
    scheme = wu_minn_hcp_acquisition_scheme()
    ball = gaussian_models.G1Ball()
    cyl = cylinder_models.C1Stick()
    models = [ball, cyl]
    S0_responses = [1., 2.]

    # generate data
    mt_mc = modeling_framework.MultiCompartmentModel(
        models=models, S0_tissue_responses=S0_responses)
    mt_mc.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
    mt_mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    mt_mc.set_fixed_parameter('C1Stick_1_mu', [0., 0.])
    param_dict = {'partial_volume_0': .5, 'partial_volume_1': .5}
    E = mt_mc.simulate_signal(scheme, param_dict)

    # do mc-sm multi tissue model.
    mt_mc_sm = modeling_framework.MultiCompartmentSphericalMeanModel(
        models=models, S0_tissue_responses=S0_responses)
    mt_mc_sm.set_fixed_parameter('C1Stick_1_lambda_par', 1.7e-9)
    mt_mc_sm.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    mt_mc_fit = mt_mc_sm.fit(scheme, E)
    sig_fracts = mt_mc_fit.fitted_parameters
    vol_fracts = mt_mc_fit.fitted_multi_tissue_fractions
    vol_fracts_norm = mt_mc_fit.fitted_multi_tissue_fractions_normalized

    assert_almost_equal(
        sig_fracts['partial_volume_0'], vol_fracts['partial_volume_0'], 2)
    assert_almost_equal(
        sig_fracts['partial_volume_1'], vol_fracts['partial_volume_1'] * 2., 2)
    assert_almost_equal(
        vol_fracts_norm['partial_volume_0'], 2 / 3., 2)
    assert_almost_equal(
        vol_fracts_norm['partial_volume_1'], 1 / 3., 2)


def test_multi_tissue_tortuosity():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    s0s = 3000.
    s0z = 4000.
    s0b = 10000.

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball],
        S0_tissue_responses=[s0s, s0z, s0b])
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 True)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == s0s and s0ec == s0z)


def test_multi_tissue_tortuosity_no_s0():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball])
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 True)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == 1 and s0ec == 1)


def test_multi_tissue_tortuosity_no_correction():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball],
        S0_tissue_responses=[4000, 5000, 10000]
    )
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 False)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == 1 and s0ec == 1)


def test_multi_tissue_tortuosity_no_s0_no_correction():
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    ball = gaussian_models.G1Ball()

    model = modeling_framework.MultiCompartmentModel(
        models=[stick, zeppelin, ball]
    )
    model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                 'C1Stick_1_lambda_par',
                                 'partial_volume_0',
                                 'partial_volume_1',
                                 False)
    tort = model.parameter_links[0][2]
    s0ic, s0ec = tort.S0_intra, tort.S0_extra
    assert_(s0ic == 1 and s0ec == 1)
