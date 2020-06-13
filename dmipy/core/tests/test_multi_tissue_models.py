from numpy.testing import assert_almost_equal, assert_raises

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

    def only_intra_s0_in_tortuosity():
        model1t = modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin, ball])
        model1t.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                       'C1Stick_1_lambda_par',
                                       'partial_volume_0',
                                       'partial_volume_1',
                                       S0_intra=s0s)

    assert_raises(ValueError, only_intra_s0_in_tortuosity)

    def only_extra_s0_in_tortuosity():
        model = modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin, ball])
        model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                     'C1Stick_1_lambda_par',
                                     'partial_volume_0',
                                     'partial_volume_1',
                                     S0_extra=s0z)

    assert_raises(ValueError, only_extra_s0_in_tortuosity)

    def one_tissue_but_two_s0_in_tortuosity():
        model = modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin, ball])
        model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                     'C1Stick_1_lambda_par',
                                     'partial_volume_0',
                                     'partial_volume_1',
                                     S0_intra=s0s,
                                     S0_extra=s0z)

    assert_raises(ValueError, one_tissue_but_two_s0_in_tortuosity)

    def mismatch_s0_intra():
        model = modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin, ball])
        model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                     'C1Stick_1_lambda_par',
                                     'partial_volume_0',
                                     'partial_volume_1',
                                     S0_intra=s0s + 10.,
                                     S0_extra=s0z)

    assert_raises(ValueError, mismatch_s0_intra)

    def mismatch_s0_extra():
        model = modeling_framework.MultiCompartmentModel(
            models=[stick, zeppelin, ball],
            S0_tissue_responses=[s0s, s0z, s0b])
        model.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                     'C1Stick_1_lambda_par',
                                     'partial_volume_0',
                                     'partial_volume_1',
                                     S0_intra=s0s,
                                     S0_extra=s0z + 10.)

    assert_raises(ValueError, mismatch_s0_extra)
