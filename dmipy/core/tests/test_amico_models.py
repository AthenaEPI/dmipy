from dmipy.distributions import distribute_models
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD2BinghamDistributed
from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
import numpy as np
import numpy.linalg
from numpy.testing import assert_, assert_raises

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentAMICOModel, MultiCompartmentModel
scheme = wu_minn_hcp_acquisition_scheme()
from numpy.testing import assert_almost_equal

def one_compartment_amico_model():
    """
    Provides a simple Stick AMICO-like model.

    This function builds an AMICO model with a single compartment given by a
    stick convolved with a watson distribution.

    All the parameters are set randomly.

    Returns:
        tuple of length 2 with
         * MultiCompartmentAMICOModel instance
         * sample data simulated on the hcp acquisition scheme
    """
    stick = cylinder_models.C1Stick()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    params = {}
    for parameter, card, in watsonstick.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             watsonstick.parameter_scales[parameter])

    model = modeling_framework.MultiCompartmentAMICOModel([watsonstick])
    data = np.atleast_2d(watsonstick(scheme, **params))
    return model, data


def two_compartment_amico_model():
    """
    Provides a simple Stick-and-Ball AMICO-like model.

    This function builds an AMICO model with two compartments given by a
    stick convolved with a watson distribution and a ball.

    All the parameters are set randomly.

    Returns:
        tuple of length 2 with
         * MultiCompartmentAMICOModel instance
         * sample data simulated on the hcp acquisition scheme
    """
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    model = modeling_framework.MultiCompartmentAMICOModel([watsonstick, ball])

    params = {}
    for parameter, card, in model.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             model.parameter_scales[parameter])

    for key, value in params.items():
        model.set_fixed_parameter(key, value)

    data = np.atleast_2d(model(scheme, **params))
    return model, data


def two_sticks_one_ball_amico_model():
    """
    Provides a simple Stick-Stick-and-Ball AMICO-like model.

    This function builds an AMICO model with two compartments given by a
    stick convolved with a watson distribution and a ball.

    All the parameters are set randomly.

    Returns:
        tuple of length 2 with
         * MultiCompartmentAMICOModel instance
         * sample data simulated on the hcp acquisition scheme
    """
    stick1 = cylinder_models.C1Stick()
    stick2 = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    watsonstick1 = distribute_models.SD1WatsonDistributed(
        [stick1])
    watsonstick2 = distribute_models.SD1WatsonDistributed(
        [stick2])
    compartments = [watsonstick1, watsonstick2, ball]
    model = modeling_framework.MultiCompartmentAMICOModel(compartments)

    params = {}
    for parameter, card, in model.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             model.parameter_scales[parameter])

    for key, value in params.items():
        model.set_fixed_parameter(key, value)

    data = np.atleast_2d(model(scheme, **params))
    return model, data


def test_forward_model_matrix():
    """Provides tests for creation of forward model matrices."""

    # Define downsampled acquisition scheme
    nM = 60
    scheme_ds = \
        acquisition_scheme_from_bvalues(scheme.bvalues[0:nM],
                                        scheme.gradient_directions[0:nM, :],
                                        scheme.delta[0:nM],
                                        scheme.Delta[0:nM],
                                        scheme.min_b_shell_distance,
                                        scheme.b0_threshold)
    # ________________________________________________________________________
    # 1. Ball and stick model
    print("Testing forward model matrix for ball and stick model with "
          "{} and {} unknown...\n\n".format('G1Ball_1_lambda_iso',
                                            'C1Stick_1_lambda_par'))
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    amico_mc = MultiCompartmentAMICOModel(models=[ball, stick])
    M = amico_mc.forward_model_matrix(scheme_ds, model_dirs=[[0, 0]])

    mc = MultiCompartmentModel(models=[ball, stick])
    p_range = mc.parameter_ranges['G1Ball_1_lambda_iso']
    p_scale = mc.parameter_scales['G1Ball_1_lambda_iso']
    G1Ball_1_lambda_iso_grid = \
        np.linspace(p_range[0], p_range[1], amico_mc.Nt, endpoint=True) * \
        p_scale
    G1Ball_1_lambda_iso_mean = np.mean(p_range) * p_scale

    p_range = mc.parameter_ranges['C1Stick_1_lambda_par']
    p_scale = mc.parameter_scales['C1Stick_1_lambda_par']
    C1Stick_1_lambda_par_grid = \
        np.linspace(p_range[0], p_range[1], amico_mc.Nt, endpoint=True) * \
        p_scale
    C1Stick_1_lambda_par_mean = np.mean(p_range) * p_scale

    M_test = np.zeros_like(M)
    arguments = {}
    arguments['G1Ball_1_lambda_iso'] = G1Ball_1_lambda_iso_grid
    arguments['C1Stick_1_lambda_par'] = C1Stick_1_lambda_par_mean
    arguments['partial_volume_0'] = 1
    arguments['partial_volume_1'] = 0
    arguments['C1Stick_1_mu'] = [0, 0]
    M_test[:, 0:amico_mc.Nt] = mc.simulate_signal(scheme_ds, arguments).T

    arguments['G1Ball_1_lambda_iso'] = G1Ball_1_lambda_iso_mean
    arguments['C1Stick_1_lambda_par'] = C1Stick_1_lambda_par_grid
    arguments['partial_volume_0'] = 0
    arguments['partial_volume_1'] = 1
    arguments['C1Stick_1_mu'] = [0, 0]
    M_test[:, amico_mc.Nt:] = mc.simulate_signal(scheme_ds, arguments).T

    assert_almost_equal(M_test, M)

    print("Testing forward model matrix for ball and stick model with "
          "{} unknown...\n\n".format('C1Stick_1_lambda_par'))
    amico_mc = MultiCompartmentAMICOModel(models=[ball, stick])
    amico_mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3.e-9)
    M = amico_mc.forward_model_matrix(scheme_ds, model_dirs=[[0, 0]])

    mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3.e-9)
    M_test = np.zeros_like(M)

    arguments = {}
    arguments['C1Stick_1_lambda_par'] = C1Stick_1_lambda_par_mean
    arguments['partial_volume_0'] = 1.
    arguments['partial_volume_1'] = 0
    arguments['C1Stick_1_mu'] = [0, 0]
    M_test[:, 0] = mc.simulate_signal(scheme_ds, arguments).T

    arguments = {}
    arguments['C1Stick_1_lambda_par'] = C1Stick_1_lambda_par_grid
    arguments['partial_volume_0'] = 0
    arguments['partial_volume_1'] = 1.
    arguments['C1Stick_1_mu'] = [0, 0]
    M_test[:, 1:] = mc.simulate_signal(scheme_ds, arguments).T

    assert_almost_equal(M_test, M)
    # ________________________________________________________________________
    # 3. NODDI Bingham model
    print("Testing forward model matrix for NODDI Bingham model with "
          "{}, {}, {} and {} unknown...\n\n".
          format('SD2BinghamDistributed_1_SD2Bingham_1_psi',
                 'SD2BinghamDistributed_1_SD2Bingham_1_odi',
                 'SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction',
                 'SD2BinghamDistributed_1_partial_volume_0'))
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    bingham_dispersed_bundle = SD2BinghamDistributed(models=[stick, zeppelin])
    bingham_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                                    'C1Stick_1_lambda_par',
                                                    'partial_volume_0')
    bingham_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par',
                                                 'C1Stick_1_lambda_par')
    bingham_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par',
                                                 1.7e-9)
    amico_mc = MultiCompartmentAMICOModel(models=[ball,
                                                  bingham_dispersed_bundle])
    amico_mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)
    M = amico_mc.forward_model_matrix(scheme_ds, model_dirs=[[0, 0]])

    mc = MultiCompartmentModel(models=[ball, bingham_dispersed_bundle])
    mc.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

    p_range = mc.parameter_ranges['SD2BinghamDistributed_1_SD2Bingham_1_psi']
    p_scale = mc.parameter_scales['SD2BinghamDistributed_1_SD2Bingham_1_psi']
    SD2BinghamDistributed_1_SD2Bingham_1_psi_grid = \
        np.linspace(p_range[0], p_range[1], amico_mc.Nt, endpoint=True) * \
        p_scale
    SD2BinghamDistributed_1_SD2Bingham_1_psi_mean = np.mean(p_range) * p_scale

    p_range = mc.parameter_ranges['SD2BinghamDistributed_1_SD2Bingham_1_odi']
    p_scale = mc.parameter_scales['SD2BinghamDistributed_1_SD2Bingham_1_odi']
    SD2BinghamDistributed_1_SD2Bingham_1_odi_grid = \
        np.linspace(p_range[0], p_range[1], amico_mc.Nt, endpoint=True) * \
        p_scale
    SD2BinghamDistributed_1_SD2Bingham_1_odi_mean = np.mean(p_range) * p_scale

    p_range = mc.parameter_ranges[
        'SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction']
    p_scale = mc.parameter_scales[
        'SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction']
    SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction_grid = \
        np.linspace(p_range[0], p_range[1], amico_mc.Nt, endpoint=True) * \
        p_scale
    SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction_mean = \
        np.mean(p_range) * p_scale

    p_range = mc.parameter_ranges['SD2BinghamDistributed_1_partial_volume_0']
    p_scale = mc.parameter_scales['SD2BinghamDistributed_1_partial_volume_0']
    SD2BinghamDistributed_1_partial_volume_0_grid = \
        np.linspace(p_range[0], p_range[1], amico_mc.Nt, endpoint=True) * \
        p_scale
    SD2BinghamDistributed_1_partial_volume_0_mean = np.mean(p_range) * p_scale

    M_test = np.zeros_like(M)
    arguments = {}
    arguments['SD2BinghamDistributed_1_SD2Bingham_1_psi'] = \
        SD2BinghamDistributed_1_SD2Bingham_1_psi_mean
    arguments['SD2BinghamDistributed_1_SD2Bingham_1_odi'] = \
        SD2BinghamDistributed_1_SD2Bingham_1_odi_mean
    arguments['SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction'] = \
        SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction_mean
    arguments['SD2BinghamDistributed_1_partial_volume_0'] = \
        SD2BinghamDistributed_1_partial_volume_0_mean
    arguments['partial_volume_0'] = 1.
    arguments['partial_volume_1'] = 0
    arguments['SD2BinghamDistributed_1_SD2Bingham_1_mu'] = [0, 0]
    M_test[:, 0] = mc.simulate_signal(scheme_ds, arguments).T

    params_mesh = np.meshgrid(*[
        SD2BinghamDistributed_1_SD2Bingham_1_psi_grid,
        SD2BinghamDistributed_1_SD2Bingham_1_odi_grid,
        SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction_grid,
        SD2BinghamDistributed_1_partial_volume_0_grid])

    arguments['partial_volume_0'] = 0.
    arguments['partial_volume_1'] = 1.

    arguments['SD2BinghamDistributed_1_SD2Bingham_1_psi'] = \
        np.ravel(params_mesh[0])
    arguments['SD2BinghamDistributed_1_SD2Bingham_1_odi'] = \
        np.ravel(params_mesh[1])
    arguments['SD2BinghamDistributed_1_SD2Bingham_1_beta_fraction'] = \
        np.ravel(params_mesh[2])
    arguments['SD2BinghamDistributed_1_partial_volume_0'] = \
        np.ravel(params_mesh[3])

    M_test[:, 1:] = mc.simulate_signal(scheme_ds, arguments).T
    assert_almost_equal(M_test, M)
    # ________________________________________________________________________


def test_nnls():
    # TODO: test the non-regularized version of the inverse problem
    pass


def test_l2_regularization():
    # TODO: test the effect of L2 regularization
    pass


def test_l1_regularization():
    # TODO: test the effect of L1 regularization
    pass


def test_l1_and_l2_regularization():
    # TODO: test the effect of using both L1 and L2 regularization
    pass


def test_fitted_amico_model_properties():
    # TODO: check that the FittedMultiCompartmentAMICOModel class has the
    #  expected properties
    pass


def test_signal_fitting():
    # TODO: test the accuracy of the fitting on a synthetic signal
    pass
