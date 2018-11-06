from dmipy.signal_models import (
    cylinder_models, gaussian_models, sphere_models)
from dmipy.core import modeling_framework
from numpy.testing import assert_array_almost_equal
import numpy as np
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()


def test_simple_stick_optimization():
    stick = cylinder_models.C1Stick()
    gt_mu = np.random.rand(2)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9

    stick_model = modeling_framework.MultiCompartmentModel(
        models=[stick])

    gt_parameters = {'C1Stick_1_lambda_par': gt_lambda_par,
                     'C1Stick_1_mu': gt_mu}

    gt_parameter_vector = stick_model.parameters_to_parameter_vector(
        **gt_parameters)

    E = stick_model.simulate_signal(scheme, gt_parameter_vector)

    stick_model.set_initial_guess_parameter('C1Stick_1_lambda_par',
                                            (np.random.rand() + 1.) * 1e-9)
    stick_model.set_initial_guess_parameter('C1Stick_1_mu', np.random.rand(2))
    fit = stick_model.fit(scheme, E)
    for parname, gt_value in gt_parameters.items():
        fitval = fit.fitted_parameters[parname][0]
        scale = stick_model.parameter_scales[parname]
        assert_array_almost_equal(
            abs(fitval / scale), gt_value / scale, 2)


def test_simple_ball_and_stick_optimization():
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()

    ball_and_stick = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, stick])
    )
    gt_mu = np.clip(np.random.rand(2), .3, np.inf)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3

    gt_parameters = {'C1Stick_1_lambda_par': gt_lambda_par,
                     'G1Ball_1_lambda_iso': gt_lambda_iso,
                     'C1Stick_1_mu': gt_mu,
                     'partial_volume_0': gt_partial_volume,
                     'partial_volume_1': 1 - gt_partial_volume}

    gt_parameter_vector = ball_and_stick.parameters_to_parameter_vector(
        **gt_parameters)

    E = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    vf_rand = np.random.rand()
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_lambda_par', (np.random.rand() + 1.) * 1e-9)
    ball_and_stick.set_initial_guess_parameter(
        'G1Ball_1_lambda_iso', gt_lambda_par / 2.)
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_mu', np.random.rand(2))
    ball_and_stick.set_initial_guess_parameter('partial_volume_0', vf_rand)
    ball_and_stick.set_initial_guess_parameter('partial_volume_1', 1 - vf_rand)

    fit = ball_and_stick.fit(scheme, E)
    for parname, gt_value in gt_parameters.items():
        fitval = fit.fitted_parameters[parname][0]
        scale = ball_and_stick.parameter_scales[parname]
        assert_array_almost_equal(
            abs(fitval / scale), gt_value / scale, 2)


def test_multi_dimensional_x0():
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    ball_and_stick = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, stick],)
    )
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3
    gt_mu_array = np.empty((10, 10, 2))

    # I'm putting the orientation of the stick all over the sphere.
    for i, mu1 in enumerate(np.linspace(0, np.pi, 10)):
        for j, mu2 in enumerate(np.linspace(-np.pi, np.pi, 10)):
            gt_mu_array[i, j] = np.r_[mu1, mu2]

    gt_parameters = {'C1Stick_1_lambda_par': gt_lambda_par,
                     'G1Ball_1_lambda_iso': gt_lambda_iso,
                     'C1Stick_1_mu': gt_mu_array,
                     'partial_volume_0': gt_partial_volume,
                     'partial_volume_1': 1 - gt_partial_volume}

    gt_parameter_vector = (
        ball_and_stick.parameters_to_parameter_vector(
            **gt_parameters))

    E_array = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_lambda_par', gt_lambda_par)
    ball_and_stick.set_initial_guess_parameter(
        'G1Ball_1_lambda_iso', gt_lambda_iso)
    ball_and_stick.set_initial_guess_parameter(
        'C1Stick_1_mu', gt_mu_array)
    ball_and_stick.set_initial_guess_parameter(
        'partial_volume_0', gt_partial_volume)
    ball_and_stick.set_initial_guess_parameter(
        'partial_volume_1', 1 - gt_partial_volume)
    # I'm giving a voxel-dependent initial condition with gt_mu_array
    fit = ball_and_stick.fit(scheme, E_array)
    # and the parameter vector dictionaries of the results and x0 should also
    # be the same.
    for parname, gt_value in gt_parameters.items():
        fitval = fit.fitted_parameters[parname]
        assert_array_almost_equal(gt_parameters[parname], fitval)


def test_stick_and_tortuous_zeppelin_to_spherical_mean_fit():
    """ this is a more complex test to see if we can generate 3D data using a
    stick and zeppelin model, where we assume the perpendicular diffusivity is
    linked to the parallel diffusivity and volume fraction using tortuosity. We
    then use the spherical mean models of stick and zeppelin with the same
    tortuosity assumption to fit the 3D data (and estimating the spherical mean
    of each shell). The final check is whether the parallel diffusivity and
    volume fraction between the 3D and spherical mean models correspond."""

    gt_mu = np.clip(np.random.rand(2), .3, np.inf)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    gt_partial_volume = 0.3

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
        'C1Stick_1_lambda_par',
        'G2Zeppelin_1_lambda_par'
    )

    gt_parameters = {'C1Stick_1_lambda_par': gt_lambda_par,
                     'C1Stick_1_mu': gt_mu,
                     'partial_volume_0': gt_partial_volume,
                     'partial_volume_1': 1 - gt_partial_volume}

    gt_parameter_vector = (
        stick_and_zeppelin.parameters_to_parameter_vector(
            **gt_parameters))

    E = stick_and_zeppelin.simulate_signal(
        scheme, gt_parameter_vector)

    # now we make the stick and zeppelin spherical mean model and check if the
    # same lambda_par and volume fraction result as the 3D generated data.
    stick_and_tortuous_zeppelin_sm = (
        modeling_framework.MultiCompartmentSphericalMeanModel(
            models=[stick, zeppelin])
    )

    stick_and_tortuous_zeppelin_sm.set_tortuous_parameter(
        'G2Zeppelin_1_lambda_perp',
        'C1Stick_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1')
    stick_and_tortuous_zeppelin_sm.set_equal_parameter(
        'G2Zeppelin_1_lambda_par',
        'C1Stick_1_lambda_par')

    fit = stick_and_tortuous_zeppelin_sm.fit(scheme, E)
    for parname, gt_value in gt_parameters.items():
        if parname not in fit.fitted_parameters.keys():
            continue
        fitval = fit.fitted_parameters[parname][0]
        scale = stick_and_tortuous_zeppelin_sm.parameter_scales[parname]
        assert_array_almost_equal(
            abs(fitval / scale), gt_value / scale, 2)


def test_fractions_add_up_to_one():
    dot1 = sphere_models.S1Dot()
    dot2 = sphere_models.S1Dot()
    dot3 = sphere_models.S1Dot()
    dot4 = sphere_models.S1Dot()
    dot5 = sphere_models.S1Dot()
    dots = modeling_framework.MultiCompartmentModel(
        models=[dot1, dot2, dot3, dot4, dot5])
    random_fractions = np.random.rand(5)
    random_fractions /= random_fractions.sum()
    parameter_vector = dots.parameters_to_parameter_vector(
        partial_volume_0=random_fractions[0],
        partial_volume_1=random_fractions[1],
        partial_volume_2=random_fractions[2],
        partial_volume_3=random_fractions[3],
        partial_volume_4=random_fractions[4])
    E = dots.simulate_signal(scheme, parameter_vector)
    assert_array_almost_equal(E, np.ones(len(E)))


def test_MIX_fitting_multimodel():
    ball = gaussian_models.G1Ball()
    zeppelin = gaussian_models.G2Zeppelin()
    ball_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, zeppelin]))

    gt_parameters = {'G1Ball_1_lambda_iso': 2.7e-9,
                     'partial_volume_0': .2,
                     'partial_volume_1': .8,
                     'G2Zeppelin_1_lambda_perp': .5e-9,
                     'G2Zeppelin_1_mu': (np.pi / 2., np.pi / 2.),
                     'G2Zeppelin_1_lambda_par': 1.7e-9}

    parameter_vector = ball_and_zeppelin.parameters_to_parameter_vector(
        **gt_parameters)

    E = ball_and_zeppelin.simulate_signal(
        scheme, parameter_vector)
    fit = ball_and_zeppelin.fit(
        scheme,
        E, solver='mix')
    for parname, gt_value in gt_parameters.items():
        fitval = fit.fitted_parameters[parname][0]
        scale = ball_and_zeppelin.parameter_scales[parname]
        assert_array_almost_equal(
            abs(fitval / scale), gt_value / scale, 2)


def test_MIX_fitting_singlemodel():
    stick = cylinder_models.C1Stick()
    stick_mod = (
        modeling_framework.MultiCompartmentModel(
            models=[stick]))

    gt_parameters = {'C1Stick_1_mu': [np.pi / 2., np.pi / 2.],
                     'C1Stick_1_lambda_par': 1.7e-9}

    parameter_vector = stick_mod.parameters_to_parameter_vector(
        **gt_parameters)

    E = stick_mod.simulate_signal(
        scheme, parameter_vector)
    fit = stick_mod.fit(
        scheme,
        E, solver='mix')
    for parname, gt_value in gt_parameters.items():
        fitval = fit.fitted_parameters[parname][0]
        scale = stick_mod.parameter_scales[parname]
        assert_array_almost_equal(
            abs(fitval / scale), gt_value / scale, 2)
