from dmipy.signal_models import (
    cylinder_models, gaussian_models, sphere_models, spherical_mean_models)
from dmipy.core import modeling_framework
from numpy.testing import (
    assert_equal, assert_array_almost_equal, assert_array_equal)
import numpy as np
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme

scheme = wu_minn_hcp_acquisition_scheme()


def test_simple_stick_optimization():
    stick = cylinder_models.C1Stick()
    gt_mu = np.random.rand(2)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9

    stick_model = modeling_framework.MultiCompartmentModel(
        models=[stick])

    gt_parameter_vector = stick_model.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=gt_lambda_par, C1Stick_1_mu=gt_mu)

    E = stick_model.simulate_signal(scheme, gt_parameter_vector)

    x0 = stick_model.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=(np.random.rand() + 1.) * 1e-9,
        C1Stick_1_mu=np.random.rand(2)
    )
    res = stick_model.fit(scheme, E, x0).fitted_parameters_vector
    assert_array_almost_equal(gt_parameter_vector, res.squeeze(), 2)


def test_simple_ball_and_stick_optimization():
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()

    ball_and_stick = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, stick],
            parameter_links=[],
            optimise_partial_volumes=True)
    )
    gt_mu = np.clip(np.random.rand(2), .3, np.inf)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3

    gt_parameter_vector = ball_and_stick.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=gt_lambda_par,
        G1Ball_1_lambda_iso=gt_lambda_iso,
        C1Stick_1_mu=gt_mu,
        partial_volume_0=gt_partial_volume,
        partial_volume_1=1 - gt_partial_volume
    )

    E = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    vf_rand = np.random.rand()
    x0 = ball_and_stick.parameters_to_parameter_vector(
        C1Stick_1_lambda_par=(np.random.rand() + 1.) * 1e-9,
        G1Ball_1_lambda_iso=gt_lambda_par / 2.,
        C1Stick_1_mu=np.random.rand(2),
        partial_volume_0=vf_rand,
        partial_volume_1=1 - vf_rand
    )
    res = ball_and_stick.fit(scheme, E, x0).fitted_parameters_vector
    assert_array_almost_equal(gt_parameter_vector, res.squeeze(), 2)


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

    gt_parameter_vector = (
        ball_and_stick.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=gt_lambda_par,
            G1Ball_1_lambda_iso=gt_lambda_iso,
            C1Stick_1_mu=gt_mu_array,
            partial_volume_0=gt_partial_volume,
            partial_volume_1=1 - gt_partial_volume)
    )

    E_array = ball_and_stick.simulate_signal(
        scheme, gt_parameter_vector)

    # I'm giving a voxel-dependent initial condition with gt_mu_array
    res = ball_and_stick.fit(scheme,
                             E_array,
                             gt_parameter_vector).fitted_parameters_vector
    # optimization should stop immediately as I'm giving the ground truth.
    assert_equal(np.all(np.ravel(res - gt_parameter_vector) == 0.), True)
    # and the parameter vector dictionaries of the results and x0 should also
    # be the same.
    res_parameters = ball_and_stick.parameter_vector_to_parameters(res)
    x0_parameters = ball_and_stick.parameter_vector_to_parameters(
        gt_parameter_vector)
    for key in res_parameters.keys():
        assert_array_equal(x0_parameters[key], res_parameters[key])


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

    gt_parameter_vector = (
        stick_and_zeppelin.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=gt_lambda_par,
            C1Stick_1_mu=gt_mu,
            partial_volume_0=gt_partial_volume,
            partial_volume_1=1 - gt_partial_volume)
    )

    E = stick_and_zeppelin.simulate_signal(
        scheme, gt_parameter_vector)

    # now we make the stick and zeppelin spherical mean model and check if the
    # same lambda_par and volume fraction result as the 3D generated data.
    stick_sm = spherical_mean_models.C1StickSphericalMean()
    zeppelin_sm = spherical_mean_models.G2ZeppelinSphericalMean()

    stick_and_tortuous_zeppelin_sm = (
        modeling_framework.MultiCompartmentModel(
            models=[stick_sm, zeppelin_sm])
    )

    stick_and_tortuous_zeppelin_sm.set_tortuous_parameter(
        'G2ZeppelinSphericalMean_1_lambda_perp',
        'C1StickSphericalMean_1_lambda_par',
        'partial_volume_0',
        'partial_volume_1')
    stick_and_tortuous_zeppelin_sm.set_equal_parameter(
        'G2ZeppelinSphericalMean_1_lambda_par',
        'C1StickSphericalMean_1_lambda_par')

    res_sm = stick_and_tortuous_zeppelin_sm.fit(scheme, E
                                                ).fitted_parameters_vector

    assert_array_almost_equal(
        np.r_[gt_lambda_par, gt_partial_volume], res_sm.squeeze()[:-1], 2)


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


def test_MIX_fitting():
    ball = gaussian_models.G1Ball()
    zeppelin = gaussian_models.G2Zeppelin()
    ball_and_zeppelin = (
        modeling_framework.MultiCompartmentModel(
            models=[ball, zeppelin]))

    parameter_vector = ball_and_zeppelin.parameters_to_parameter_vector(
        G1Ball_1_lambda_iso=2.7e-9,
        partial_volume_0=.2,
        partial_volume_1=.8,
        G2Zeppelin_1_lambda_perp=.5e-9,
        G2Zeppelin_1_mu=(np.pi / 2., np.pi / 2.),
        G2Zeppelin_1_lambda_par=1.7e-9
    )

    E = ball_and_zeppelin.simulate_signal(
        scheme, parameter_vector)
    fit = ball_and_zeppelin.fit(
        scheme,
        E, solver='mix').fitted_parameters_vector
    assert_array_almost_equal(abs(fit).squeeze(), parameter_vector, 2)
