from os.path import join
from microstruktur.signal_models import (
    three_dimensional_models)
from microstruktur.signal_models.utils import (
    T1_tortuosity, parameter_equality, define_shell_indices
)
from numpy.testing import (
    assert_equal, assert_array_almost_equal, assert_array_equal)
import numpy as np


bvals = np.loadtxt(
    join(three_dimensional_models.GRADIENT_TABLES_PATH,
         'bvals_hcp_wu_minn.txt')
)
bvals *= 1e6
gradient_directions = np.loadtxt(
    join(three_dimensional_models.GRADIENT_TABLES_PATH,
         'bvecs_hcp_wu_minn.txt')
)


def test_simple_stick_optimization():
    gt_mu = np.random.rand(2)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    stick = three_dimensional_models.I1Stick(
        mu=gt_mu, lambda_par=gt_lambda_par)

    E = stick(bvals, gradient_directions)

    x0 = stick.parameters_to_parameter_vector(
        lambda_par=(np.random.rand() + 1.) * 1e-9,
        mu=np.random.rand(2)
    )
    res = stick.fit(E, bvals, gradient_directions, x0)[0]
    assert_array_almost_equal(np.r_[gt_lambda_par, gt_mu], res, 4)


def test_simple_ball_and_stick_optimization():
    stick = three_dimensional_models.I1Stick()
    ball = three_dimensional_models.E3Ball()

    ball_and_stick = (
        three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
            models=[ball, stick],
            parameter_links=[],
            optimise_partial_volumes=True)
    )
    gt_mu = np.clip(np.random.rand(2), .3, np.inf)
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3

    gt_parameter_vector = ball_and_stick.parameters_to_parameter_vector(
        I1Stick_1_lambda_par=gt_lambda_par,
        E3Ball_1_lambda_iso=gt_lambda_iso,
        I1Stick_1_mu=gt_mu,
        partial_volume_0=gt_partial_volume
    )

    E = ball_and_stick(
        bvals, gradient_directions,
        **ball_and_stick.parameter_vector_to_parameters(gt_parameter_vector))

    x0 = ball_and_stick.parameters_to_parameter_vector(
        I1Stick_1_lambda_par=(np.random.rand() + 1.) * 1e-9,
        E3Ball_1_lambda_iso=gt_lambda_par / 2.,
        I1Stick_1_mu=np.random.rand(2),
        partial_volume_0=np.random.rand()
    )
    res = ball_and_stick.fit(E, bvals, gradient_directions, x0)[0]
    assert_array_almost_equal(gt_parameter_vector, res, 3)


def test_multi_dimensional_x0():
    stick = three_dimensional_models.I1Stick()
    ball = three_dimensional_models.E3Ball()

    ball_and_stick = (
        three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
            models=[ball, stick],
            parameter_links=[],
            optimise_partial_volumes=True)
    )
    gt_lambda_par = (np.random.rand() + 1.) * 1e-9
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3
    E_array = np.empty((10, 10, len(bvals)), dtype=float)
    gt_mu_array = np.empty((10, 10, 2))

    # I'm putting the orientation of the stick all over the sphere.
    for i, mu1 in enumerate(np.linspace(0, np.pi, 10)):
        for j, mu2 in enumerate(np.linspace(-np.pi, np.pi, 10)):
            gt_mu = np.r_[mu1, mu2]
            gt_mu_array[i, j] = gt_mu
            gt_parameter_vector = (
                ball_and_stick.parameters_to_parameter_vector(
                    I1Stick_1_lambda_par=gt_lambda_par,
                    E3Ball_1_lambda_iso=gt_lambda_iso,
                    I1Stick_1_mu=gt_mu,
                    partial_volume_0=gt_partial_volume)
            )
            E_array[i, j] = ball_and_stick(
                bvals, gradient_directions,
                **ball_and_stick.parameter_vector_to_parameters(
                    gt_parameter_vector)
            )

    # I'm giving a voxel-dependent initial condition with gt_mu_array
    x0_gt = ball_and_stick.parameters_to_parameter_vector(
        I1Stick_1_lambda_par=gt_lambda_par,
        E3Ball_1_lambda_iso=gt_lambda_iso,
        I1Stick_1_mu=gt_mu_array,
        partial_volume_0=gt_partial_volume
    )
    res = ball_and_stick.fit(E_array, bvals, gradient_directions, x0_gt)
    # optimization should stop immediately as I'm giving the ground truth.
    assert_equal(np.all(np.ravel(res - x0_gt) == 0.), True)
    # and the parameter vector dictionaries of the results and x0 should also
    # be the same.
    res_parameters = ball_and_stick.parameter_vector_to_parameters(res)
    x0_parameters = ball_and_stick.parameter_vector_to_parameters(x0_gt)
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

    stick = three_dimensional_models.I1Stick()
    zeppelin = three_dimensional_models.E4Zeppelin()

    parameter_links_stick_and_tortuous_zeppelin = [
        (  # tortuosity assumption
            zeppelin, 'lambda_perp',
            T1_tortuosity, [
                (None, 'partial_volume_0'),
                (stick, 'lambda_par')
            ]
        ),
        (  # equal parallel diffusivities
            zeppelin, 'lambda_par',
            parameter_equality, [
                (stick, 'lambda_par')
            ]
        ),
        (  # equal parallel diffusivities
            zeppelin, 'mu',
            parameter_equality, [
                (stick, 'mu')
            ]
        )
    ]

    stick_and_tortuous_zeppelin = (
        three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
            models=[stick, zeppelin],
            parameter_links=parameter_links_stick_and_tortuous_zeppelin,
            optimise_partial_volumes=True)
    )

    gt_parameter_vector = (
        stick_and_tortuous_zeppelin.parameters_to_parameter_vector(
            I1Stick_1_lambda_par=gt_lambda_par,
            I1Stick_1_mu=gt_mu,
            partial_volume_0=gt_partial_volume)
    )

    E = stick_and_tortuous_zeppelin(
        bvals, gradient_directions,
        **stick_and_tortuous_zeppelin.parameter_vector_to_parameters(
            gt_parameter_vector)
    )

    # now we make the stick and zeppelin spherical mean model and check if the
    # same lambda_par and volume fraction result as the 3D generated data.
    stick_sm = three_dimensional_models.I1StickSphericalMean()
    zeppelin_sm = three_dimensional_models.E4ZeppelinSphericalMean()

    parameter_links_stick_and_tortuous_zeppelin_smt = [
        (  # tortuosity assumption
            zeppelin_sm, 'lambda_perp',
            T1_tortuosity, [
                (None, 'partial_volume_0'),
                (stick_sm, 'lambda_par')
            ]
        ),
        (  # equal parallel diffusivities
            zeppelin_sm, 'lambda_par',
            parameter_equality, [
                (stick_sm, 'lambda_par')
            ]
        )
    ]

    stick_and_tortuous_zeppelin_sm = (
        three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
            models=[stick_sm, zeppelin_sm],
            parameter_links=parameter_links_stick_and_tortuous_zeppelin_smt,
            optimise_partial_volumes=True)
    )
    x0 = stick_and_tortuous_zeppelin_sm.parameters_to_parameter_vector(
        I1StickSphericalMean_1_lambda_par=.6 * 1e-9,
        partial_volume_0=0.55
    )

    shell_indices, _ = define_shell_indices(
        bvals, ((0, 10e6), (995e6, 1005e6), (1995e6, 2005e6), (2995e6, 3005e6))
    )

    res_sm = stick_and_tortuous_zeppelin_sm.fit(
        E, bvals, gradient_directions, x0, shell_indices=shell_indices)[0]

    assert_array_almost_equal(
        np.r_[gt_lambda_par, gt_partial_volume], res_sm, 2)
