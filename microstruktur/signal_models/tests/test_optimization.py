from os.path import join
from microstruktur.signal_models import (
    three_dimensional_models)
from numpy.testing import assert_array_almost_equal
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
    gt_lambda_par = np.random.rand() + 1.
    stick = three_dimensional_models.I1Stick(
        mu=gt_mu, lambda_par=gt_lambda_par)

    E = stick(bvals, gradient_directions)

    x0 = stick.parameters_to_parameter_vector(
        lambda_par=np.random.rand(),
        mu=np.random.rand(2)
    )
    res = stick.fit(E, bvals, gradient_directions, x0)[0]
    assert_array_almost_equal(np.r_[gt_lambda_par, gt_mu], res, 4)


def test_simple_ball_and_stick_optimization():
    stick = three_dimensional_models.I1Stick()
    ball = three_dimensional_models.E3Ball()

    ball_and_stick = three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
        models=[ball, stick],
        parameter_links=[],
        optimise_partial_volumes=True
    )
    gt_mu = np.clip(np.random.rand(2), .3, np.inf)
    gt_lambda_par = np.random.rand() + 1.
    gt_lambda_iso = gt_lambda_par / 2.
    gt_partial_volume = 0.3

    gt_parameter_vector = ball_and_stick.parameters_to_parameter_vector(
        I1Stick_1_lambda_par=gt_lambda_par,
        E3Ball_1_lambda_iso=gt_lambda_iso,
        I1Stick_1_mu=gt_mu,
        partial_volume_0=gt_partial_volume
    )

    E = ball_and_stick(bvals, gradient_directions,
                       **ball_and_stick.parameter_vector_to_parameters(gt_parameter_vector))

    x0 = ball_and_stick.parameters_to_parameter_vector(
        I1Stick_1_lambda_par=np.random.rand() + 1.,
        E3Ball_1_lambda_iso=gt_lambda_par / 2.,
        I1Stick_1_mu=np.random.rand(2),
        partial_volume_0=np.random.rand()
    )
    res = ball_and_stick.fit(E, bvals, gradient_directions, x0)[0]
    assert_array_almost_equal(gt_parameter_vector, res, 3)
