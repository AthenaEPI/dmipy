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
