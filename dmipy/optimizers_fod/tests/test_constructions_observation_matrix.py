from dmipy.signal_models import cylinder_models
from dmipy.distributions import distribute_models, distributions
import numpy as np
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy.optimizers_fod.construct_observation_matrix import (
    construct_model_based_A_matrix)


scheme = wu_minn_hcp_acquisition_scheme()


def test_construction_observation_matrix(
        odi=0.15, mu=[0., 0.], lambda_par=1.7e-9, lmax=8):
    stick = cylinder_models.C1Stick(lambda_par=lambda_par)
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])

    params = {'SD1Watson_1_odi': odi,
              'SD1Watson_1_mu': mu,
              'C1Stick_1_lambda_par': lambda_par}

    data = watsonstick(scheme, **params)
    watson = distributions.SD1Watson(mu=mu, odi=odi)
    sh_watson = watson.spherical_harmonics_representation(lmax)

    stick_rh = stick.rotational_harmonics_representation(scheme)
    A = construct_model_based_A_matrix(scheme, stick_rh, lmax)

    data_approximated = A.dot(sh_watson)
    np.testing.assert_array_almost_equal(data_approximated, data, 4)
