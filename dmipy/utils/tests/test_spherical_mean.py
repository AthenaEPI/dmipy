import numpy as np
from numpy.testing import assert_array_almost_equal
from dmipy.utils.spherical_mean import (
    estimate_spherical_mean_multi_shell
)
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy.signal_models import gaussian_models

scheme = wu_minn_hcp_acquisition_scheme()


def test_estimate_spherical_mean_multi_shell(
    lambda_par=1.7e-9, lambda_perp=0.8e-9,
    mu=np.r_[0, 0]
):
    zeppelin = gaussian_models.G2Zeppelin()
    zeppelin_smt = zeppelin.spherical_mean(
        scheme,
        lambda_par=lambda_par,
        lambda_perp=lambda_perp,
        mu=mu)
    zeppelin_multishell = zeppelin(
        scheme,
        lambda_par=lambda_par,
        lambda_perp=lambda_perp,
        mu=mu)

    smt_multi_shell = estimate_spherical_mean_multi_shell(
        zeppelin_multishell, scheme
    )
    assert_array_almost_equal(smt_multi_shell, zeppelin_smt)
