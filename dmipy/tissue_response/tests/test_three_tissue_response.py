import numpy as np
from dmipy.tissue_response.three_tissue_response import (
    optimal_threshold, signal_decay_metric)
from dmipy.signal_models.gaussian_models import G1Ball
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
scheme = wu_minn_hcp_acquisition_scheme()


def test_optimal_threshold():
    data = np.linspace(0, 8, 101)
    opt = optimal_threshold(data)
    np.testing.assert_(np.round(opt) == 8 // 2)


def test_signal_decay_metric():
    data_iso1 = G1Ball(lambda_iso=2.5e-9)(scheme)
    data_iso2 = G1Ball(lambda_iso=1.5e-9)(scheme)
    data = np.array([data_iso1, data_iso2])
    sdm = signal_decay_metric(scheme, data)
    np.testing.assert_(sdm[0] > sdm[1])
