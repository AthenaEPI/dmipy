import numpy as np
from os.path import join
import pkg_resources
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)

_GRADIENT_TABLES_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data/gradient_tables'
)


def wu_minn_hcp_acquisition_scheme():
    _bvals = np.loadtxt(
        join(_GRADIENT_TABLES_PATH,
             'bvals_hcp_wu_minn.txt')
    ) * 1e6
    _gradient_directions = np.loadtxt(
        join(_GRADIENT_TABLES_PATH,
             'bvecs_hcp_wu_minn.txt')
    )
    _delta = 0.0106
    _Delta = 0.0431
    return acquisition_scheme_from_bvalues(
        _bvals, _gradient_directions, _delta, _Delta)
