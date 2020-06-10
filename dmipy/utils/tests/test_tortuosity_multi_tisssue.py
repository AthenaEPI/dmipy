import numpy as np

from dmipy.utils.utils import T1_tortuosity


def test_single_tissue():
    lpar = 1.7e-9
    icvf = 0.7  # intra cellular volume fraction

    # only intra and extra cellular compartments
    lperp = (1. - icvf) * lpar
    np.testing.assert_almost_equal(T1_tortuosity(lpar, icvf), lperp)

    # with additional compartment
    ecvf = 0.2
    lperp = ecvf / (icvf + ecvf) * lpar
    np.testing.assert_almost_equal(T1_tortuosity(lpar, icvf, ecvf), lperp)


def test_multi_tissue():
    lpar = 1.7e-9
    icsf = 0.7  # intra cellular signal fraction
    s0ic = 3000.
    s0ec = 4000.

    # only intra and extra cellular compartments
    ecsf = 1. - icsf
    ecvf = icsf * s0ec / (icsf * s0ec + ecsf * s0ec)
    lperp = ecvf * lpar
    actual = T1_tortuosity(lpar, icsf, ecsf, s0ic, s0ec)
    np.testing.assert_almost_equal(actual, lperp)

    # with additional compartment
    ecsf = 0.2
    ecvf = icsf * s0ec / (icsf * s0ec + ecsf * s0ec)
    lperp = ecvf * lpar
    actual = T1_tortuosity(lpar, icsf, ecsf, s0ic, s0ec)
    np.testing.assert_almost_equal(actual, lperp)
