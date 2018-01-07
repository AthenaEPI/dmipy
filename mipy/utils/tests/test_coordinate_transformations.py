from numpy.testing import assert_almost_equal
import numpy as np
from mipy.utils import utils


def test_coordinate_transformation():
    # test for a vector of size 3
    cartesian_vector = np.random.rand(3)
    spherical_vector = utils.cart2sphere(cartesian_vector)
    recovered_cartesian_vector = utils.sphere2cart(spherical_vector)
    assert_almost_equal(cartesian_vector, recovered_cartesian_vector)

    # test for a vector of size N x 3
    cartesian_vector = np.random.rand(10, 3)
    spherical_vector = utils.cart2sphere(cartesian_vector)
    recovered_cartesian_vector = utils.sphere2cart(spherical_vector)
    assert_almost_equal(cartesian_vector, recovered_cartesian_vector)
