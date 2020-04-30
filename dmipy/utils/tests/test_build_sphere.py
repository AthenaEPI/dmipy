from nose.tools import assert_raises, assert_equal, assert_true
from dmipy.utils.build_sphere import get_hemisphere
import numpy as np
from numpy.testing import assert_almost_equal

def test_get_hemisphere():
    assert_raises(TypeError, get_hemisphere, 'string')

    array1d = np.random.rand(3)
    assert_raises(ValueError, get_hemisphere, array1d)

    array_with_4_cols = np.random.rand(10, 4)
    assert_raises(ValueError, get_hemisphere, array_with_4_cols)

    # setup dummy transposed set of directions and process it
    proper_array_transposed = np.random.rand(3, 10)
    proper_array_transposed /= np.linalg.norm(proper_array_transposed, axis=0)
    processed = get_hemisphere(proper_array_transposed)

    # check dimensions
    assert_equal(processed.shape[1], 3)
    # check normalization
    assert_almost_equal(np.linalg.norm(processed, axis=1), 1)

    # what if the input is a list?
    dirlist = [d for d in processed]
    directions = get_hemisphere(dirlist)
    assert_true(isinstance(directions, np.ndarray))
    assert_equal(directions.shape[0], len(dirlist))
    assert_equal(directions.shape[1], 3)

    # what if the input is an int?
    n = 64
    directions = get_hemisphere(n)
    assert_true(isinstance(directions, np.ndarray))
    assert_equal(directions.shape[0], n)
    assert_equal(directions.shape[1], 3)

