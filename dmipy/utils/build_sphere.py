import numpy as np
from dipy.core.sphere import disperse_charges, HemiSphere


def get_hemisphere(directions=1000):
    """
    Get a set of directions as a numpy array with 3 columns.

    It relies on the disperse_charges function of dipy.

    Args:
        directions: int or N-by-3 array
            If integer, this corresponds to the number of points on the
            hemisphere that will be used as directions for the look-up-table.
            If array, it corresponds to the list of directions that will be
            used for building the look up table.
    """
    if isinstance(directions, int) or isinstance(directions, float):
        # this is a copypaste of the code from the dipy example
        n_pts = int(directions)
        theta = np.pi * np.random.rand(n_pts)
        phi = 2 * np.pi * np.random.rand(n_pts)
        hsph_initial = HemiSphere(theta=theta, phi=phi)
        hsph_updated, _ = disperse_charges(hsph_initial, 5000)
        return hsph_updated.vertices
    elif isinstance(directions, np.ndarray) or isinstance(directions, list):
        directions = np.squeeze(np.asarray(directions))
        if directions.ndim != 2:
            raise ValueError('Directions must be passed as a 2d array.')
        if 3 not in directions.shape:
            raise ValueError('One of the directions must be 3.')
        if directions.shape[0] == 3:
            directions = directions.T
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        return directions
    else:
        raise TypeError('Input argument must be an integer or a list/array.')