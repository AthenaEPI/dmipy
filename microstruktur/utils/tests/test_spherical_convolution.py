from numpy.testing import assert_almost_equal, assert_equal
from microstruktur.signal_models import cylinder_models, distributions
from microstruktur.utils.spherical_convolution import (kernel_sh_to_rh,
                                                       sh_convolution)
from microstruktur.utils import utils
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.data import get_sphere
import numpy as np
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)

delta = 0.01
Delta = 0.03


def test_spherical_convolution_watson_sh(sh_order=4):
    sphere = get_sphere('symmetric724')

    n = sphere.vertices
    bval = np.tile(1e9, len(n))
    scheme = acquisition_scheme_from_bvalues(bval, n, delta, Delta)
    indices_sphere_orientations = np.arange(sphere.vertices.shape[0])
    np.random.shuffle(indices_sphere_orientations)
    mu_index = indices_sphere_orientations[0]
    mu_watson = sphere.vertices[mu_index]
    mu_watson_sphere = utils.cart2sphere(mu_watson)[1:]

    watson = distributions.SD1Watson(mu=mu_watson_sphere, kappa=10.)
    f_sf = watson(n=sphere.vertices)
    f_sh = sf_to_sh(f_sf, sphere, sh_order)

    lambda_par = 2e-9
    stick = cylinder_models.C1Stick(mu=[0, 0], lambda_par=lambda_par)
    k_sf = stick(scheme)
    k_sh = sf_to_sh(k_sf, sphere, sh_order)
    k_rh = kernel_sh_to_rh(k_sh)

    fk_convolved_sh = sh_convolution(f_sh, k_rh)
    fk_convolved_sf = sh_to_sf(fk_convolved_sh, sphere, sh_order)

    # assert if spherical mean is the same between kernel and convolved kernel
    assert_almost_equal(abs(np.mean(k_sf) - np.mean(fk_convolved_sf)), 0., 2)
    # assert if the lowest signal attenuation (E(b,n)) is orientation along
    # the orientation of the watson distribution.
    min_position = np.argmin(fk_convolved_sf)

    if min_position == mu_index:
        assert_equal(min_position, mu_index)
    else:  # then it's the opposite direction
        sphere_positions = np.arange(sphere.vertices.shape[0])
        opposite_index = np.all(
            np.round(sphere.vertices - mu_watson, 2) == 0, axis=1
        )
        min_position_opposite = sphere_positions[opposite_index]
        assert_equal(min_position_opposite, mu_index)
