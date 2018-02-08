from dmipy.core import modeling_framework
from dmipy.signal_models import plane_models, gaussian_models
from dmipy.distributions import distribute_models
from numpy.testing import assert_raises


def test_raise_combination_NRM_and_others():
    ball = gaussian_models.G1Ball()
    plane = plane_models.P3PlaneCallaghanApproximation()

    assert_raises(
        ValueError, modeling_framework.MultiCompartmentModel,
        [ball, plane])


def test_raise_spherical_distribution_in_spherical_mean():
    zeppelin = gaussian_models.G2Zeppelin()
    watson = distribute_models.SD1WatsonDistributed([zeppelin])
    assert_raises(
        ValueError,
        modeling_framework.MultiCompartmentSphericalMeanModel,
        [watson])


def test_raise_NRMmodel_in_spherical_mean():
    plane = plane_models.P3PlaneCallaghanApproximation()
    assert_raises(
        ValueError,
        modeling_framework.MultiCompartmentSphericalMeanModel,
        [plane])
