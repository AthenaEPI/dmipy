from dmipy.distributions import distribute_models
from dmipy.signal_models import sphere_models, cylinder_models
from numpy.testing import assert_raises


def test_raise_mixed_parameter_types():
    sphere = sphere_models.S2SphereStejskalTannerApproximation()
    cylinder = cylinder_models.C2CylinderStejskalTannerApproximation()
    assert_raises(AttributeError,
                  distribute_models.DD1GammaDistributed,
                  [sphere, cylinder])


def test_set_fixed_parameter_raises():
    cyl = cylinder_models.C1Stick()
    distcyl = distribute_models.SD1WatsonDistributed([cyl])
    assert_raises(ValueError, distcyl.set_fixed_parameter,
                  'SD1Watson_1_odi', [1])
    assert_raises(ValueError, distcyl.set_fixed_parameter,
                  'SD1Watson_1_mu', [1])
    assert_raises(ValueError, distcyl.set_fixed_parameter,
                  'blabla', [1])
