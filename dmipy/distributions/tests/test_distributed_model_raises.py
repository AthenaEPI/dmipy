from dmipy.distributions import distribute_models
from dmipy.signal_models import sphere_models, cylinder_models
from numpy.testing import assert_raises


def test_raise_mixed_parameter_types():
    sphere = sphere_models.S2SphereSodermanApproximation()
    cylinder = cylinder_models.C2CylinderSodermanApproximation()
    assert_raises(AttributeError,
                  distribute_models.DD1GammaDistributed,
                  [sphere, cylinder])
