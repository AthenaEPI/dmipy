from dmipy.signal_models import (
    cylinder_models, gaussian_models, sphere_models)
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalMeanModel)
import numpy as np
from numpy.testing import assert_equal

scheme = wu_minn_hcp_acquisition_scheme()

models = [
    cylinder_models.C1Stick(),
    cylinder_models.C2CylinderSodermanApproximation(),
    cylinder_models.C3CylinderCallaghanApproximation(),
    cylinder_models.C4CylinderGaussianPhaseApproximation(),
    gaussian_models.G1Ball(),
    gaussian_models.G2Zeppelin(),
    gaussian_models.G3RestrictedZeppelin(),
    sphere_models.S2SphereSodermanApproximation()
]


def test_model_spherical_mean():
    for model in models:
        params = {}
        for param, card in model.parameter_cardinality.items():
            params[param] = (np.random.rand(card) *
                             model.parameter_scales[param])
        assert_equal(isinstance(
            model.spherical_mean(scheme, **params), np.ndarray),
            True)


def test_MultiCompartmentSphericalMeanModel():
    for model in models:
        mc_sm_model = MultiCompartmentSphericalMeanModel(
            [model])

        params = {}
        for param, card in mc_sm_model.parameter_cardinality.items():
            params[param] = (np.random.rand(card) *
                             mc_sm_model.parameter_scales[param])
        assert_equal(isinstance(
            mc_sm_model(scheme, **params), np.ndarray), True)

        param_vector = mc_sm_model.parameters_to_parameter_vector(
            **params)
        assert_equal(isinstance(
            mc_sm_model.simulate_signal(scheme, param_vector), np.ndarray),
            True)
