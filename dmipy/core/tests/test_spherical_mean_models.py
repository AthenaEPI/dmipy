from dmipy.signal_models import (
    cylinder_models, gaussian_models, sphere_models)
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalMeanModel)
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
import numpy as np
from dmipy.utils.spherical_mean import (
    estimate_spherical_mean_shell)
from numpy.testing import assert_equal, assert_almost_equal
from dipy.data import get_sphere

sphere = get_sphere().subdivide()

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


def test_model_spherical_mean_analytic_vs_numerical(
        bvalue=1e9, delta=1e-2, Delta=2e-2):
    bvals = np.tile(bvalue, len(sphere.vertices))
    scheme = acquisition_scheme_from_bvalues(
        bvals, sphere.vertices, delta, Delta)
    for model in models:
        params = {}
        for param, card in model.parameter_cardinality.items():
            params[param] = (np.random.rand(card) *
                             model.parameter_scales[param])

        signal_shell = model(scheme, **params)
        signal_shell_smt = np.mean(signal_shell)
        signal_smt = model.spherical_mean(scheme, **params)
        assert_almost_equal(signal_shell_smt, signal_smt, 2)


def test_model_spherical_mean_analytic_vs_sh(
        bvalue=1e9, delta=1e-2, Delta=2e-2):
    bvals = np.tile(bvalue, len(sphere.vertices))
    scheme = acquisition_scheme_from_bvalues(
        bvals, sphere.vertices, delta, Delta)
    for model in models:
        params = {}
        for param, card in model.parameter_cardinality.items():
            params[param] = (np.random.rand(card) *
                             model.parameter_scales[param])

        signal_shell = model(scheme, **params)
        signal_shell_sh = estimate_spherical_mean_shell(
            signal_shell, sphere.vertices)
        signal_smt = model.spherical_mean(scheme, **params)
        assert_almost_equal(signal_shell_sh, signal_smt, 2)


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
