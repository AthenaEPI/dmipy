from microstruktur.signal_models import dispersed_models
from microstruktur.core.modeling_framework import (
    MultiCompartmentMicrostructureModel)
from dipy.data import get_sphere
from os.path import join
from microstruktur.core import modeling_framework
from microstruktur.utils.spherical_mean import (
    estimate_spherical_mean_shell)
import numpy as np
from numpy.testing import assert_almost_equal
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)
sphere = get_sphere()

bvals = np.loadtxt(
    join(modeling_framework.GRADIENT_TABLES_PATH,
         'bvals_hcp_wu_minn.txt')
)
bvals *= 1e6
gradient_directions = np.loadtxt(
    join(modeling_framework.GRADIENT_TABLES_PATH,
         'bvecs_hcp_wu_minn.txt')
)
delta = 0.01
Delta = 0.03
scheme = acquisition_scheme_from_bvalues(
    bvals, gradient_directions, delta, Delta)


def test_fod():
    dispersed_models_ = (
        (dispersed_models.SD2C1BinghamDispersedStick(),
         dispersed_models.SD2C1BinghamDispersedStick()),
        (dispersed_models.SD2C2BinghamDispersedSodermanCylinder(),
         dispersed_models.SD2C2BinghamDispersedSodermanCylinder()),
        (dispersed_models.SD2C3BinghamDispersedCallaghanCylinder(),
         dispersed_models.SD2C3BinghamDispersedCallaghanCylinder()),
        (dispersed_models.SD2C4BinghamDispersedGaussianPhaseCylinder(),
         dispersed_models.SD2C4BinghamDispersedGaussianPhaseCylinder()),
        (dispersed_models.SD2G4BinghamDispersedZeppelin(),
         dispersed_models.SD2G4BinghamDispersedZeppelin()),
        (dispersed_models.SD1C1WatsonDispersedStick(),
         dispersed_models.SD1C1WatsonDispersedStick()),
        (dispersed_models.SD1C2WatsonDispersedSodermanCylinder(),
         dispersed_models.SD1C2WatsonDispersedSodermanCylinder()),
        (dispersed_models.SD1C3WatsonDispersedCallaghanCylinder(),
         dispersed_models.SD1C3WatsonDispersedCallaghanCylinder()),
        (dispersed_models.SD1C4WatsonDispersedGaussianPhaseCylinder(),
         dispersed_models.SD1C4WatsonDispersedGaussianPhaseCylinder()),
        (dispersed_models.SD1G4WatsonDispersedZeppelin(),
         dispersed_models.SD1G4WatsonDispersedZeppelin()))

    vertices = sphere.subdivide().vertices
    for (model1, model2) in dispersed_models_:
        mc_model = MultiCompartmentMicrostructureModel(
            acquisition_scheme=scheme,
            models=[model1, model2])
        mc_params = {}
        for name in mc_model.parameter_cardinality:
            card, scale = (
                mc_model.parameter_cardinality[name],
                mc_model.parameter_scales[name])
            mc_params[name] = np.random.rand(card) * scale
        mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
        fod = mc_model.fod(vertices, mc_params_vector)
        spherical_mean = estimate_spherical_mean_shell(
            fod, vertices, sh_order=10)
        assert_almost_equal(spherical_mean * 4 * np.pi, 1., 4)
