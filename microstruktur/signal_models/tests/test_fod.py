from microstruktur.signal_models import dispersed_models
from microstruktur.signal_models.three_dimensional_models import (
    PartialVolumeCombinedMicrostrukturModel)
from dipy.data import get_sphere
from microstruktur.signal_models.spherical_mean import (
    estimate_spherical_mean_shell)
import numpy as np
from numpy.testing import assert_almost_equal
sphere = get_sphere()


def test_fod():
    dispersed_models_ = (
        dispersed_models.SD2I1BinghamDispersedStick(),
        dispersed_models.SD2I2BinghamDispersedSodermanCylinder(),
        dispersed_models.SD2I3BinghamDispersedCallaghanCylinder(),
        dispersed_models.SD2I4BinghamDispersedGaussianPhaseCylinder(),
        dispersed_models.SD2E4BinghamDispersedZeppelin(),
        dispersed_models.SD3I1WatsonDispersedStick(),
        dispersed_models.SD3I2WatsonDispersedSodermanCylinder(),
        dispersed_models.SD3I3WatsonDispersedCallaghanCylinder(),
        dispersed_models.SD3I4WatsonDispersedGaussianPhaseCylinder(),
        dispersed_models.SD3E4WatsonDispersedZeppelin())

    vertices = sphere.subdivide().vertices
    for model in dispersed_models_:
        mc_model = PartialVolumeCombinedMicrostrukturModel([model, model])
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
