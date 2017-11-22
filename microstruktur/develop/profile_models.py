from os.path import join
from microstruktur.signal_models import (
    cylinder_models,
    gaussian_models,
    spherical_mean_models,
    dispersed_models)
from microstruktur.core import modeling_framework
import cProfile
import numpy as np
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues)


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


def profile_C1Stick(reps=1000):
    model = cylinder_models.C1Stick()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_C2CylinderSodermanApproximation(reps=1000):
    model = cylinder_models.C2CylinderSodermanApproximation()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_C3CylinderCallaghanApproximation(reps=1000):
    model = cylinder_models.C3CylinderCallaghanApproximation()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G2Dot(reps=1000):
    model = gaussian_models.G2Dot()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G2Dot(reps=1000):
    model = gaussian_models.G2Dot()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G3Ball(reps=1000):
    model = gaussian_models.G3Ball()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G4Zeppelin(reps=1000):
    model = gaussian_models.G4Zeppelin()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G5RestrictedZeppelin(reps=1000):
    model = gaussian_models.G5RestrictedZeppelin()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD2C1BinghamDispersedStick(reps=1000):
    model = dispersed_models.SD2C1BinghamDispersedStick()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD2C2BinghamDispersedSodermanCylinder(reps=1000):
    model = dispersed_models.SD2C2BinghamDispersedSodermanCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD2C3BinghamDispersedCallaghanCylinder(reps=1000):
    model = dispersed_models.SD2C3BinghamDispersedCallaghanCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD2C4BinghamDispersedGaussianPhaseCylinder(reps=1000):
    model = dispersed_models.SD2C4BinghamDispersedGaussianPhaseCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD1C1WatsonDispersedStick(reps=1000):
    model = dispersed_models.SD1C1WatsonDispersedStick()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD1C2WatsonDispersedSodermanCylinder(reps=1000):
    model = dispersed_models.SD1C2WatsonDispersedSodermanCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD1C3WatsonDispersedCallaghanCylinder(reps=1000):
    model = dispersed_models.SD1C3WatsonDispersedCallaghanCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD1C4WatsonDispersedGaussianPhaseCylinder(reps=1000):
    model = dispersed_models.SD1C4WatsonDispersedGaussianPhaseCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD2G4BinghamDispersedZeppelin(reps=1000):
    model = dispersed_models.SD2G4BinghamDispersedZeppelin()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_SD1G4WatsonDispersedZeppelin(reps=1000):
    model = dispersed_models.SD1G4WatsonDispersedZeppelin()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_DD1C2GammaDistributedSodermanCylinder(reps=1000):
    model = dispersed_models.DD1C2GammaDistributedSodermanCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_DD1C3GammaDistributedCallaghanCylinder(reps=1000):
    model = dispersed_models.DD1C3GammaDistributedCallaghanCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_DD1C4GammaDistributedGaussianPhaseCylinder(reps=1000):
    model = dispersed_models.DD1C4GammaDistributedGaussianPhaseCylinder()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_C1StickSphericalMean(reps=1000):
    model = spherical_mean_models.C1StickSphericalMean()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G4ZeppelinSphericalMean(reps=1000):
    model = spherical_mean_models.G4ZeppelinSphericalMean()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())


def profile_G5RestrictedZeppelinSphericalMean(reps=1000):
    model = spherical_mean_models.G5RestrictedZeppelinSphericalMean()
    mc_model = modeling_framework.MultiCompartmentMicrostructureModel(
        acquisition_scheme=scheme,
        models=[model])
    mc_params = {}
    for name in mc_model.parameter_cardinality:
        card, scale = (
            mc_model.parameter_cardinality[name],
            mc_model.parameter_scales[name])
        mc_params[name] = np.random.rand(card) * scale
    mc_params_vector = mc_model.parameters_to_parameter_vector(**mc_params)
    mc_params_vector = np.tile(mc_params_vector, (reps, 1))
    cProfile.runctx('mc_model.simulate_signal(scheme, mc_params_vector)',
                    globals(), locals())
