from dmipy.signal_models import (
    cylinder_models, gaussian_models, sphere_models, plane_models)
from dmipy.distributions import distribute_models, distributions
from dmipy.data.saved_acquisition_schemes import wu_minn_hcp_acquisition_scheme
import numpy as np
from numpy.testing import (
    assert_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises)


def test_all_models_dispersable():
    scheme = wu_minn_hcp_acquisition_scheme()

    dispersable_models = [
        [cylinder_models.C1Stick()],
        [cylinder_models.C2CylinderSodermanApproximation()],
        [cylinder_models.C3CylinderCallaghanApproximation()],
        [cylinder_models.C4CylinderGaussianPhaseApproximation()],
        [gaussian_models.G1Ball(), gaussian_models.G2Zeppelin()],
        [gaussian_models.G3RestrictedZeppelin()],
        [sphere_models.S1Dot(), gaussian_models.G2Zeppelin()],
        [sphere_models.S2SphereSodermanApproximation(),
         gaussian_models.G2Zeppelin()]
    ]

    spherical_distributions = [
        distribute_models.SD1WatsonDistributed,
        distribute_models.SD2BinghamDistributed
    ]

    for model in dispersable_models:
        for distribution in spherical_distributions:
            dist_mod = distribution(model)
            params = {}
            for param, card in dist_mod.parameter_cardinality.items():
                params[param] = np.random.rand(
                    card) * dist_mod.parameter_scales[param]
            assert_equal(isinstance(
                dist_mod(scheme, **params), np.ndarray), True)


def test_raises_models_with_no_orientation():
    non_dispersable_models = [
        gaussian_models.G1Ball(),
        sphere_models.S1Dot(),
        sphere_models.S2SphereSodermanApproximation()
    ]

    spherical_distributions = [
        distribute_models.SD1WatsonDistributed,
        distribute_models.SD2BinghamDistributed
    ]

    for model in non_dispersable_models:
        for distribution in spherical_distributions:
            assert_raises(ValueError, distribution, [model])


def test_gamma_pdf_unity():
    normalizations = ['standard', 'plane', 'cylinder', 'sphere']

    alpha = 1.
    beta = 3e-6

    for normalization in normalizations:
        gamma = distributions.DD1Gamma(
            alpha=alpha, beta=beta, normalization=normalization)
        x, Px = gamma()
        assert_almost_equal(np.trapz(Px, x=x), 1.)


def test_all_models_distributable():
    scheme = wu_minn_hcp_acquisition_scheme()

    distributable_models = [
        plane_models.P3PlaneCallaghanApproximation,
        cylinder_models.C2CylinderSodermanApproximation,
        cylinder_models.C3CylinderCallaghanApproximation,
        cylinder_models.C4CylinderGaussianPhaseApproximation,
        sphere_models.S2SphereSodermanApproximation
    ]

    spatial_distributions = [
        distribute_models.DD1GammaDistributed
    ]

    for model in distributable_models:
        for distribution in spatial_distributions:
            mod = model()
            dist_mod = distribution([mod])
            params = {}
            for param, card in dist_mod.parameter_cardinality.items():
                params[param] = np.random.rand(
                    card) * dist_mod.parameter_scales[param]
            assert_equal(isinstance(
                dist_mod(scheme, **params), np.ndarray), True)


def test_C2_watson_gamma_equals_gamma_watson():
    scheme = wu_minn_hcp_acquisition_scheme()

    cylinder = cylinder_models.C2CylinderSodermanApproximation()
    watsoncyl = distribute_models.SD1WatsonDistributed([cylinder])

    gammawatsoncyl = distribute_models.DD1GammaDistributed(
        [watsoncyl],
        target_parameter='C2CylinderSodermanApproximation_1_diameter')

    params1 = {
        'SD1WatsonDistributed_1_C2CylinderSodermanApproximation_1_lambda_par':
        1.7e-9,
        'DD1Gamma_1_alpha': 2.,
        'DD1Gamma_1_beta': 4e-6,
        'SD1WatsonDistributed_1_SD1Watson_1_odi': 0.4,
        'SD1WatsonDistributed_1_SD1Watson_1_mu': [0., 0.]
    }
    gammacyl = distribute_models.DD1GammaDistributed([cylinder])
    watsongammacyl = distribute_models.SD1WatsonDistributed(
        [gammacyl])

    params2 = {
        'DD1GammaDistributed_1_C2CylinderSodermanApproximation_1_lambda_par':
        1.7e-9,
        'DD1GammaDistributed_1_DD1Gamma_1_alpha': 2.,
        'DD1GammaDistributed_1_DD1Gamma_1_beta': 4e-6,
        'SD1Watson_1_odi': 0.4,
        'SD1Watson_1_mu': [0., 0.]
    }

    assert_array_almost_equal(watsongammacyl(scheme, **params2),
                              gammawatsoncyl(scheme, **params1), 5)


def test_C3_watson_gamma_equals_gamma_watson():
    scheme = wu_minn_hcp_acquisition_scheme()

    cylinder = cylinder_models.C3CylinderCallaghanApproximation()
    watsoncyl = distribute_models.SD1WatsonDistributed([cylinder])

    gammawatsoncyl = distribute_models.DD1GammaDistributed(
        [watsoncyl],
        target_parameter='C3CylinderCallaghanApproximation_1_diameter')

    params1 = {
        'SD1WatsonDistributed_1_C3CylinderCallaghanApproximation_1_lambda_par':
        1.7e-9,
        'DD1Gamma_1_alpha': 2.,
        'DD1Gamma_1_beta': 4e-6,
        'SD1WatsonDistributed_1_SD1Watson_1_odi': 0.4,
        'SD1WatsonDistributed_1_SD1Watson_1_mu': [0., 0.]
    }
    gammacyl = distribute_models.DD1GammaDistributed([cylinder])
    watsongammacyl = distribute_models.SD1WatsonDistributed(
        [gammacyl])

    params2 = {
        'DD1GammaDistributed_1_C3CylinderCallaghanApproximation_1_lambda_par':
        1.7e-9,
        'DD1GammaDistributed_1_DD1Gamma_1_alpha': 2.,
        'DD1GammaDistributed_1_DD1Gamma_1_beta': 4e-6,
        'SD1Watson_1_odi': 0.4,
        'SD1Watson_1_mu': [0., 0.]
    }

    assert_array_almost_equal(watsongammacyl(scheme, **params2),
                              gammawatsoncyl(scheme, **params1), 5)


def test_C4_watson_gamma_equals_gamma_watson():
    scheme = wu_minn_hcp_acquisition_scheme()

    cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation()
    watsoncyl = distribute_models.SD1WatsonDistributed([cylinder])

    gammawatsoncyl = distribute_models.DD1GammaDistributed(
        [watsoncyl],
        target_parameter='C4CylinderGaussianPhaseApproximation_1_diameter')

    param = 'SD1WatsonDistributed_1_C4CylinderGaussianPhaseApproximation'
    param += '_1_lambda_par'

    params1 = {
        param: 1.7e-9,
        'DD1Gamma_1_alpha': 2.,
        'DD1Gamma_1_beta': 4e-6,
        'SD1WatsonDistributed_1_SD1Watson_1_odi': 0.4,
        'SD1WatsonDistributed_1_SD1Watson_1_mu': [0., 0.]
    }
    gammacyl = distribute_models.DD1GammaDistributed([cylinder])
    watsongammacyl = distribute_models.SD1WatsonDistributed(
        [gammacyl])

    param = 'DD1GammaDistributed_1_C4CylinderGaussianPhaseApproximation'
    param += '_1_lambda_par'

    params2 = {
        param: 1.7e-9,
        'DD1GammaDistributed_1_DD1Gamma_1_alpha': 2.,
        'DD1GammaDistributed_1_DD1Gamma_1_beta': 4e-6,
        'SD1Watson_1_odi': 0.4,
        'SD1Watson_1_mu': [0., 0.]
    }

    assert_array_almost_equal(watsongammacyl(scheme, **params2),
                              gammawatsoncyl(scheme, **params1), 5)
