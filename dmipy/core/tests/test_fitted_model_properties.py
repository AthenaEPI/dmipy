from dmipy.distributions import distribute_models
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
import numpy as np
from numpy.testing import assert_, assert_raises

scheme = wu_minn_hcp_acquisition_scheme()


def test_all_fitted_model_properties():
    stick = cylinder_models.C1Stick()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    params = {}
    for parameter, card, in watsonstick.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             watsonstick.parameter_scales[parameter])
    data = np.atleast_2d(watsonstick(scheme, **params))

    mcmod = modeling_framework.MultiCompartmentModel([watsonstick])
    mcfit = mcmod.fit(scheme, data)

    vertices = np.random.rand(10, 3)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]

    assert_(isinstance(mcfit.fitted_parameters, dict))
    assert_(isinstance(mcfit.fitted_parameters_vector, np.ndarray))
    assert_(isinstance(mcfit.fod(vertices), np.ndarray))
    assert_(isinstance(mcfit.fod_sh(), np.ndarray))
    assert_(isinstance(mcfit.peaks_spherical(), np.ndarray))
    assert_(isinstance(mcfit.peaks_cartesian(), np.ndarray))
    assert_(isinstance(mcfit.mean_squared_error(data), np.ndarray))
    assert_(isinstance(mcfit.R2_coefficient_of_determination(data),
                       np.ndarray))
    assert_(isinstance(mcfit.predict(), np.ndarray))

    mcmod_sm = modeling_framework.MultiCompartmentSphericalMeanModel(
        [stick])
    mcfit_sm = mcmod_sm.fit(scheme, data)
    assert_(isinstance(mcfit_sm.fitted_parameters, dict))
    assert_(isinstance(mcfit_sm.fitted_parameters_vector, np.ndarray))
    assert_(isinstance(mcfit.mean_squared_error(data), np.ndarray))
    assert_(isinstance(mcfit.R2_coefficient_of_determination(data),
                       np.ndarray))
    assert_(isinstance(mcfit.predict(), np.ndarray))


def test_parametric_fod_spherical_mean_model():
    stick = cylinder_models.C1Stick()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    params = {}
    for parameter, card, in watsonstick.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             watsonstick.parameter_scales[parameter])
    data = np.atleast_2d(watsonstick(scheme, **params))

    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    smt = modeling_framework.MultiCompartmentSphericalMeanModel(
        [stick, zeppelin])
    smt.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                               'C1Stick_1_lambda_par',
                               'partial_volume_0',
                               'partial_volume_1')
    smt.set_equal_parameter('G2Zeppelin_1_lambda_par',
                            'C1Stick_1_lambda_par')

    smt_fit = smt.fit(scheme, data)

    assert_raises(ValueError,
                  smt_fit.return_parametric_fod_model, Ncompartments=1.5)

    assert_raises(ValueError,
                  smt_fit.return_parametric_fod_model, Ncompartments=0)

    assert_raises(ValueError,
                  smt_fit.return_parametric_fod_model,
                  distribution='bla')

    for distribution_name in ['watson', 'bingham']:
        fod_model = smt_fit.return_parametric_fod_model(
            distribution=distribution_name, Ncompartments=1)
        fitted_fod_model = fod_model.fit(scheme, data)
        assert_(isinstance(fitted_fod_model.fitted_parameters, dict))
