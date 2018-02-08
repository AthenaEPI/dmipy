from dmipy.distributions import distribute_models
from dmipy.signal_models import cylinder_models
from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
import numpy as np

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

    isinstance(mcfit.fitted_parameters, dict)
    isinstance(mcfit.fitted_parameters_vector, np.ndarray)
    isinstance(mcfit.fod(vertices), np.ndarray)
    isinstance(mcfit.fod_sh(), np.ndarray)
    isinstance(mcfit.peaks_spherical(), np.ndarray)
    isinstance(mcfit.peaks_cartesian(), np.ndarray)
    isinstance(mcfit.mean_squared_error(data), np.ndarray)
    isinstance(mcfit.R2_coefficient_of_determination(data), np.ndarray)
    isinstance(mcfit.predict(), np.ndarray)

    mcmod_sm = modeling_framework.MultiCompartmentSphericalMeanModel(
        [stick])
    mcfit_sm = mcmod_sm.fit(scheme, data)
    isinstance(mcfit_sm.fitted_parameters, dict)
    isinstance(mcfit_sm.fitted_parameters_vector, np.ndarray)
    isinstance(mcfit.mean_squared_error(data), np.ndarray)
    isinstance(mcfit.R2_coefficient_of_determination(data), np.ndarray)
    isinstance(mcfit.predict(), np.ndarray)
