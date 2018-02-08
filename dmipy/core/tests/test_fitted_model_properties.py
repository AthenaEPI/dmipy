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
    data = watsonstick(scheme, **params)

    mcmod = modeling_framework.MultiCompartmentModel([watsonstick])
    mcfit = mcmod.fit(scheme, data)

    vertices = np.random.rand(10, 3)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]

    mcfit.fitted_parameters
    mcfit.fitted_parameters_vector
    mcfit.fod(vertices)
    mcfit.fod_sh()
    mcfit.peaks_spherical()
    mcfit.peaks_cartesian()

    mcmod_sm = modeling_framework.MultiCompartmentSphericalMeanModel(
        [stick])
    mcfit_sm = mcmod_sm.fit(scheme, data)
    mcfit_sm.fitted_parameters
    mcfit_sm.fitted_parameters_vector
