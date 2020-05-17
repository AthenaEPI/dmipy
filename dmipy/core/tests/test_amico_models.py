from dmipy.distributions import distribute_models
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core import modeling_framework
from dmipy.data.saved_acquisition_schemes import (
    wu_minn_hcp_acquisition_scheme)
import numpy as np
from numpy.testing import assert_, assert_raises

scheme = wu_minn_hcp_acquisition_scheme()


def one_compartment_amico_model():
    """
    Provides a simple Stick AMICO-like model.

    This function builds an AMICO model with a single compartment given by a
    stick convolved with a watson distribution.

    All the parameters are set randomly.

    Returns:
        tuple of length 2 with
         * MultiCompartmentAMICOModel instance
         * sample data simulated on the hcp acquisition scheme
    """
    stick = cylinder_models.C1Stick()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    params = {}
    for parameter, card, in watsonstick.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             watsonstick.parameter_scales[parameter])

    model = modeling_framework.MultiCompartmentAMICOModel([watsonstick])
    data = np.atleast_2d(watsonstick(scheme, **params))
    return model, data


def two_compartment_amico_model():
    """
    Provides a simple Stick-and-Ball AMICO-like model.

    This function builds an AMICO model with two compartments given by a
    stick convolved with a watson distribution and a ball.

    All the parameters are set randomly.

    Returns:
        tuple of length 2 with
         * MultiCompartmentAMICOModel instance
         * sample data simulated on the hcp acquisition scheme
    """
    stick = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    watsonstick = distribute_models.SD1WatsonDistributed(
        [stick])
    model = modeling_framework.MultiCompartmentAMICOModel([watsonstick, ball])

    params = {}
    for parameter, card, in model.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             model.parameter_scales[parameter])

    for key, value in params.items():
        model.set_fixed_parameter(key, value)

    data = np.atleast_2d(model(scheme, **params))
    return model, data


def two_sticks_one_ball_amico_model():
    """
    Provides a simple Stick-Stick-and-Ball AMICO-like model.

    This function builds an AMICO model with two compartments given by a
    stick convolved with a watson distribution and a ball.

    All the parameters are set randomly.

    Returns:
        tuple of length 2 with
         * MultiCompartmentAMICOModel instance
         * sample data simulated on the hcp acquisition scheme
    """
    stick1 = cylinder_models.C1Stick()
    stick2 = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    watsonstick1 = distribute_models.SD1WatsonDistributed(
        [stick1])
    watsonstick2 = distribute_models.SD1WatsonDistributed(
        [stick2])
    compartments = [watsonstick1, watsonstick2, ball]
    model = modeling_framework.MultiCompartmentAMICOModel(compartments)

    params = {}
    for parameter, card, in model.parameter_cardinality.items():
        params[parameter] = (np.random.rand(card) *
                             model.parameter_scales[parameter])

    for key, value in params.items():
        model.set_fixed_parameter(key, value)

    data = np.atleast_2d(model(scheme, **params))
    return model, data


def test_forward_model():
    # TODO: test that the observation matrix is defined as expected
    pass


def test_nnls():
    # TODO: test the non-regularized version of the inverse problem
    pass


def test_l2_regularization():
    # TODO: test the effect of L2 regularization
    pass


def test_l1_regularization():
    # TODO: test the effect of L1 regularization
    pass


def test_l1_and_l2_regularization():
    # TODO: test the effect of using both L1 and L2 regularization
    pass


def test_fitted_amico_model_properties():
    # TODO: check that the FittedMultiCompartmentAMICOModel class has the
    #  expected properties
    pass

def test_signal_fitting():
    # TODO: test the accuracy of the fitting on a synthetic signal
    pass