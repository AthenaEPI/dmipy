import numpy as np

from numpy.testing import assert_almost_equal
from dmipy.data import saved_acquisition_schemes

from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel

from dmipy.optimizers import amico_cvxpy


def create_noddi_watson_model(lambda_iso_diff=3.e-9, lambda_par_diff=1.7e-9):
    """Creates NODDI mulit-compartment model with Watson distribution."""
    """
        Arguments:
            lambda_iso_diff: float
                isotropic diffusivity
            lambda_par_diff: float
                parallel diffusivity
        Returns: MultiCompartmentModel instance
            NODDI Watson multi-compartment model instance
    """
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
    watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp',
                                                   'C1Stick_1_lambda_par',
                                                   'partial_volume_0')
    watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par',
                                                'C1Stick_1_lambda_par')
    watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par',
                                                lambda_par_diff)

    NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso_diff)

    return NODDI_mod


def forward_model_matrix(mc_model, acquisition_scheme, model_dirs, Nt=12):
    """Creates forward model matrix."""
    """
        Arguments:
            mc_model: MultiCompartmentModel instance
                multi-compartment model instance
            acquisition_scheme: DmipyAcquisitionScheme instance
                acquisition scheme
            model_dirs: list
                containing direction of all models in multi-compartment model
            Nt: int
                number of samples for tessellation of parameter range
        Returns:
            M: Array of size (Ndata, Nx)
                The observation matrix containing Nx model atoms
            grid: dict
                Dictionary containing tessellation of parameters to be
                estimated for each model within multi-compartment model
            idx: dict
                Dictionary containing indices that correspond to the
                parameters to be estimated for each model within
                multi-compartment model
    """
    N_models = len(mc_model.models)

    dir_params = [p for p in mc_model.parameter_names if p.endswith('mu')]
    if len(dir_params) != len(model_dirs):
        raise ValueError("Length of model_dirs should correspond "
                         "to the number of directional parameters!")
    grid_params = [p for p in mc_model.parameter_names
                   if not p.endswith('mu') and
                   not p.startswith('partial_volume')]

    _amico_grid, _amico_idx = {}, {}

    # Compute length of the vector x0
    x0_len = 0
    for m_idx in range(N_models):
        m_atoms = 1
        for p in mc_model.models[m_idx].parameter_names:
            if mc_model.model_names[m_idx] + p in grid_params:
                m_atoms *= Nt
        x0_len += m_atoms

    for m_idx in range(N_models):
        model = mc_model.models[m_idx]
        model_name = mc_model.model_names[m_idx]

        param_sampling, grid_params_names = [], []
        m_atoms = 1
        for p in model.parameter_names:
            if model_name + p not in grid_params:
                continue
            grid_params_names.append(model_name + p)
            p_range = mc_model.parameter_ranges[model_name + p]
            _amico_grid[model_name + p] = np.full(x0_len, np.mean(p_range))
            param_sampling.append(np.linspace(p_range[0], p_range[1], Nt,
                                              endpoint=True))
            m_atoms *= Nt

        _amico_idx[model_name] =\
            sum([len(_amico_idx[k]) for k in _amico_idx]) + np.arange(m_atoms)

        params_mesh = np.meshgrid(*param_sampling)
        for p_idx, p in enumerate(grid_params_names):
            _amico_grid[p][_amico_idx[model_name]] = \
                np.ravel(params_mesh[p_idx])

        _amico_grid['partial_volume_' + str(m_idx)] = np.zeros(x0_len)
        _amico_grid['partial_volume_' +
                    str(m_idx)][_amico_idx[model_name]] = 1.

    for d_idx, dp in enumerate(dir_params):
        _amico_grid[dp] = model_dirs[d_idx]

    return (mc_model.simulate_signal(acquisition_scheme, _amico_grid).T,
            _amico_grid, _amico_idx)


def simulate_signals(mc_model, acquisition_scheme, n_samples=100):
    """Simulates signals for given multi-compartment model."""
    """
        Arguments:
            mc_model: MultiCompartmentModel instance
                multi-compartment model instance
            acquisition_scheme: DmipyAcquisitionScheme instance
                acquisition scheme
            n_samples: int
                number of samples to generate
        Returns:
            simulated_signals: Array of size (Nsamples, Ndata)
                simulated signals
            ground_truth: dict
                dictionary containing ground truth for each parameter
            dirs: Array of size (Nsamples, 2)
                direction of anisotropic compartment
    """
    np.random.seed(123)
    arguments = {}
    arguments['partial_volume_0'] = np.random.uniform(0., 1., n_samples)
    arguments['partial_volume_1'] = 1. - arguments['partial_volume_0']
    for p in mc_model.parameter_names:
        if p.startswith('partial_volume'):
            continue
        p_range = mc_model.parameter_ranges[p]
        if p.endswith('mu'):
            theta = np.random.uniform(p_range[0][0], p_range[0][1], n_samples)
            phi = np.random.uniform(p_range[1][0], p_range[1][1], n_samples)
            arg_samples = np.column_stack((theta, phi))
            directions = arg_samples
        else:
            arg_samples = np.random.uniform(p_range[0], p_range[1], n_samples)
        arguments[p] = arg_samples

    return (mc_model.simulate_signal(acquisition_scheme, arguments),
            arguments, directions)


def test_amico(lambda_1=[0, 0.0001], lambda_2=[0, 0.0000001], Nt=12):
    """Tests amico optimizer."""
    """
        Arguments:
            lambda_1: list
                list of L1 regularization constants for each compartment
            lambda_2: list
                list of L2 regularization constants for each compartment
            Nt: int
                number of samples for tessellation of parameter range
    """
    mc_model = create_noddi_watson_model()
    scheme_hcp = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    grid_params = [p for p in mc_model.parameter_names
                   if not p.endswith('mu') and
                   not p.startswith('partial_volume')]

    x0_len = 0
    for m_idx in range(len(mc_model.models)):
        m_atoms = 1
        for p in mc_model.models[m_idx].parameter_names:
            if mc_model.model_names[m_idx] + p in grid_params:
                m_atoms *= Nt
        x0_len += m_atoms

    amico_opt = amico_cvxpy.AmicoCvxpyOptimizer(mc_model, scheme_hcp,
                                                lambda_1=lambda_1,
                                                lambda_2=lambda_2)

    signals, gt, dirs = simulate_signals(mc_model, scheme_hcp)

    v_iso_estim = np.zeros(signals.shape[0])
    v_ic_estim = np.zeros(signals.shape[0])
    od_estim = np.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        model_dirs = [dirs[i, :]]
        M, grid, idx = \
            forward_model_matrix(mc_model, scheme_hcp, model_dirs, Nt)
        parameters = amico_opt(signals[i, :], M, grid, idx)
        v_iso_estim[i] = parameters[0]
        od_estim[i] = parameters[2]
        v_ic_estim[i] = parameters[3]

    assert_almost_equal(v_iso_estim,
                        gt['partial_volume_0'], 2)
    assert_almost_equal(v_ic_estim,
                        gt['SD1WatsonDistributed_1_partial_volume_0'], 2)
    assert_almost_equal(od_estim,
                        gt['SD1WatsonDistributed_1_SD1Watson_1_odi'], 1)
