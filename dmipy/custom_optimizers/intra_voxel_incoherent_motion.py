from dmipy.signal_models.gaussian_models import G1Ball
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
import numpy as np
from time import time


def ivim_2step(acquisition_scheme, data, mask=None, bvalue_threshold=4e8,
               solver='brute2fine', optimize_S0=True, **fit_args):
    """
    Dmipy implementation of the classic 2-compartment intra-voxel incoherent
    motion (IVIM) model [1]_, following the 2-step optimization scheme. The
    model consists of 2 Ball compartments (isotropic Gaussian), each fitting
    the blood flow and diffusion volume fractions and diffusivities,
    respectively. Changes in e.g. blood volume fraction has been linked to many
    pathologies such as the vasculature in tumor tissue [2]_.

    Because the apparent diffusivity of blood flow is much higher than that of
    Brownian motion, the optimization bounds for the diffusivities of the two
    Balls are disjoint; the diffusivies of the diffusion compartment range
    between [0.5 - 6]e-3 mm^2/s (results in more precise fit according to [3]),
    and those of the blood compartment range between [6 - 20]e-3 mm^2/s
    (following [4]).

    The 2-step optimization [5] hinges on the observation that the blood-flow
    signal is negligible at b-values above 200-400 s/mm^2, but it does have
    a constribution below that bvalue (and to the b0).
    The optimization steps are as follows:
    - step 1: fit only the "diffusion" part of the data using a single Ball
        compartment, so the data is truncated to only include measurements
        above the bvalue_threshold value. This step estimates the "diffusion"
        S0 (which is lower or equal to the actual SO) and the "diffusion"
        diffusivity of this compartment.
    - step 2: fit the 2-compartment model to the whole signal, but fixing the
        "diffusion" diffusivity to the value estimated in step 1.

    In the fitted ivim_fit model, partial_volume_0 and G1Ball_1_lambda_iso
    represent the tissue fraction and diffusivity, and partial_volume_1 and
    G1Ball_2_lambda_iso represent the blood fraction and diffusivity.

    Parameters
    ----------
    acquisition_scheme: Dmipy AcquisitionScheme instance,
        acquisition scheme containing all the information of the ivim
        acquisition.
    data: ND-array of shape (Nx, ..., N_DWI),
        measured data corresponding to the acquisition scheme.
    mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
        Optional mask of voxels to be included in the optimization.
    bvalue_threshold: float,
        the bvalue threshold at which to separate the blood/diffusion parts of
        the data.
        Default: 400s/mm^2, but other works experiment with this value.
    solver: float,
        which solver to use for the algorithm. Default: 'brute2fine'.
    optimize_S0: boolean,
        whether or not to optimize (or just fix it to the mean of the b0-data)
        the S0 value in the second optimization step.
    fit_args: other keywords that are passed to the optimizer

    Returns
    -------
    ivim_fit: Dmipy FittedMultiCompartmentModel instance,
        contains the fitted IVIM parameters.

    References
    ----------
    .. [1] Le Bihan, D., Breton, E., Lallemand, D., Aubin, M. L., Vignaud, J.,
        & Laval-Jeantet, M. (1988). Separation of diffusion and perfusion in
        intravoxel incoherent motion MR imaging. Radiology, 168(2), 497-505.
    .. [2] Le Bihan, D. (2017). What can we see with IVIM MRI?. NeuroImage.
    .. [3] Gurney-Champion OJ, Froeling M, Klaassen R, Runge JH, Bel A, Van
        Laarhoven HWM, et al. Minimizing the Acquisition Time for Intravoxel
        Incoherent Motion Magnetic Resonance Imaging Acquisitions in the Liver
        and Pancreas. Invest Radiol. 2016;51: 211–220.
    .. [4] Park HJ, Sung YS, Lee SS, Lee Y, Cheong H, Kim YJ, et al. Intravoxel
        incoherent motion diffusion-weighted MRI of the abdomen: The effect of
        fitting algorithms on the accuracy and reliability of the parameters.
        J Magn Reson Imaging. 2017;45: 1637–1647.
    """
    start = time()

    if fit_args is None:
        fit_args = {}
    fit_args.update({'verbose': False, 'mask': mask, 'solver': solver})

    bvalue_mask = acquisition_scheme.bvalues > bvalue_threshold
    gaussian_acquisition_scheme = acquisition_scheme_from_bvalues(
        bvalues=acquisition_scheme.bvalues[bvalue_mask],
        gradient_directions=acquisition_scheme.gradient_directions[
            bvalue_mask])

    gaussian_data = np.atleast_2d(data)[..., bvalue_mask]

    gaussian_mod = MultiCompartmentModel([G1Ball()])
    gaussian_mod.set_parameter_optimization_bounds(
        'G1Ball_1_lambda_iso', [0.5e-9, 6e-9])  # [3]
    print('Starting step 1 of IVIM 2-step algorithm.')
    gaussian_fit = gaussian_mod.fit(
        acquisition_scheme=gaussian_acquisition_scheme,
        data=gaussian_data,
        optimize_S0=True,
        **fit_args)

    ivim_mod = MultiCompartmentModel([G1Ball(), G1Ball()])
    ivim_mod.set_parameter_optimization_bounds(
        'G1Ball_2_lambda_iso', [6e-9, 20e-9])  # [4]
    ivim_mod.set_fixed_parameter(
        parameter_name='G1Ball_1_lambda_iso',
        value=gaussian_fit.fitted_parameters['G1Ball_1_lambda_iso'])
    print('Starting step 2 of IVIM 2-step algorithm.')
    ivim_fit = ivim_mod.fit(
        acquisition_scheme=acquisition_scheme,
        data=data,
        optimize_S0=optimize_S0,
        **fit_args)

    computation_time = time() - start
    N_voxels = np.sum(ivim_fit.mask)
    msg = 'IVIM 2-step optimization of {0:d} voxels'.format(N_voxels)
    msg += ' complete in {0:.3f} seconds'.format(computation_time)
    print(msg)
    return ivim_fit


def ivim_Dstar_fixed(acquisition_scheme, data, mask=None, Dstar_value=7e-9,
                     solver='brute2fine', optimize_S0=True, **fit_args):
    """
    Implementation of second best performing IVIM algorithm following [1]_.
    Basically, it is just a non-linear least squares fit with fixing the
    blood diffusivity Dstar to 7e-3 mm^2/s. This value apparently improves the
    stability of the fit (in healthy volunteers) [2]_.

    The optimization range for the tissue diffusivity is set to
    [0.5 - 6]e-3 mm^2/s to improve precision [3]_.

    In the fitted ivim_fit model, partial_volume_0 and G1Ball_1_lambda_iso
    represent the tissue fraction and diffusivity, and partial_volume_1 and
    G1Ball_2_lambda_iso represent the blood fraction and diffusivity.

    Parameters
    ----------
    acquisition_scheme: Dmipy AcquisitionScheme instance,
        acquisition scheme containing all the information of the ivim
        acquisition.
    data: ND-array of shape (Nx, ..., N_DWI),
        measured data corresponding to the acquisition scheme.
    mask : (N-1)-dimensional integer/boolean array of size (N_x, N_y, ...),
        Optional mask of voxels to be included in the optimization.
    Dstar_value: float,
        the fixed Dstar blood diffusivity value. Default: 7e-9 m^2/s [2]_.
    solver: float,
        which solver to use for the algorithm. Default: 'brute2fine'.
    optimize_S0: boolean,
        whether or not to optimize (or just fix it to the mean of the b0-data)
        the S0 value in the second optimization step.
    fit_args: other keywords that are passed to the optimizer

    Returns
    -------
    ivim_fit: Dmipy FittedMultiCompartmentModel instance,
        contains the fitted IVIM parameters.

    References
    ----------
    .. [1] Gurney-Champion, O. J., Klaassen, R., Froeling, M., Barbieri, S.,
        Stoker, J., Engelbrecht, M. R., ... & Nederveen, A. J. (2018).
        Comparison of six fit algorithms for the intra-voxel incoherent motion
        model of diffusion-weighted magnetic resonance imaging data of
        pancreatic cancer patients. PloS one, 13(4), e0194590.
    .. [2] Gurney-Champion OJ, Froeling M, Klaassen R, Runge JH, Bel A, Van
        Laarhoven HWM, et al. Minimizing the Acquisition Time for Intravoxel
        Incoherent Motion Magnetic Resonance Imaging Acquisitions in the Liver
        and Pancreas. Invest Radiol. 2016;51: 211–220.
    .. [3] Park HJ, Sung YS, Lee SS, Lee Y, Cheong H, Kim YJ, et al. Intravoxel
        incoherent motion diffusion-weighted MRI of the abdomen: The effect of
        fitting algorithms on the accuracy and reliability of the parameters.
        J Magn Reson Imaging. 2017;45: 1637–1647.
    """
    start = time()

    if fit_args is None:
        fit_args = {}

    print('Starting IVIM Dstar-fixed algorithm.')
    ivim_mod = MultiCompartmentModel([G1Ball(), G1Ball()])
    ivim_mod.set_fixed_parameter(
        'G1Ball_2_lambda_iso', Dstar_value)  # following [2]
    ivim_mod.set_parameter_optimization_bounds(
        'G1Ball_1_lambda_iso', [.5e-9, 6e-9])  # following [3]
    ivim_fit = ivim_mod.fit(
        acquisition_scheme=acquisition_scheme,
        data=data,
        mask=mask,
        solver=solver,
        optimize_S0=optimize_S0,
        verbose=False,
        **fit_args)
    computation_time = time() - start
    N_voxels = np.sum(ivim_fit.mask)
    msg = 'IVIM Dstar-fixed optimization of {0:d} voxels'.format(N_voxels)
    msg += ' complete in {0:.3f} seconds'.format(computation_time)
    print(msg)
    return ivim_fit
