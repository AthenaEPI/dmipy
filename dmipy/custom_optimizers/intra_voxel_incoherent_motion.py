from dmipy.signal_models.gaussian_models import G1Ball
from dmipy.core.modeling_framework import MultiCompartmentModel
import numpy as np
from time import time
import logging


def ivim_Dstar_fixed(acquisition_scheme, data, mask=None, Dstar_value=7e-9,
                     solver='brute2fine', **fit_args):
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

    logging.info('Starting IVIM Dstar-fixed algorithm.')
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
        **fit_args)
    computation_time = time() - start
    N_voxels = np.sum(ivim_fit.mask)
    msg = 'IVIM Dstar-fixed optimization of {0:d} voxels'.format(N_voxels)
    msg += ' complete in {0:.3f} seconds'.format(computation_time)
    logging.info(msg)
    return ivim_fit
