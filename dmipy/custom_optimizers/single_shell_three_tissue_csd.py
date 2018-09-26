import time
import numpy as np
from dmipy.core.modeling_framework import (
    MultiCompartmentSphericalHarmonicsModel)
from dmipy.tissue_response.three_tissue_response import (
    three_tissue_response_dhollander16)


def single_shell_three_tissue_csd(
        acquisition_scheme, data, tissue_responses=None, N_iterations=4,
        return_all_csd_fits=False, csd_fit_verbose=False, ss3t_verbose=True):
    """
    Implementation of Single-Shell (1 shell + b0) three-tissue CSD algorithm
    [1]_. The algorithm is based on a biconvex optimization strategy that is
    formulated as two steps that repeat:

    for k in iterations:
        # step 1
        if k == 0:
            fix WM fraction to 0 and optimize CSF + GM.
            this leads to an overestimation of GM and underestimation of CSF.
        else:
            fix WM to WM estimate of second step of last iteration.
        # step 2
        Fix CSF and fit WM + GM.
        Since CSF is an underestimate, WM will be as well.

    The authors of [1]_ don't really elaborate on this, but this algorithm will
    just slowly converge to the naive optimization approach of just fitting
    WM + GM + CSF together. The trick is that by forcing a particular starting
    point for the biconvex optimization (WM=0), typically after around 4
    iterations of this algorithm, the fractions will *temporarily* have values
    that are quite close to the multi-shell multi-tissue algorithm.

    This algorithm seems to work better for higher b-value HARDI data than for
    DTI data when it comes to GM fractions.

    Parameters
    ----------
    acquisition_scheme: Dmipy acquisition scheme,
        single shell (or whatever-shell) acquisition scheme.
    data: ND-array of shape (Nx...., NDWI),
        the fitted that is to be fitted.
    tissue_responses: list of Dmipy tissue response models,
        assumed to be in the same order as they are generated using the
        dhollander16 tissue response estimation, i.e. [wm, gm, csf].
        if not given it is estimated from the data using the Dhollander16
        heuristic three-tissue estimation algorithm [2]_.
    N_iterations: positive integer,
        number of biconvex optimization steps to do in the algorithm.
        default is 4 according to [1]_.
    return_all_csd_fits: bool,
        whether to return all csd fits of all iterations or just the last one.
    csd_fit_verbose: bool,
        whether to suppress csd fitting prints or not.
    ss3t_verbose: bool,
        whether to suppress the single-shell three tissue prints or not.

    Returns
    -------
    mt_csd_fits or mt_csd_fit: list of, or singular,
        FittedMultiCompartmentSphericalHarmonicsModel.

    References
    ----------
    .. [1] Dhollander, Thijs, and Alan Connelly. "A novel iterative approach to
        reap the benefits of multi-tissue CSD from just single-shell (+ b= 0)
        diffusion MRI data." 24th International Society of Magnetic Resonance
        in Medicine 24 (2016): 3010.
    .. [2] Dhollander, T.; Raffelt, D. & Connelly, A. Unsupervised 3-tissue
        response function estimation from single-shell or multi-shell diffusion
        MR data without a co-registered T1 image. ISMRM Workshop on Breaking
        the Barriers of Diffusion MRI, 2016, 5
    """
    if tissue_responses is None:
        wm, gm, csf, selection_map = three_tissue_response_dhollander16(
            acquisition_scheme, data)
        tissue_responses = [wm, gm, csf]

    fit_args = {
        'acquisition_scheme': acquisition_scheme,
        'data': data,
        'mask': data[..., 0] > 0,
        'fit_S0_response': True,
        'verbose': csd_fit_verbose}

    if return_all_csd_fits:
        mt_csd_fits = []
    for it in range(N_iterations):
        start = time.time()
        # step one: fix WM and fit GM + CSF
        mt_csd_mod = MultiCompartmentSphericalHarmonicsModel(tissue_responses)
        if it == 0:
            mt_csd_mod.set_fixed_parameter(
                'partial_volume_0',
                np.zeros(data.shape[:-1]))
        else:
            mt_csd_mod.set_fixed_parameter(
                'partial_volume_0',
                mt_csd_fit.fitted_parameters['partial_volume_0'])  # noqa: F821
        mt_csd_fit = mt_csd_mod.fit(**fit_args)
        if return_all_csd_fits:
            mt_csd_fits.append(mt_csd_fit)

        # step two: fix CSF and fit WM + GM
        mt_csd_mod = MultiCompartmentSphericalHarmonicsModel(tissue_responses)
        mt_csd_mod.set_fixed_parameter(
            'partial_volume_2',
            mt_csd_fit.fitted_parameters['partial_volume_2'])
        mt_csd_fit = mt_csd_mod.fit(**fit_args)
        if return_all_csd_fits:
            mt_csd_fits.append(mt_csd_fit)
        computation_time = time.time() - start
        if ss3t_verbose:
            print('finish it {} of {} in {} seconds'.format(
                it + 1, N_iterations, int(computation_time)))
    if return_all_csd_fits:
        return mt_csd_fits
    else:
        return mt_csd_fit
