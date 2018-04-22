import numpy as np

__all__ = [
    'construct_model_based_A_matrix'
]


def construct_model_based_A_matrix(acquisition_scheme, model, lmax):
    """Constructs the multi-shell observation matrix from spherical_harmonics
    to DWIs. Follows the notation of Eq. (2) in [1]_.

    The dmipy acquisition_scheme object contains all the information on which
    DWIs belong to which acquisition shells, what are the maximum spherical
    harmonics order used for each shell, and the observation matrix that maps
    the DWIs of each shell to a spherical harmonics representation.

    The dmipy model must be have all parameters fixed to be able to generate
    the rotational harmonics of each shell.

    Parameters
    ----------
    acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dmipy.
    model: dmipy signal model,
        dmipy model with all parameters fixed.
    lmax: even positive integer,
        even maximum spherical harmonics order of the to-be-estimated FOD.

    Returns
    -------
    Ams: array of size (N_DWIs, N_sh_coef),
        observation matrix to map spherical harmonics to DWIs.

    References
    ----------
    .. [1] Jeurissen, Ben, et al. "Multi-tissue constrained spherical
        deconvolution for improved analysis of multi-shell diffusion MRI data."
        NeuroImage 103 (2014): 411-426.
    """
    Ncoef = int((lmax + 2) * (lmax + 1) // 2)
    Ams = np.zeros([acquisition_scheme.number_of_measurements, Ncoef])
    Ams[acquisition_scheme.b0_mask, 0] = 2 * np.sqrt(np.pi)

    model_rh = model.rotational_harmonics_representation(acquisition_scheme)

    sh_eigenvalues = np.zeros([len(model_rh), Ncoef])

    # prepare the rotational harmonics of the kernel
    counter = 0
    for n_ in range(0, lmax + 1, 2):
        coef_in_order = 2 * n_ + 1
        sh_eigenvalues[:, counter: counter + coef_in_order] = (
            np.sqrt((4 * np.pi) / (2 * n_ + 1)) *   # sh eigenvalues
            model_rh[:, n_ // 2: n_ // 2 + 1])

        counter += coef_in_order

    # construct the multi-shell observation matrix.
    for i, shell_index in enumerate(acquisition_scheme.unique_dwi_indices):
        shell_mask = acquisition_scheme.shell_indices == shell_index
        shell_sh_matrix = acquisition_scheme.shell_sh_matrices[shell_index]
        if Ncoef < shell_sh_matrix.shape[1]:
            Ams[shell_mask, :Ncoef] = np.dot(
                shell_sh_matrix[:, :Ncoef], np.diag(sh_eigenvalues[i]))
        else:
            Ams[shell_mask, :shell_sh_matrix.shape[1]] = np.dot(
                shell_sh_matrix,
                np.diag(sh_eigenvalues[i, :shell_sh_matrix.shape[1]]))
    return Ams
