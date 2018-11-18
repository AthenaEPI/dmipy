import numpy as np
from ..utils.construct_observation_matrix import construct_model_based_A_matrix


class AnisotropicSignalModelProperties:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError()

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        kwargs.update({'mu': [0., 0.]})
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped),
                             rh_scheme.shell_sh_orders.max() // 2 + 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[sh_order],
                    E_reshaped[i])
            )
        return rh_array

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the model for every acquisition shell.
        """
        E_mean = np.ones_like(acquisition_scheme.shell_bvalues)
        rh_array = self.rotational_harmonics_representation(
            acquisition_scheme, **kwargs)
        E_mean[acquisition_scheme.unique_dwi_indices] = (
            rh_array[:, 0] / (2 * np.sqrt(np.pi))
        )
        return E_mean

    def convolution_response_kernel(self, acquisition_scheme, lmax):
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
        model_rh = self.rotational_harmonics_representation(
            acquisition_scheme, **kwargs)
        return construct_model_based_A_matrix(acquisition_scheme, model_rh, lmax)


class IsotropicSignalModelProperties:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError()

    def rotational_harmonics_representation(
            self, acquisition_scheme, **kwargs):
        r""" The rotational harmonics of the model, such that Y_lm = Yl0.
        Axis aligned with z-axis to be used as kernel for spherical
        convolution. Returns an array with rotational harmonics for each shell.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        rh_array : array, shape(Nshells, N_rh_coef),
            Rotational harmonics coefficients for each shell.
        """
        rh_scheme = acquisition_scheme.rotational_harmonics_scheme
        E_kernel_sf = self(rh_scheme, **kwargs)
        E_reshaped = E_kernel_sf.reshape([-1, rh_scheme.Nsamples])
        rh_array = np.zeros((len(E_reshaped), 1))

        for i, sh_order in enumerate(rh_scheme.shell_sh_orders):
            rh_array[i, :sh_order // 2 + 1] = (
                np.dot(
                    rh_scheme.inverse_rh_matrix[0],
                    E_reshaped[i])
            )
        return rh_array

    def spherical_mean(self, acquisition_scheme, **kwargs):
        """
        Estimates spherical mean for every shell in acquisition scheme.

        Parameters
        ----------
        acquisition_scheme : DmipyAcquisitionScheme instance,
            An acquisition scheme that has been instantiated using dMipy.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        E_mean : float,
            spherical mean of the model for every acquisition shell.
        """
        return self(acquisition_scheme.spherical_mean_scheme, **kwargs)

    def convolution_response_kernel(self, acquisition_scheme, lmax, **kwargs):
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

        model_rh = self.rotational_harmonics_representation(
            acquisition_scheme, **kwargs)

        Ncoef = int((lmax + 2) * (lmax + 1) // 2)
        Ams = np.zeros([acquisition_scheme.number_of_measurements, Ncoef])

        Ams[:, 0] = construct_model_based_A_matrix(
            acquisition_scheme, model_rh, 0)
        return Ams
