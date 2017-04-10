from dipy.reconst.shm import real_sym_sh_mrtrix, cart2sphere
from microstruktur.signal_models.three_dimensional_models import (
                                    I1_stick_rh, E4_zeppelin_rh, SD3_watson_sh)
from microstruktur.signal_models.spherical_convolution import sh_convolution
from microstruktur.signal_models.utils import T1_tortuosity
from microstruktur.signal_models.spherical_mean import (
                                spherical_mean_stick, spherical_mean_zeppelin)
import numpy as np


def noddi_watson_kaden(acquisition_params, f_intra, mu, kappa,
                       lambda_par=1.7e-3, sh_order=14):
    r""" The NODDI microstructure model [1] using using slow-exchange for the
    extra-axonal compartment according to [2], without isotropic compartment. 

    Parameters
    ----------
    acquisition_params : array, shape(N, 4),
        b-values and b-vectors of the measured acquisition scheme.
        first column is b-values in s/mm^2 and second through fourth column
        are x, y, z components of cartesian unit b-vectors.
    f_intra : float,
        intra-axonal volume fraction [0 - 1].
    mu : array, shape (3),
        Cartesian unit vector representing the axis of the Watson distribution,
        in turn describing the orientation of the estimated axon bundle.
    kappa : float,
        concentration parameter of the Watson distribution [0 - 16].
    lambda_par : float,
        parallel diffusivity in mm^2/s. Preset to 1.7e-3 according to [1].
    sh_order : int,
        maximum spherical harmonics order to be used in the approximation.
        we found 14 to be sufficient to represent concentrations of kappa=17.

    Returns
    -------
    watson_sh : array,
        spherical harmonics of Watson probability density.

    References
    ----------
    .. [1] Zhang et al.
           "NODDI: practical in vivo neurite orientation dispersion and density
            imaging of the human brain". NeuroImage (2012)
    .. [2] Kaden et al. "Multi-compartment microscopic diffusion imaging."
           NeuroImage 139 (2016): 346-359.
    """
    bvals = acquisition_params[:, 0]
    bvecs = acquisition_params[:, 1:]

    # what b-values are the different shells
    bshells = np.unique(bvals)[np.unique(bvals) > 0]

    # use tortuosity to get perpendicular diffusivity
    lambda_perp = T1_tortuosity(f_intra, lambda_par)
    # spherical harmonics of watson distribution
    sh_watson = SD3_watson_sh(mu, kappa)

    E = np.ones_like(bvals)
    for bval_ in bshells:  # for every shell
        bval_mask = bvals == bval_
        bvecs_shell = bvecs[bval_mask]  # what bvecs in that shell
        _, theta_, phi_ = cart2sphere(bvecs_shell[:, 0],
                                      bvecs_shell[:, 1],
                                      bvecs_shell[:, 2])
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]

        # rotational harmonics of stick
        rh_stick = I1_stick_rh(bval_, lambda_par)
        # rotational harmonics of zeppelin
        rh_zeppelin = E4_zeppelin_rh(bval_, lambda_par, lambda_perp)
        # rotational harmonics of one tissue micro-environment
        E_undispersed_rh = f_intra * rh_stick + (1 - f_intra) * rh_zeppelin
        # convolving micro-environment with watson distribution
        E_dispersed_sh = sh_convolution(sh_watson, E_undispersed_rh, sh_order)
        # recover signal values from watson-convolved spherical harmonics
        E[bval_mask] = np.dot(sh_mat, E_dispersed_sh)
    return E


def multi_compartment_smt(acquisition_params, f_intra, lambda_par):
    r""" Multi-compartment spherical mean technique by Kaden et al [1].
    Uses the spherical mean of the signal attenuation to estimate intra-axonal
    volume fraction and diffusivity without having to estimate dispersion or
    crossings. Requires multi-shell data.

    Parameters
    ----------
    acquisition_params : array, shape(N, 4),
        b-values and b-vectors of the measured acquisition scheme.
        first column is b-values in s/mm^2 and second through fourth column
        are x, y, z components of cartesian unit b-vectors.
        in this model only the b-values are used.
    f_intra : float,
        intra-axonal volume fraction [0 - 1].
    lambda_par : float,
        parallel diffusivity in [0 - 4e-3] mm^2/s.

    Returns
    -------
    E_mean : array, shape (number of b-shells)
        spherical means of the signal attenuation for the given f_intra and
        lambda_par at the b-values of acquisition scheme. For example, if there
        are three shells in the acquisition then the array is of length 3.

    References
    ----------
    .. [1] Kaden et al. "Multi-compartment microscopic diffusion imaging."
           NeuroImage 139 (2016): 346-359.
    """
    bvals = acquisition_params[:, 0]
    # recover b-values of the different shells of the data
    unique_bvals = np.unique(bvals)[np.unique(bvals) > 0]
    E_mean = np.zeros_like(unique_bvals)
    # use tortuosity to get perpendicular diffusivity
    lambda_perp = T1_tortuosity(f_intra, lambda_par)
    for i, b_ in enumerate(unique_bvals):
        E_mean_intra = spherical_mean_stick(b_, lambda_par)
        E_mean_extra = spherical_mean_zeppelin(b_, lambda_par, lambda_perp)
        E_mean[i] = f_intra * E_mean_intra + (1 - f_intra) * E_mean_extra
    return E_mean
