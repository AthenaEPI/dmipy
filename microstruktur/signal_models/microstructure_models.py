from dipy.reconst.shm import real_sym_sh_mrtrix, cart2sphere, sphere2cart
from microstruktur.signal_models.three_dimensional_models import (
                I1_stick, I1_stick_rh, E3_ball, E4_zeppelin_rh, SD3_watson_sh)
from microstruktur.signal_models.spherical_convolution import sh_convolution
from microstruktur.signal_models.utils import T1_tortuosity
from microstruktur.signal_models.spherical_mean import (
                                spherical_mean_stick, spherical_mean_zeppelin)
from .constants import CONSTANTS
import numpy as np

# instra-axonal diffusion constant in mm^2/s
intra_axonal_diffusivity = CONSTANTS['water_in_axons_diffusion_constant'] * 1e6


def ball_and_stick_spherical(bvals, bvecs, f_intra, theta, phi, lambda_par,
                         lambda_iso):
    r"""Wrapper function for the Cartesian ball_and_stick model. This function
    converts the euler angles theta [0, pi] and theta [-pi, pi] to [x, y, z]
    and returns the Cartesian ball_and_stick function.
    """
    x, y, z = sphere2cart(theta, phi)
    mu = np.r_[x, y, z]
    return ball_and_stick(bvals, bvecs, f_intra, mu, lambda_par, lambda_iso)


def ball_and_stick(bvals, bvecs, f_intra, mu, lambda_par, lambda_iso):
    r""" The Ball and Stick model [1], consisting of a Stick model oriented
    along mu with parallel diffusivity lambda_par, and an isotropic Ball
    compartment with isotropic diffusivity lambda_iso.

    Parameters
    ----------
    bvals : array, shape (N),
        b-values in s/mm^2.
    bvecs : array, shape (N x 3),
        x, y, z components of cartesian unit b-vectors.
    f_intra : float,
        intra-axonal volume fraction [0 - 1].
    mu : array, shape (3),
        Cartesian unit vector representing the axis of the Stick model.
    lambda_par : float,
        parallel diffusivity in mm^2/s. Preset to 1.7e-3 according to [1].
    lambda_iso : float,
        isotropic diffusivity in mm^2/s.

    Returns
    -------
    E_ball_and_stick : array, shape (N),
        signal attenuation of the ball and stick model for given parameters.

    References
    ----------
    .. [1] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """
    E_stick = I1_stick(bvals, bvecs, mu, lambda_par)
    E_ball = E3_ball(bvals, lambda_iso)
    E_ball_and_stick = f_intra * E_stick + (1 - f_intra) * E_ball
    return E_ball_and_stick


#def ball_and_racket(bvals, bvecs, b_shell_indices, f_intra, mu, lambda_par,
#                    lambda_iso, kappa, beta):
#    r""" The Ball and Rackets model [1], consisting of a Stick model with
#    parallel diffusivity lambda_par - distributed by a Bingham distribution
#    oriented along mu with concentration parameters kappa and beta - and an
#    isotropic Ball compartment with isotropic diffusivity lambda_iso.
#
#    Parameters
#    ----------
#    bvals : array, shape (N),
#        b-values in s/mm^2.
#    bvecs : array, shape (N x 3),
#        x, y, z components of cartesian unit b-vectors.
#    b_shell_indices : array, shape (N)
#        array of integers indicating which measurement belongs to which shell.
#        0 should be used for b0 measurements, 1 for the lowest b-value shell,
#        2 for the second lowest etc.
#    f_intra : float,
#        intra-axonal volume fraction [0 - 1].
#    mu : array, shape (3),
#        Cartesian unit vector representing the axis of the Watson distribution,
#        in turn describing the orientation of the estimated axon bundle.
#    lambda_par : float,
#        parallel diffusivity in mm^2/s. Preset to 1.7e-3 according to [1].
#    lambda_iso : float,
#        isotropic diffusivity in mm^2/s.
#    kappa : float,
#        first concentration parameter of the Bingham distribution.
#        defined as kappa = kappa1 - kappa3.
#    beta : float,
#        second concentration parameter of the Bingham distribution.
#        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.
#
#    Returns
#    -------
#    E_ball_and_racket : array, shape (N),
#        signal attenuation of the ball and racket model for given parameters.
#
#    References
#    ----------
#    .. [1] Sotiropoulos et al. "Ball and rackets: inferring fiber fanning from
#        diffusion-weighted MRI." NeuroImage 60.2 (2012): 1412-1425.
#    """
#    return


#def noddi_watson_zhang(bvals, bvecs, b_shell_indices, f_intra, mu, kappa,
#                       lambda_par=1.7e-3, sh_order=14):
#    r""" The NODDI microstructure model [1] using using fast-exchange for the
#    extra-axonal compartment as in the original paper, without isotropic
#    compartment.
#
#    Parameters
#    ----------
#    bvals : array, shape (N),
#        b-values in s/mm^2.
#    bvecs : array, shape (N x 3),
#        x, y, z components of cartesian unit b-vectors.
#    b_shell_indices : array, shape (N)
#        array of integers indicating which measurement belongs to which shell.
#        0 should be used for b0 measurements, 1 for the lowest b-value shell,
#        2 for the second lowest etc.
#    f_intra : float,
#        intra-axonal volume fraction [0 - 1].
#    mu : array, shape (3),
#        Cartesian unit vector representing the axis of the Watson distribution,
#        in turn describing the orientation of the estimated axon bundle.
#    kappa : float,
#        concentration parameter of the Watson distribution [0 - 16].
#    lambda_par : float,
#        parallel diffusivity in mm^2/s. Preset to 1.7e-3 according to [1].
#    sh_order : int,
#        maximum spherical harmonics order to be used in the approximation.
#        we found 14 to be sufficient to represent concentrations of kappa=17.
#
#    Returns
#    -------
#    E : array,
#        signal attenuation of the noddi Watson model with fast exchange.
#
#    References
#    ----------
#    .. [1] Zhang et al.
#           "NODDI: practical in vivo neurite orientation dispersion and density
#            imaging of the human brain". NeuroImage (2012)
#    """
#
#    # use tortuosity to get perpendicular diffusivity
#    lambda_perp = T1_tortuosity(f_intra, lambda_par)
#    # spherical harmonics of watson distribution
#    sh_watson = SD3_watson_sh(mu, kappa)
#
#    E = np.ones_like(bvals)
#    for bval_ in bshells:  # for every shell
#        bval_mask = bvals == bval_
#        bvecs_shell = bvecs[bval_mask]  # what bvecs in that shell
#        _, theta_, phi_ = cart2sphere(bvecs_shell[:, 0],
#                                      bvecs_shell[:, 1],
#                                      bvecs_shell[:, 2])
#        sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]
#
#        # rotational harmonics of stick
#        rh_stick = I1_stick_rh(bval_, lambda_par)
#        # rotational harmonics of one tissue micro-environment
#        stick_undispersed_rh = rh_stick
#        # convolving micro-environment with watson distribution
#        stick_dispersed_sh = sh_convolution(sh_watson, stick_undispersed_rh,
#                                            sh_order)
#        E_stick_dispersed = np.dot(sh_mat, stick_dispersed_sh)
#        #E_zeppelin_dispersed = E4_zeppelin_zhang()
#        # recover signal values from watson-convolved spherical harmonics
#        E[bval_mask] = (f_intra * E_stick_dispersed +
#                        (1 - f_intra) * E_zeppelin_dispersed)
#    return E


def noddi_watson_kaden_spherical(bvals, bvecs, b_shell_indices, f_intra, theta,
                                 phi, kappa,
                                 lambda_par=intra_axonal_diffusivity,
                                 sh_order=14):
    r"""Wrapper function for the Cartesian noddi_watson_kaden model. This
    function converts the euler angles theta [0, pi] and theta [-pi, pi] to
    [x, y, z] and returns the Cartesian noddi_watson_kaden function.
    """
    x, y, z = sphere2cart(theta, phi)
    mu = np.r_[x, y, z]
    return noddi_watson_kaden(bvals, bvecs, b_shell_indices, f_intra, mu, kappa,
                              lambda_par, sh_order)


def noddi_watson_kaden(bvals, bvecs, b_shell_indices, f_intra, mu, kappa,
                       lambda_par=intra_axonal_diffusivity, sh_order=14):
    r""" The NODDI microstructure model [1] using using slow-exchange for the
    extra-axonal compartment according to [2], without isotropic compartment.

    Parameters
    ----------
    bvals : array, shape (N),
        b-values in s/mm^2.
    bvecs : array, shape (N x 3),
        x, y, z components of cartesian unit b-vectors.
    b_shell_indices : array, shape (N)
        array of integers indicating which measurement belongs to which shell.
        0 should be used for b0 measurements, 1 for the lowest b-value shell,
        2 for the second lowest etc.
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
    E : array,
        signal attenuation of the NODDI Watson model with slow exchange.

    References
    ----------
    .. [1] Zhang et al.
           "NODDI: practical in vivo neurite orientation dispersion and density
            imaging of the human brain". NeuroImage (2012)
    .. [2] Kaden et al. "Multi-compartment microscopic diffusion imaging."
           NeuroImage 139 (2016): 346-359.
    """
    # use tortuosity to get perpendicular diffusivity
    lambda_perp = T1_tortuosity(f_intra, lambda_par)
    # spherical harmonics of watson distribution
    sh_watson = SD3_watson_sh(mu, kappa)

    E = np.ones_like(bvals)
    for b_shell_index in np.arange(1, b_shell_indices.max() + 1):  # per shell
        bval_mask = b_shell_indices == b_shell_index
        bvecs_shell = bvecs[bval_mask]  # what bvecs in that shell
        bval_mean = bvals[bval_mask].mean()
        _, theta_, phi_ = cart2sphere(bvecs_shell[:, 0],
                                      bvecs_shell[:, 1],
                                      bvecs_shell[:, 2])
        sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]

        # rotational harmonics of stick
        rh_stick = I1_stick_rh(bval_mean, lambda_par)
        # rotational harmonics of zeppelin
        rh_zeppelin = E4_zeppelin_rh(bval_mean, lambda_par, lambda_perp)
        # rotational harmonics of one tissue micro-environment
        E_undispersed_rh = f_intra * rh_stick + (1 - f_intra) * rh_zeppelin
        # convolving micro-environment with watson distribution
        E_dispersed_sh = sh_convolution(sh_watson, E_undispersed_rh, sh_order)
        # recover signal values from watson-convolved spherical harmonics
        E[bval_mask] = np.dot(sh_mat, E_dispersed_sh)
    return E


def multi_compartment_smt(bvals, b_shell_indices, f_intra, lambda_par):
    r""" Multi-compartment spherical mean technique by Kaden et al [1].
    Uses the spherical mean of the signal attenuation to estimate intra-axonal
    volume fraction and diffusivity without having to estimate dispersion or
    crossings. Requires multi-shell data.

    Parameters
    ----------
    bvals : array, shape (N),
        b-values in s/mm^2.
    b_shell_indices : array, shape (N)
        array of integers indicating which measurement belongs to which shell.
        0 should be used for b0 measurements, 1 for the lowest b-value shell,
        2 for the second lowest etc.
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
    b_mean = np.zeros(b_shell_indices.max())
    for b_shell_index in np.arange(1, b_shell_indices.max() + 1):  # per shell
        bvals_shell = bvals[b_shell_indices == b_shell_index]
        b_mean[b_shell_index - 1] = bvals_shell.mean()
    
    # use tortuosity to get perpendicular diffusivity
    lambda_perp = T1_tortuosity(f_intra, lambda_par)
    E_mean_intra = spherical_mean_stick(b_mean, lambda_par)
    E_mean_extra = spherical_mean_zeppelin(b_mean, lambda_par, lambda_perp)
    E_mean = f_intra * E_mean_intra + (1 - f_intra) * E_mean_extra
    return E_mean


#def activeax(acquisition_params):
#    return
#
#def axcaliber_assaf(acquisition_params, f_intra, lambda_extra, alpha, beta):  # also needs choice for cylinder
#    return
#
#
#def axcaliber_burcaw(acquisition_params, f_intra, lambda_extra, alpha, beta,
#                                                                           A):
#    return
