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


def multi_compartment_smt(b, f_intra, lambda_par):
    """ Multi-compartment spherical mean technique from kaden et al.
    """
    # use tortuosity to get perpendicular diffusivity
    lambda_perp = T1_tortuosity(f_intra, lambda_par)
    E_mean_intra = f_intra * spherical_mean_stick(b, lambda_par)
    E_mean_extra = (1 - f_intra) * spherical_mean_zeppelin(b, lambda_par,
                                                           lambda_perp)
    E_mean = E_mean_intra + E_mean_extra
    return E_mean
