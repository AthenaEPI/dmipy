import numpy as np
from . import three_dimensional_models
from . import utils
from microstruktur.signal_models.spherical_convolution import sh_convolution
from dipy.reconst.shm import real_sym_sh_mrtrix
MicrostrukturModel = three_dimensional_models.MicrostrukturModel
WATSON_SH_ORDER = 14

class SD3I2WatsonDispersedSodermanCylinder(MicrostrukturModel):
    r""" The Watson-Dispersed Soderman cylinder model - assuming limits of
    pulse separation towards infinity and pulse duration towards zero - for
    intra-axonal diffusion.

    Parameters
    ----------
    bvals : array, shape(N),
        b-values in s/m^2.
    n : array, shape(N x 3),
        b-vectors in cartesian coordinates.
    delta : array, shape(N),
        pulse duration in seconds.
    Delta : array, shape(N),
        pulse separation in seconds
    mu : array, shape(3),
        unit vector representing orientation of the Stick.
    lambda_par : float,
        parallel diffusivity in 10^9m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, np.inf),
        'kappa': (0, 16),
        'diameter': (0, np.inf)
    }

    def __init__(self, mu=None, lambda_par=None, kappa=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.kappa = kappa
        self.diameter = diameter

    def __call__(self, bvals, n, delta=None, Delta=None, shell_indices=None,
                 **kwargs):
        r'''
        Parameters
        ----------
        bvals : array, shape(N),
            b-values in s/mm^2.
        n : array, shape(N x 3),
            b-vectors in cartesian coordinates.
        delta : array, shape(N),
            pulse duration in seconds.
        Delta : array, shape(N),
            pulse separation in seconds
        shell_indices : array, shape(N),
            array with integers reflecting to which acquisition shell a
            measurement belongs. zero for b0 measurements, 1 for the first
            shell, 2 for the second, etc.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        sh_order = WATSON_SH_ORDER
        if (
            delta is None or Delta is None
        ):
            raise ValueError('This class needs non-None delta and Delta')
        if shell_indices is None:
            msg = "This class needs shell_indices"
            raise ValueError(msg)
        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)

        watson = three_dimensional_models.SD3Watson(mu=mu, kappa=kappa)
        sh_watson = watson.spherical_harmonics_representation()
        soderman = three_dimensional_models.I2CylinderSodermanApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones_like(bvals)
        for shell_index in np.arange(1, shell_indices.max() + 1):  # per shell
            bval_mask = shell_indices == shell_index
            bvecs_shell = n[bval_mask]  # what bvecs in that shell
            bval_mean = bvals[bval_mask].mean()
            delta_mean = delta[bval_mask].mean()
            Delta_mean = Delta[bval_mask].mean()
            _, theta_, phi_ = utils.cart2sphere(bvecs_shell).T
            sh_mat = real_sym_sh_mrtrix(sh_order, theta_, phi_)[0]

            # rotational harmonics of stick
            rh_stick = soderman.rotational_harmonics_representation(
                bval=bval_mean, delta=delta_mean, Delta=Delta_mean)
            # convolving micro-environment with watson distribution
            E_dispersed_sh = sh_convolution(sh_watson, rh_stick, sh_order)
            # recover signal values from watson-convolved spherical harmonics
            E[bval_mask] = np.dot(sh_mat, E_dispersed_sh)
        return E