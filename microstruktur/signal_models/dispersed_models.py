# -*- coding: utf-8 -*-
import numpy as np
from . import gaussian_models, cylinder_models, distributions
from microstruktur.utils.spherical_convolution import sh_convolution
from scipy import stats
from microstruktur.core import modeling_framework
MicrostructureModel = modeling_framework.MicrostructureModel
DIFFUSIVITY_SCALING = 1e-9
DIAMETER_SCALING = 1e-6
BETA_SCALING = 1e-6
A_SCALING = 1e-12


class SD2C1BinghamDispersedStick(MicrostructureModel):
    r""" The Bingham-Dispersed [1] Stick model [2] - a cylinder with zero
    radius - for intra-axonal diffusion. Allows for anisotropic dispersion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.

    References
    ----------
    .. [2] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, 3),
        'psi': (0, np.pi),
        'kappa': (0, 16),
        'beta': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'psi': 1.,
        'kappa': 1.,
        'beta': 1.
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None,
                 kappa=None, beta=None, psi=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.psi = psi
        self.kappa = kappa
        self.beta = beta

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)

        bingham = distributions.SD2Bingham(mu=mu, psi=psi,
                                           kappa=kappa, beta=beta)
        sh_bingham = bingham.spherical_harmonics_representation()
        stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = stick.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with bingham distribution
            E_dispersed_sh = sh_convolution(sh_bingham, rh_stick)
            # recover signal values from bingham-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat, E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        bingham = distributions.SD2Bingham(
            mu=mu, psi=psi, kappa=kappa, beta=beta)
        bingham_density = bingham(vertices)
        return bingham_density


class SD2C2BinghamDispersedSodermanCylinder(MicrostructureModel):
    r""" The Bingham-Dispersed [1] Soderman cylinder model [2] - assuming
    limits of pulse separation towards infinity and pulse duration towards zero
    - for intra-axonal diffusion. Allows for anisotropic dispersion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Söderman, Olle, and Bengt Jönsson. "Restricted diffusion in
        cylindrical geometry." Journal of Magnetic Resonance, Series A
        117.1 (1995): 94-97.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, 3),
        'diameter': (1e-10, 50e-6),
        'psi': (0, np.pi),
        'kappa': (0, 16),
        'beta': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
        'psi': 1.,
        'kappa': 1.,
        'beta': 1.
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, diameter=None,
                 kappa=None, beta=None, psi=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter
        self.psi = psi
        self.kappa = kappa
        self.beta = beta

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation.
        """
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)

        bingham = distributions.SD2Bingham(mu=mu, psi=psi,
                                           kappa=kappa, beta=beta)
        sh_bingham = bingham.spherical_harmonics_representation()
        soderman = cylinder_models.C2CylinderSodermanApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = soderman.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                delta=acquisition_scheme.shell_delta[shell_index],
                Delta=acquisition_scheme.shell_Delta[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with bingham distribution
            E_dispersed_sh = sh_convolution(sh_bingham, rh_stick)
            # recover signal values from bingham-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat, E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        bingham = distributions.SD2Bingham(
            mu=mu, psi=psi, kappa=kappa, beta=beta)
        bingham_density = bingham(vertices)
        return bingham_density


class SD2C3BinghamDispersedCallaghanCylinder(MicrostructureModel):
    r""" The Bingham-Dispersed [1] Callaghan cylinder model [2] - assuming
    finite pulse separation and limit of pulse duration towards zero - for
    intra-axonal diffusion. Allows for anisotropic dispersion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
        cylindrical, and spherical pores under conditions of wall
        relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
        53-59.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, 3),
        'diameter': (1e-10, 50e-6),
        'psi': (0, np.pi),
        'kappa': (0, 16),
        'beta': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
        'psi': 1.,
        'kappa': 1.,
        'beta': 1.
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, diameter=None,
                 kappa=None, beta=None, psi=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter
        self.psi = psi
        self.kappa = kappa
        self.beta = beta

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation.
        """
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)

        bingham = distributions.SD2Bingham(mu=mu, psi=psi,
                                           kappa=kappa, beta=beta)
        sh_bingham = bingham.spherical_harmonics_representation()
        callaghan = cylinder_models.C3CylinderCallaghanApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = callaghan.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                delta=acquisition_scheme.shell_delta[shell_index],
                Delta=acquisition_scheme.shell_Delta[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with bingham distribution
            E_dispersed_sh = sh_convolution(sh_bingham, rh_stick)
            # recover signal values from bingham-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat, E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        bingham = distributions.SD2Bingham(
            mu=mu, psi=psi, kappa=kappa, beta=beta)
        bingham_density = bingham(vertices)
        return bingham_density


class SD2C4BinghamDispersedGaussianPhaseCylinder(MicrostructureModel):
    r""" The Bingham-Dispersed [1] van gelderen cylinder model [2] - assuming
    finite pulse separation and pulse duration - for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [1] Van Gelderen et al.
        "Evaluation of Restricted Diffusion
        in Cylinders. Phosphocreatine in Rabbit Leg Muscle"
        Journal of Magnetic Resonance Series B (1994)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, 3),
        'diameter': (1e-10, 50e-6),
        'psi': (0, np.pi),
        'kappa': (0, 16),
        'beta': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING,
        'psi': 1.,
        'kappa': 1.,
        'beta': 1.
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, diameter=None,
                 kappa=None, beta=None, psi=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter
        self.psi = psi
        self.kappa = kappa
        self.beta = beta

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation.
        """
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)

        bingham = distributions.SD2Bingham(mu=mu, psi=psi,
                                           kappa=kappa, beta=beta)
        sh_bingham = bingham.spherical_harmonics_representation()
        vg = cylinder_models.C4CylinderGaussianPhaseApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = vg.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                delta=acquisition_scheme.shell_delta[shell_index],
                Delta=acquisition_scheme.shell_Delta[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with bingham distribution
            E_dispersed_sh = sh_convolution(sh_bingham, rh_stick)
            # recover signal values from bingham-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat, E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        bingham = distributions.SD2Bingham(
            mu=mu, psi=psi, kappa=kappa, beta=beta)
        bingham_density = bingham(vertices)
        return bingham_density


class SD1C1WatsonDispersedStick(MicrostructureModel):
    r""" The Watson-Dispersed [1] Stick model [2] - a cylinder with zero radius
    - for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Behrens et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
           Magnetic Resonance in Medicine (2003)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'kappa': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'kappa': 1.,
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, kappa=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.kappa = kappa

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)

        watson = distributions.SD1Watson(mu=mu, kappa=kappa)
        sh_watson = watson.spherical_harmonics_representation()
        stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = stick.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with watson distribution
            E_dispersed_sh = sh_convolution(sh_watson, rh_stick)
            # recover signal values from watson-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)
        watson = distributions.SD1Watson(
            mu=mu, kappa=kappa)
        watson_density = watson(vertices)
        return watson_density


class SD1C2WatsonDispersedSodermanCylinder(MicrostructureModel):
    r""" The Watson-Dispersed [1] Soderman cylinder model [2] - assuming limits
    of pulse separation towards infinity and pulse duration towards zero - for
    intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi]..
    lambda_par : float,
        parallel diffusivity in 10^9m^2/s.
    kappa : float,
        concentration parameter of Watson distribution.
    diameter : float,
        cylinder (axon) diameter in meters.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Söderman, Olle, and Bengt Jönsson. "Restricted diffusion in
        cylindrical geometry." Journal of Magnetic Resonance, Series A
        117.1 (1995): 94-97.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'kappa': (0, 16),
        'diameter': (1e-10, 50e-6),
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'kappa': 1.,
        'diameter': DIAMETER_SCALING,
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, kappa=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.kappa = kappa
        self.diameter = diameter

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation.
        """
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)

        watson = distributions.SD1Watson(mu=mu, kappa=kappa)
        sh_watson = watson.spherical_harmonics_representation()
        soderman = cylinder_models.C2CylinderSodermanApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = soderman.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                delta=acquisition_scheme.shell_delta[shell_index],
                Delta=acquisition_scheme.shell_Delta[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with watson distribution
            E_dispersed_sh = sh_convolution(sh_watson, rh_stick)
            # recover signal values from watson-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)
        watson = distributions.SD1Watson(
            mu=mu, kappa=kappa)
        watson_density = watson(vertices)
        return watson_density


class SD1C3WatsonDispersedCallaghanCylinder(MicrostructureModel):
    r""" The Watson-Dispersed [1] Callaghan cylinder model [2] - assuming
    finite pulse separation and the limit of pulse duration towards zero - for
    intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    kappa : float,
        concentration parameter of Watson distribution.
    diameter : float,
        cylinder (axon) diameter in meters.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [2] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
        cylindrical, and spherical pores under conditions of wall
        relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
        53-59.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'kappa': (0, 16),
        'diameter': (1e-10, 50e-6),
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'kappa': 1.,
        'diameter': DIAMETER_SCALING,
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, kappa=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.kappa = kappa
        self.diameter = diameter

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation.
        """
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)

        watson = distributions.SD1Watson(mu=mu, kappa=kappa)
        sh_watson = watson.spherical_harmonics_representation()
        callaghan = cylinder_models.C3CylinderCallaghanApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = callaghan.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                delta=acquisition_scheme.shell_delta[shell_index],
                Delta=acquisition_scheme.shell_Delta[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with watson distribution
            E_dispersed_sh = sh_convolution(sh_watson, rh_stick)
            # recover signal values from watson-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)
        watson = distributions.SD1Watson(
            mu=mu, kappa=kappa)
        watson_density = watson(vertices)
        return watson_density


class SD1C4WatsonDispersedGaussianPhaseCylinder(MicrostructureModel):
    r""" The Watson-Dispersed [1] Van Gelderen cylinder model [2] - assuming
    finite pulse separation and pulse duration - for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    kappa : float,
        concentration parameter of Watson distribution.
    diameter : float,
        cylinder (axon) diameter in meters.

    References
    ----------
    .. [1] Kaden et al.
        "Parametric spherical deconvolution: inferring anatomical
        connectivity using diffusion MR imaging". NeuroImage (2007)
    .. [1] Van Gelderen et al.
        "Evaluation of Restricted Diffusion
        in Cylinders. Phosphocreatine in Rabbit Leg Muscle"
        Journal of Magnetic Resonance Series B (1994)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'kappa': (0, 16),
        'diameter': (1e-10, 50e-6),
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'kappa': 1.,
        'diameter': DIAMETER_SCALING,
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, kappa=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.kappa = kappa
        self.diameter = diameter

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation
        """
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)

        watson = distributions.SD1Watson(mu=mu, kappa=kappa)
        sh_watson = watson.spherical_harmonics_representation()
        vg = cylinder_models.C4CylinderGaussianPhaseApproximation(
            mu=mu, lambda_par=lambda_par, diameter=diameter
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of stick
            rh_stick = vg.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                delta=acquisition_scheme.shell_delta[shell_index],
                Delta=acquisition_scheme.shell_Delta[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with watson distribution
            E_dispersed_sh = sh_convolution(sh_watson, rh_stick)
            # recover signal values from watson-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)
        watson = distributions.SD1Watson(
            mu=mu, kappa=kappa)
        watson_density = watson(vertices)
        return watson_density


class SD2G4BinghamDispersedZeppelin(MicrostructureModel):
    r""" The Bingham-Dispersed [1] Zeppelin model [2] - for typically
    extra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    lambda_perp : float,
        perpendicular diffusivity in 10^9 m^2/s.
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].
    kappa : float,
        first concentration parameter of the Bingham distribution.
        defined as kappa = kappa1 - kappa3.
    beta : float,
        second concentration parameter of the Bingham distribution.
        defined as beta = kappa2 - kappa3. Bingham becomes Watson when beta=0.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (0, 3),
        'lambda_perp': (0, 3),
        'psi': (0, np.pi),
        'kappa': (0, 16),
        'beta': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING,
        'psi': 1.,
        'kappa': 1.,
        'beta': 1.
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None,
                 kappa=None, beta=None, psi=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp
        self.psi = psi
        self.kappa = kappa
        self.beta = beta

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        attenuation : float or array, shape(N),
            signal attenuation
        '''
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        psi = kwargs.get('psi', self.psi)

        bingham = distributions.SD2Bingham(mu=mu, kappa=kappa,
                                           beta=beta, psi=psi)
        sh_bingham = bingham.spherical_harmonics_representation()
        zeppelin = gaussian_models.G4Zeppelin(
            mu=mu, lambda_par=lambda_par, lambda_perp=lambda_perp
        )

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of zeppelin
            rh_zeppelin = zeppelin.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with bingham distribution
            E_dispersed_sh = sh_convolution(sh_bingham, rh_zeppelin)
            # recover signal values from bingham-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        psi = kwargs.get('psi', self.psi)
        kappa = kwargs.get('kappa', self.kappa)
        beta = kwargs.get('beta', self.beta)
        bingham = distributions.SD2Bingham(
            mu=mu, psi=psi, kappa=kappa, beta=beta)
        bingham_density = bingham(vertices)
        return bingham_density


class SD1G4WatsonDispersedZeppelin(MicrostructureModel):
    r""" The Watson-Dispersed Zeppelin model [1] - a cylinder with zero radius-
    for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    kappa : float,
        concentration parameter of Watson distribution.
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    lambda_perp : float,
        perpendicular diffusivity in 10^9 m^2/s.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """
    spherical_mean = False

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'lambda_perp': (.1, 3),
        'kappa': (0, 16)
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'lambda_perp': DIFFUSIVITY_SCALING,
        'kappa': 1.,
    }

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None, kappa=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp
        self.kappa = kappa

    def __call__(self, acquisition_scheme, **kwargs):
        r'''
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : float or array, shape(N),
            signal attenuation
        '''
        shell_indices = acquisition_scheme.shell_indices
        unique_dwi_indices = acquisition_scheme.unique_dwi_indices

        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)

        watson = distributions.SD1Watson(mu=mu, kappa=kappa)
        sh_watson = watson.spherical_harmonics_representation()
        zeppelin = gaussian_models.G4Zeppelin(mu=mu,
                                              lambda_par=lambda_par,
                                              lambda_perp=lambda_perp)

        E = np.ones(acquisition_scheme.number_of_measurements)
        for shell_index in unique_dwi_indices:  # per shell
            shell_mask = shell_indices == shell_index
            sh_mat = acquisition_scheme.shell_sh_matrices[shell_index]
            # rotational harmonics of zeppelin
            rh_zeppelin = zeppelin.rotational_harmonics_representation(
                bvalue=acquisition_scheme.shell_bvalues[shell_index],
                rh_order=acquisition_scheme.shell_sh_orders[shell_index])
            # convolving micro-environment with watson distribution
            E_dispersed_sh = sh_convolution(sh_watson, rh_zeppelin)
            # recover signal values from watson-convolved spherical harmonics
            E[shell_mask] = np.dot(sh_mat[:, :len(E_dispersed_sh)],
                                   E_dispersed_sh)
        return E

    def fod(self, vertices, **kwargs):
        """
        Estimates Fiber Orientation Density (FOD) of the model.

        Parameters
        ----------
        vertices: array of shape (N, 3)
            cartesian unit vectors at which to evaluate the fiber orientation
            density (FOD).

        Returns
        -------
        bingham_density: array of shape (N,)
            probability density of distribution for model parameters
        """
        mu = kwargs.get('mu', self.mu)
        kappa = kwargs.get('kappa', self.kappa)
        watson = distributions.SD1Watson(
            mu=mu, kappa=kappa)
        watson_density = watson(vertices)
        return watson_density


class DD1C2GammaDistributedSodermanCylinder(MicrostructureModel):
    r""" The Gamma-Distributed Soderman cylinder model [1] - assuming
    limits of pulse separation towards infinity and pulse duration towards zero
    - for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    alpha : float,
        shape of the gamma distribution.
    beta : float,
        scale of the gamma distrubution. Different from Bingham distribution!

    References
    ----------
    .. [1] Söderman, Olle, and Bengt Jönsson. "Restricted diffusion in
        cylindrical geometry." Journal of Magnetic Resonance, Series A
        117.1 (1995): 94-97.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'alpha': (1e-10, np.inf),
        'beta': (1e-10, np.inf),
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'alpha': 1.,
        'beta': BETA_SCALING
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, alpha=None, beta=None,
                 radius_integral_steps=35):
        self.mu = mu
        self.lambda_par = lambda_par
        self.alpha = alpha
        self.beta = beta
        self.radius_integral_steps = radius_integral_steps

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)

        gamma_dist = stats.gamma(alpha, scale=beta)
        radius_max = gamma_dist.mean() + 6 * gamma_dist.std()
        radii = np.linspace(1e-50, radius_max, self.radius_integral_steps)
        area = np.pi * radii ** 2
        radii_pdf = gamma_dist.pdf(radii)
        radii_pdf_area = radii_pdf * area
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapz(x=radii, y=radii_pdf_area)
        )

        soderman = cylinder_models.C2CylinderSodermanApproximation(
            mu=mu, lambda_par=lambda_par
        )

        E = np.empty(
            (self.radius_integral_steps,
             acquisition_scheme.number_of_measurements))
        for i, radius in enumerate(radii):
            E[i] = (
                radii_pdf_normalized[i] *
                soderman(acquisition_scheme, diameter=radius * 2)
            )

        E = np.trapz(E, x=radii, axis=0)
        return E


class DD1C3GammaDistributedCallaghanCylinder(MicrostructureModel):
    r""" Gamma-distributed Callaghan cylinder model [1] - assuming
    finite pulse separation and the limit of pulse duration towards zero - for
    intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    alpha : float,
        shape of the gamma distribution.
    beta : float,
        scale of the gamma distrubution. Different from Bingham distribution!

    References
    ----------
    .. [1] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
        cylindrical, and spherical pores under conditions of wall
        relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
        53-59.
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'alpha': (1e-10, np.inf),
        'beta': (1e-10, np.inf),
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'alpha': 1.,
        'beta': BETA_SCALING
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, alpha=None, beta=None,
                 radius_integral_steps=35):
        self.mu = mu
        self.lambda_par = lambda_par
        self.alpha = alpha
        self.beta = beta
        self.radius_integral_steps = radius_integral_steps

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)

        gamma_dist = stats.gamma(alpha, scale=beta)
        radius_max = gamma_dist.mean() + 6 * gamma_dist.std()
        radii = np.linspace(1e-50, radius_max, self.radius_integral_steps)
        area = np.pi * radii ** 2
        radii_pdf = gamma_dist.pdf(radii)
        radii_pdf_area = radii_pdf * area
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapz(x=radii, y=radii_pdf_area)
        )

        callaghan = cylinder_models.C3CylinderCallaghanApproximation(
            mu=mu, lambda_par=lambda_par
        )

        E = np.empty(
            (self.radius_integral_steps,
             acquisition_scheme.number_of_measurements))
        for i, radius in enumerate(radii):
            E[i] = (
                radii_pdf_normalized[i] *
                callaghan(acquisition_scheme, diameter=radius * 2)
            )

        E = np.trapz(E, x=radii, axis=0)
        return E


class DD1C4GammaDistributedGaussianPhaseCylinder(MicrostructureModel):
    r""" The Watson-Dispersed Van Gelderen cylinder model [1] - assuming
    finite pulse separation and pulse duration - for intra-axonal diffusion.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    alpha : float,
        shape of the gamma distribution.
    beta : float,
        scale of the gamma distrubution. Different from Bingham distribution!

    References
    ----------
    .. [1] Van Gelderen et al.
        "Evaluation of Restricted Diffusion
        in Cylinders. Phosphocreatine in Rabbit Leg Muscle"
        Journal of Magnetic Resonance Series B (1994)
    """

    _parameter_ranges = {
        'mu': ([0, -np.pi], [np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'alpha': (1e-10, np.inf),
        'beta': (1e-10, np.inf),
    }
    _parameter_scales = {
        'mu': 1.,
        'lambda_par': DIFFUSIVITY_SCALING,
        'alpha': 1.,
        'beta': BETA_SCALING
    }
    spherical_mean = False

    def __init__(self, mu=None, lambda_par=None, alpha=None, beta=None,
                 radius_integral_steps=35):
        self.mu = mu
        self.lambda_par = lambda_par
        self.alpha = alpha
        self.beta = beta
        self.radius_integral_steps = radius_integral_steps

    def __call__(self, acquisition_scheme, **kwargs):
        r"""
        Parameters
        ----------
        acquisition_scheme : acquisition scheme object
            contains all information on acquisition parameters such as bvalues,
            gradient directions, etc. Created from acquisition_scheme module.

        Returns
        -------
        E : array, shape(N),
            signal attenuation
        """
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)

        gamma_dist = stats.gamma(alpha, scale=beta)
        radius_max = gamma_dist.mean() + 6 * gamma_dist.std()
        radii = np.linspace(1e-50, radius_max, self.radius_integral_steps)
        area = np.pi * radii ** 2
        radii_pdf = gamma_dist.pdf(radii)
        radii_pdf_area = radii_pdf * area
        radii_pdf_normalized = (
            radii_pdf_area /
            np.trapz(x=radii, y=radii_pdf_area)
        )

        vg = cylinder_models.C4CylinderGaussianPhaseApproximation(
            mu=mu, lambda_par=lambda_par
        )

        E = np.empty(
            (self.radius_integral_steps,
             acquisition_scheme.number_of_measurements))
        for i, radius in enumerate(radii):
            E[i] = (
                radii_pdf_normalized[i] *
                vg(acquisition_scheme, diameter=radius * 2)
            )

        E = np.trapz(E, x=radii, axis=0)
        return E
