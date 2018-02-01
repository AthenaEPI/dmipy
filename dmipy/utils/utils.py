# -*- coding: utf-8 -*-
import numpy as np
from dipy.data import get_sphere
from dipy.utils.optpkg import optional_package
SPHERE = get_sphere('symmetric362')
numba, have_numba, _ = optional_package("numba")


def perpendicular_vector(v):
    """Returns a perpendicular vector to vector "v".

    Parameters
    ----------
    v : array, shape (3)
        normally Cartesian unit vector, but can also be any vector.

    Returns
    -------
    v_perp : array, shape (3)
        If v is unit vector, v_perp is a Cartesian unit vector perpendicular
        to v.
    """
    v_ = v * 1.
    if v_[1] == 0 and v_[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            v_perp = np.cross(v_, [0, 1, 0])
            v_perp /= np.linalg.norm(v_perp)
            return v_perp
    v_perp = np.cross(v_, [1, 0, 0])
    v_perp /= np.linalg.norm(v_perp)
    return v_perp


def rotation_matrix_around_100(psi):
    """Generates a rotation matrix that rotates around the x-axis (1, 0, 0).

    Parameters
    ----------
    psi : float,
        euler angle [0, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    R = np.zeros((3, 3))
    R[0, 0] = 1.
    R[1, 1] = cos_psi
    R[1, 2] = -sin_psi
    R[2, 1] = sin_psi
    R[2, 2] = cos_psi
    return R


def rotation_matrix_100_to_theta_phi(theta, phi):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position on the unit sphere.

    Parameters
    ----------
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    x, y, z = unitsphere2cart_1d([theta, phi])
    return rotation_matrix_100_to_xyz(x, y, z)


def rotation_matrix_100_to_xyz(x, y, z):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position in Cartesian space.

    Parameters
    ----------
    x, y, z : floats,
        position in Cartesian space.

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    if x == 1.:
        if y == 0.:
            if z == 0.:
                return np.eye(3)
    y2 = y ** 2
    z2 = z ** 2
    yz = y * z
    R = np.array([[x, -y, -z],
                  [y, (x * y2 + z2) / (y2 + z2), ((x - 1) * yz) / (y2 + z2)],
                  [z, ((x - 1) * yz) / (y2 + z2), (y2 + x * z2) / (y2 + z2)]])
    return R


def rotation_matrix_001_to_xyz(x, y, z):
    """Generates a rotation matrix that rotates from the z-axis (0, 0, 1) to
    an other position in Cartesian space.

    Parameters
    ----------
    x, y, z : floats,
        position in Cartesian space.

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    if np.all(np.r_[x, y, z] == np.r_[0., 0., 1.]):
        return np.eye(3)
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y
    R = np.array([[(y2 + x2 * z) / (x2 + y2), (xy * (z - 1)) / (x2 + y2), x],
                  [(xy * (z - 1)) / (x2 + y2), (x2 + y2 * z) / (x2 + y2), y],
                  [-x, -y, z]])
    return R


def rotation_matrix_100_to_theta_phi_psi(theta, phi, psi):
    """Generates a rotation matrix that rotates from the x-axis (1, 0, 0) to
    an other position in Cartesian space, and rotates about its axis.

    Parameters
    ----------
    theta : float,
        inclination of polar angle of main angle mu [0, pi].
    phi : float,
        polar angle of main angle mu [-pi, pi].
    psi : float,
        angle in radians of the bingham distribution around mu [0, pi].

    Returns
    -------
    R : array, shape (3 x 3)
        Rotation matrix.
    """
    R_100_to_theta_phi = rotation_matrix_100_to_theta_phi(theta, phi)
    R_around_100 = rotation_matrix_around_100(psi)
    return np.dot(R_100_to_theta_phi, R_around_100)


def T1_tortuosity(lambda_par, vf_intra, vf_extra=None):
    """Tortuosity model for perpendicular extra-axonal diffusivity [1, 2, 3].
    If vf_extra=None, then vf_intra must be a nested volume fraction, in the
    sense that E_bundle = vf_intra * E_intra + (1 - vf_intra) * E_extra, with
    vf_intra + (1 - vf_intra) = 1.
    If both vf_intra and vf_extra are given, then they have be be normalized
    fractions, in the sense that vf_intra + vf_extra <= 1.

    Parameters
    ----------
    lambda_par : float,
        parallel diffusivity.
    vf_intra : float,
        intra-axonal volume fraction [0, 1].
    vf_extra : float, (optional)
        extra-axonal volume fraction [0, 1].

    Returns
    -------
    lambda_perp : float,
        Rotation matrix.

    References
    -------
    .. [1] Bruggeman, Von DAG. "Berechnung verschiedener physikalischer
        Konstanten von heterogenen Substanzen. I. Dielektrizitätskonstanten und
        Leitfähigkeiten der Mischkörper aus isotropen Substanzen." Annalen der
        physik 416.7 (1935): 636-664.
    .. [2] Sen et al. "A self-similar model for sedimentary rocks with
        application to the dielectric constant of fused glass beads."
        Geophysics 46.5 (1981): 781-795.
    .. [3] Szafer et al. "Theoretical model for water diffusion in tissues."
        Magnetic resonance in medicine 33.5 (1995): 697-712.
    """
    if vf_extra is None:
        lambda_perp = (1 - vf_intra) * lambda_par
    else:
        fraction_intra = vf_intra / (vf_intra + vf_extra)
        lambda_perp = (1 - fraction_intra) * lambda_par
    return lambda_perp


def parameter_equality(param):
    "Function to force two model parameters to be equal in the optimization"
    return param


def define_shell_indices(bvals, b_value_ranges):
    """ Function to facilitate defining shell indices given some manual ranges
    in b-values. This function is useful as in practice the actual b-values may
    fluctuate slightly around the planned b-value. This information is needed
    by some models doing spherical convolutions or spherical means.
    CAUTION: If a data set has variations in pulse duration delta and/or pulse
    separation Delta, then different shells can have similar b-values. This
    means these shells may not be separable in b-value, and you'll have to do
    it manually.

    Parameters
    ----------
    bvals : 1D array of size (N_data),
        The b-values corresponding to the measured DWIs.
    b_value_ranges : 2D array of size (N_shells, 2)
        A list indicating for every shell the lower and upper b-value range.

    Returns
    -------
    shell_indices : 1D integer array of size (N_data),
        The shell indices corresponding to each DWI measurement. The index 0
        corresponds with the b0 measurements, while higher numbers indicate
        other shells. The numbers are ordered in the same order they are given
        in b_value_ranges.
    mean_shell_bvals : 1D array of size (N_shells),
        The mean b-value in each shell.
    """

    shell_indices = np.empty_like(bvals, dtype=int)
    mean_shell_bvals = np.zeros(len(b_value_ranges))
    shell_counter = 0
    dwi_counter = 0
    for b_range in b_value_ranges:
        lower_range = b_range[0]
        upper_range = b_range[1]
        shell_mask = np.all([bvals >= lower_range, bvals <= upper_range],
                            axis=0)
        shell_indices[shell_mask] = shell_counter
        mean_shell_bvals[shell_counter] = np.mean(bvals[shell_mask])
        dwi_counter += sum(shell_mask)
        shell_counter += 1
    if dwi_counter < len(bvals):
        msg = ("b_value_ranges only covered " + str(dwi_counter) +
               " out of " + str(len(bvals)) + " dwis")
        raise ValueError(msg)
    return shell_indices, mean_shell_bvals


def cart2sphere(cartesian_coordinates):
    """"Function to estimate spherical coordinates from cartesian coordinates
    according to wikipedia. Conforms with the dipy notation.

    Parameters
    ----------
    cartesian_coordinates : array of size (3) or (N x 3),
        array of cartesian coordinate vectors [x, y, z].

    Returns
    -------
    spherical_coordinates : array of size (3) or (N x 3),
        array of spherical coordinate vectors [r, theta, phi].
        range of theta [0, pi]. range of phi [-pi, pi].
    """
    if np.ndim(cartesian_coordinates) == 1:
        x, y, z = cartesian_coordinates
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r > 0:
            theta = np.arccos(z / r)
        else:
            theta = 0
        phi = np.arctan2(y, x)
        spherical_coordinates = np.r_[r, theta, phi]
    elif np.ndim(cartesian_coordinates) == 2:
        x, y, z = cartesian_coordinates.T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        theta = np.where(r > 0, theta, 0.)
        phi = np.arctan2(y, x)
        spherical_coordinates = np.c_[r, theta, phi]
    else:
        msg = "coordinates must be array of size 3 or N x 3."
        raise ValueError(msg)
    return spherical_coordinates


def cart2mu(xyz):
    """
    Function to estimate spherical coordinates from cartesian coordinates
    according to wikipedia. Conforms with the dipy notation.

    Parameters
    ----------
    cartesian_coordinates : array of size (3) or (N x 3),
        array of cartesian coordinate vectors [x, y, z].

    Returns
    -------
    spherical_coordinates : array of size (2) or (N x 2),
        array of spherical coordinate vectors [theta, phi].
        range of theta [0, pi]. range of phi [-pi, pi].
    """
    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])
    r = np.linalg.norm(xyz, axis=-1)
    mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
    mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
    mu[r == 0] = 0, 0
    return mu


def R2mu_psi(R):
    """
    Function to estimate orientation mu and secondary orientation angle psi
    from a 3x3 rotation matrix. Can be given array of rotation matrices.

    Parameters
    ----------
    R : Array of size (N, 3, 3)
        rotation matrices that possibly can be estimated by DTI.

    Returns
    -------
    mu : array of size (N, 2),
        orientations in [theta, phi] angles
    psi : array of size (N),
        secondary orientation psi (for Bingham for example).
    """
    mu = cart2mu(R[..., :, 0])
    mu_flat = mu.reshape([-1, 2])
    R_flat = R.reshape([-1, 3, 3])
    psi_flat = np.zeros(len(mu_flat))
    for i in xrange(len(mu_flat)):
        R_theta_phi = rotation_matrix_100_to_theta_phi(
            mu_flat[i, 0], mu_flat[i, 0])
        psi_flat[i] = np.arcsin(np.dot(R_theta_phi.T, R_flat[i])[2, 1])
    psi = psi_flat.reshape(mu.shape[:-1])
    psi[psi < 0] += np.pi
    return mu, psi


def sphere2cart(spherical_coordinates):
    """"Function to estimate cartesian coordinates from spherical coordinates
    according to wikipedia. Conforms with the dipy notation.

    Parameters
    ----------
    spherical_coordinates : array of size (3) or (N x 3),
        array of spherical coordinate vectors [r, theta, phi].
        range of theta [0, pi]. range of phi [-pi, pi].

    Returns
    -------
    cartesian_coordinates : array of size (3) or (N x 3),
        array of cartesian coordinate vectors [x, y, z].
    """
    if np.ndim(spherical_coordinates) == 1:
        r, theta, phi = spherical_coordinates
        sintheta = np.sin(theta)
        x = r * sintheta * np.cos(phi)
        y = r * sintheta * np.sin(phi)
        z = r * np.cos(theta)
        cartesian_coordinates = np.r_[x, y, z]
    elif np.ndim(spherical_coordinates) == 2:
        r, theta, phi = spherical_coordinates.T
        sintheta = np.sin(theta)
        x = r * sintheta * np.cos(phi)
        y = r * sintheta * np.sin(phi)
        z = r * np.cos(theta)
        cartesian_coordinates = np.c_[x, y, z]
    else:
        msg = "coordinates must be array of size 3 or N x 3."
        raise ValueError(msg)
    return cartesian_coordinates


def unitsphere2cart_1d(mu):
    """Optimized function dedicated to convert 1D unit sphere coordinates
    to cartesian coordinates.

    Parameters
    ----------
    mu : array of size (2)
        unit sphere coordinates, as theta, phi = mu

    Returns
    -------
    mu_cart, array of size (3)
        mu in cartesian coordinates, as x, y, z = mu_cart
    """
    theta, phi = mu
    mu_cart = np.zeros(3)
    sintheta = np.sin(theta)
    mu_cart[0] = sintheta * np.cos(phi)
    mu_cart[1] = sintheta * np.sin(phi)
    mu_cart[2] = np.cos(theta)
    return mu_cart


def unitsphere2cart_Nd(mu):
    """Optimized function deicated to convert 1D unit sphere coordinates
    to cartesian coordinates.

    Parameters
    ----------
    mu : Nd array of size (..., 2)
        unit sphere coordinates, as theta, phi = mu

    Returns
    -------
    mu_cart, Nd array of size (..., 3)
        mu in cartesian coordinates, as x, y, z = mu_cart
    """
    mu_cart = np.zeros(np.r_[mu.shape[:-1], 3])
    theta = mu[..., 0]
    phi = mu[..., 1]
    sintheta = np.sin(theta)
    mu_cart[..., 0] = sintheta * np.cos(phi)
    mu_cart[..., 1] = sintheta * np.sin(phi)
    mu_cart[..., 2] = np.cos(theta)
    return mu_cart


if have_numba:
    unitsphere2cart_1d = numba.njit()(unitsphere2cart_1d)
