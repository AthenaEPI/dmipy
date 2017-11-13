# -*- coding: utf-8 -*-
import numpy as np
from dipy.data import get_sphere
SPHERE = get_sphere('symmetric362')


class SphericalIntegrator:
    def __init__(self, sphere=SPHERE):
        self.sphere = sphere
        self.spherical_excesses = np.empty(len(sphere.faces))
        self.spherical_face_centroids = np.empty((len(sphere.faces), 3))

        for i, face in enumerate(sphere.faces):
            face_vertices = sphere.vertices[face]
            self.spherical_excesses[i] = spherical_excess(face_vertices)

            self.spherical_face_centroids[i] = face_vertices.mean(0)
            self.spherical_face_centroids[i] /= np.linalg.norm(
                self.spherical_face_centroids[i]
            )

    def integrate(self, f, args=tuple()):
        return (
            np.atleast_2d(f(self.spherical_face_centroids, *args)) *
            self.spherical_excesses[:, None]
        ).sum(0)


def spherical_excess(abc):
    a, b, c = abc
    vs = np.r_[
        np.linalg.norm(b - a),
        np.linalg.norm(c - a),
        np.linalg.norm(b - c)
    ]
    s = vs.sum() / 2
    vs = 2 * np.arcsin(vs / 2)
    vs_ = np.tan((s - vs) / 2)

    excess = np.arctan(4 * np.sqrt(np.tan(s / 2) * np.prod(vs_)))

    return excess


def spherical_triangle_centroid_value(f, abc, args=tuple()):
    mean_point = abc.mean(axis=0)
    mean_point /= np.linalg.norm(mean_point)
    return f(mean_point, *args)


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
    R = np.array([[1, 0, 0],
                  [0, np.cos(psi), -np.sin(psi)],
                  [0, np.sin(psi), np.cos(psi)]])
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
    x, y, z = sphere2cart(np.r_[1., theta, phi])
    return rotation_matrix_100_to_xyz(float(x), float(y), float(z))


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
    if np.all(np.r_[x, y, z] == np.r_[1., 0., 0.]):
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


def T1_tortuosity(f_intra, lambda_par):
    """Tortuosity model for perpendicular extra-axonal diffusivity [1, 2, 3].

    Parameters
    ----------
    f_intra : float,
        intra-axonal volume fraction [0, 1].
    lambda_par : float,
        parallel diffusivity.

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
    lambda_perp = (1 - f_intra) * lambda_par
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
