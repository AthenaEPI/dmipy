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
