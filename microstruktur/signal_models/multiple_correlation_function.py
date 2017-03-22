from __future__ import division
from os.path import dirname, join, abspath
from warnings import warn
from six.moves import range

import numpy as np
import scipy.special
import scipy.linalg

from .constants import CONSTANTS


def dot3(a, b, c):
    return np.dot(
        a, np.dot(b, c)
    )


class FromMatrices:
    def __init__(self, Lambda, B):
        self.Lambda_matrix = np.diag(Lambda).copy()
        self.B_matrix = B.copy()

        self.U_matrix = np.zeros_like(Lambda)[:, None]
        self.U_matrix[0, 0] = 1

        self.Lambda_matrix.setflags(write=False)
        self.B_matrix.setflags(write=False)
        self.d_matrix.setflags(write=False)


class FromFile:
    def __init__(self, geometry='disk'):
        base = dirname(__file__)
        if geometry == 'disk':
            postfix = 'cl'
        elif geometry == 'slab':
            postfix = 'sl'
        elif geometry == 'sphere':
            postfix = 'sp'
        else:
            raise ValueError(
                'geometry must be one of disk/slab/sphere'
            )

        # pkg_resources.resource_string(__name__, 'MCF_L%s.npz' % postfix)
        # , '**'
        self.Lambda_matrix = np.diag(
            np.load(join(base, 'MCF_L%s.npz' % postfix))
        )
        self.B_matrix = np.load(join(base, 'MCF_B%s.npz' % postfix))

        self.Lambda_matrix.setflags(write=False)
        self.B_matrix.setflags(write=False)


class Slab:
    def __init__(self, N=10, epsilon=1e-10):
        self.N = N
        m, n = np.mgrid[0:N, 0:N]
        self.Lambda_matrix = self.Lambda(m, n)
        self.B_matrix = self.B(m, n)
        self.U_matrix = self.U(np.arange(N))[:, None]

        self.Lambda_matrix[abs(self.Lambda_matrix) < epsilon] = 0
        self.B_matrix[abs(self.B_matrix) < epsilon] = 0
        self.U_matrix[abs(self.U_matrix) < epsilon] = 0

        self.Lambda_matrix.setflags(write=False)
        self.B_matrix.setflags(write=False)
        self.U_matrix.setflags(write=False)

    def volume(self, length):
        return length

    def alpha(self, m):
        return m * np.pi

    def Lambda(self, m, n):
        ret = np.zeros_like(m, dtype=float)
        diag = m == n
        d = m[diag]
        ret[diag.nonzero()] = self.alpha(d) ** 2
        return ret

    def u(self, m, x):
        x = np.atleast_1d(x)
        res = self.epsilon(m) * np.cos(np.pi * m * x)
        res[(x <= 0) + (1 <= x)] = 0
        return res

    def U(self, m):
        ret = np.empty_like(m, dtype=float)
        ret[m == 0] = 1.
        m_ = m[m > 0]
        ret[m > 0] = np.sin(np.pi * m_) / (np.pi * m_)

        return ret

    def epsilon(self, m):
        delta = np.zeros_like(m)
        delta[m == 0] = 1
        return np.sqrt(2 - delta)

    def Bs(self, m, n):
        epsilon_m = self.epsilon(m)
        epsilon_n = self.epsilon(n)

        ret = (
            (1 + (-1) ** (m + n)) *
            epsilon_m * epsilon_n
        )
        return ret

    def B(self, m, n):
        epsilon_m = self.epsilon(m)
        epsilon_n = self.epsilon(n)

        ret = (
            ((-1) ** (m + n) - 1) *
            epsilon_m * epsilon_n
        )

        off_diag = m != n

        m_od = m[off_diag]
        n_od = n[off_diag]
        lambda_m = self.Lambda(m_od, m_od)
        lambda_n = self.Lambda(n_od, n_od)

        ret[off_diag] *= (lambda_m + lambda_n) / ((lambda_m - lambda_n) ** 2)

        ret[m == n] = .5
        return ret

    def propagator(self, t, r, domain_discretization=100):
        x = np.linspace(0, 1, domain_discretization)
        r = np.atleast_1d(r)

        prop = np.zeros((len(x), len(r)))
        arrivals = r[None, :] + x[:, None]
        for i in range(self.N):
            u1 = self.u(i, arrivals)
            u = self.u(i, x)[:, None]
            prop += (
                np.exp(-1 * self.Lambda_matrix[i, i] * t) *
                u * u1
            )

        prop = np.trapz(prop.T, x=x)
        return prop


class Disk:
    '''
    In this Disk implementation, the length parameter stands for
    the Disk radius
    '''
    def __init__(self, N=20, K=7, epsilon=1e-10, max_dim=60):
        self.N = N
        self.K = K

        self.alphas = np.empty((N, K))
        self.alphas[0, 0] = 0
        self.alphas[0, 1:] = scipy.special.jnp_zeros(0, K - 1)
        for n in range(1, N):
            self.alphas[n] = scipy.special.jnp_zeros(n, K)

        Ngrid, Kgrid = np.mgrid[0:N, 0:K]
        flat_alphas = self.alphas.ravel()
        m_map = np.argsort(flat_alphas)
        self.sorted_nk = np.c_[
            Ngrid.ravel()[m_map],
            Kgrid.ravel()[m_map]
        ]
        self.sorted_alphas = flat_alphas[m_map]

        mg, mg_ = np.mgrid[0: len(self.sorted_nk), 0: len(self.sorted_nk)]
        self.Lambda_matrix = self.Lambda(
            mg.ravel(), mg_.ravel()
        ).reshape(mg.shape)
        self.B_matrix = self.B(mg.ravel(), mg_.ravel()).reshape(mg.shape)
        self.U_matrix = self.U(np.arange(len(flat_alphas)))[:, None]

        self.sorted_alphas = self.sorted_alphas[:max_dim]
        self.sorted_nk = self.sorted_nk[:max_dim]
        self.Lambda_matrix = self.Lambda_matrix[:max_dim, :max_dim]
        self.B_matrix = self.B_matrix[:max_dim, :max_dim]
        self.U_matrix = self.U_matrix[:max_dim]

        for mat in (self.Lambda_matrix, self.B_matrix, self.U_matrix):
            mat[abs(mat) < epsilon] = 0
            mat.setflags(write=False)

    def volume(self, radius):
        return np.pi * radius ** 2

    def epsilon(self, m):
        delta = np.zeros_like(m)
        delta[m == 0] = 1
        return np.sqrt(2 - delta)

    def alpha(self, n, k):
        if n == 0 and k == 0:
            return 0
        elif n == 0:
            k = k + 1
        return scipy.special.jnp_zeros(n, k)[-1]

    def u_(self, n, k, r, phi):
        phi = np.atleast_1d(phi)

        alpha = self.alpha(n, k)
        beta = self.beta(n, k)
        res = (
            self.epsilon(n) * beta /
            (np.sqrt(np.pi) * scipy.special.jn(n, alpha))
        ) * (
            scipy.special.jn(n, alpha * r) * np.cos(n * phi)
        )
        res[r > 1] = 0
        return res

    def u(self, m, r, phi):
        r = np.atleast_1d(r)
        phi = np.atleast_1d(phi)

        n, k = self.sorted_nk[m]
        alpha = self.sorted_alphas[m]
        beta = self.beta(m)
        res = (
            self.epsilon(n) * beta /
            (np.sqrt(np.pi) * scipy.special.jn(n, alpha))
        ) * (
            scipy.special.jn(n, alpha * r) * np.cos(n * phi)
        )
        res[r > 1] = 0
        return res

    def U(self, m):
        '''
        U matrix for uniform sampling
        '''
        ret = np.zeros_like(m, dtype=float)
        n, k = self.sorted_nk[m].T
        ret[(n == 0) * (k == 0)] = 1.
        # alphas = self.sorted_alphas[m]
        # mask = (n == 0) * (alphas > 0)
        # betas = self.beta(m)[mask]
        # masked_alphas = alphas[mask]
        # epsilons = self.epsilon(0)
        # ret[mask] = (
        #     (epsilons * betas) /
        #     (sqrt(np.pi) * scipy.special.jn(0, masked_alphas)) *
        #     2 * scipy.special.jn(1, masked_alphas) / masked_alphas
        # )
        # ret[(n == 0) * (alphas == 0)] = 1
        return ret

    def Lambda(self, m, n):
        ret = np.zeros_like(m, dtype=float)
        diag = m == n
        ret[diag] = self.lambdas(m[diag])
        return ret

    def lambdas(self, m):
        return (self.sorted_alphas[m] ** 2)

    def beta(self, m):
        m = np.atleast_1d(m)
        res = np.empty_like(m, dtype=float)
        n, _ = self.sorted_nk[m].T
        res[m == 0] = 1.

        mask = m > 0
        # m_ = m[mask]
        n_ = n[mask]
        lambdas = self.lambdas(m)[mask]
        # alphas_ = self.sorted_alphas[m_]

        res[mask] = (
            (lambdas / (lambdas - n_)) ** .5 *
            1  # alphas_ / scipy.special.jn(n_, alphas_)
        )

        return res

    def B(self, m, m_):
        n, k = self.sorted_nk[m].T
        n_, k_ = self.sorted_nk[m_].T

        mask = (n == n_ + 1) + (n == n_ - 1)
        res = (
            mask *
            (1. + (n == 0) + (n_ == 0)) ** .5
        )
        res[mask] *= self.beta(m)[mask] * self.beta(m_)[mask]

        lambdas = self.lambdas(m)[mask]
        lambdas_ = self.lambdas(m_)[mask]
        nn_ = (n * n_)[mask]

        res[mask] *= (
            (lambdas + lambdas_ - 2 * nn_) /
            (lambdas - lambdas_) ** 2
        )

        return res

    def propagator(self, t, r, theta, domain_discretization=100):
        raise
        r = np.atleast_2d(r)
        theta = np.atleast_2d(theta)

        x_delta = 2. / domain_discretization
        phi_delta = 2 * np.pi / domain_discretization
        x, phi = np.mgrid[
            0:2 + x_delta:x_delta,
            -np.pi:np.pi + phi_delta:phi_delta
        ]

        n, k = self.sorted_nk.T
        k = k[n == 0]
        Ms = np.arange(len(n))[n == 0]

        prop = np.zeros((len(x), len(r)))
        arrivals = r[None, ...] + x[..., None]
        arrivals_angle = theta[None, ...] + phi[..., None]

        for m in Ms:
            u1 = self.u(m, arrivals, arrivals_angle)
            u = self.u(m, x, theta)[:, None]
            prop += (
                np.exp(-1 * self.Lambda_matrix[m, m] * t) *
                u * u1 * x[:, None]
            ) * 4 * np.pi ** 2

        prop = np.trapz(prop.T, x=x)
        return prop


class Sphere:
    '''
    In this Sphere implementation, the length parameter stands for
    the Sphere radius
    '''
    def __init__(self, epsilon=1e-10, max_dim=60):
        self.sorted_alphas = np.load(join(
            dirname(abspath(__file__)),
            'sphere_alphas.npy'
        ))[:max_dim]

        if max_dim > len(self.sorted_alphas):
            raise NotImplementedError(
                'Maximum number of basis functions is %d' %
                len(self.sorted_alphas)
            )

        self.sorted_nk = np.load(join(
            dirname(abspath(__file__)),
            'sphere_alphas_nk.npy'
        ))[:max_dim]

        mg, mg_ = np.mgrid[0: len(self.sorted_nk), 0: len(self.sorted_nk)]

        self.Lambda_matrix = self.Lambda(
            mg.ravel(), mg_.ravel()
        ).reshape(mg.shape)

        self.B_matrix = self.B(mg.ravel(), mg_.ravel()).reshape(mg.shape)
        self.U_matrix = self.U(np.arange(len(self.sorted_alphas)))[:, None]

        for mat in (self.Lambda_matrix, self.B_matrix, self.U_matrix):
            mat[abs(mat) < epsilon] = 0
            mat.setflags(write=False)

    def volume(self, radius):
        return 4 * np.pi * radius ** 3 / 3

    def epsilon(self, m):
        delta = np.zeros_like(m)
        delta[m == 0] = 1
        return np.sqrt(2 - delta)

    def alpha(self, n, k):
        mask = (self.sorted_nk == (2, 1)).all(1)
        if not mask:
            raise NotImplementedError(
                'Current implementation does not support '
                'calculating alpha for %d, %d' % (n, k)
            )
        return self.sorted_alphas[int(np.argwhere(mask))]

    def u_(self, n, k, r, phi):
        raise NotImplementedError('')
        phi = np.atleast_1d(phi)

        alpha = self.alpha(n, k)
        beta = self.beta(n, k)
        res = (
            self.epsilon(n) * beta /
            (np.sqrt(np.pi) * scipy.special.jn(n, alpha))
        ) * (
            scipy.special.jn(n, alpha * r) * np.cos(n * phi)
        )
        res[r > 1] = 0
        return res

    def u(self, m, r, phi):
        raise NotImplementedError('')
        r = np.atleast_1d(r)
        phi = np.atleast_1d(phi)

        n, k = self.sorted_nk[m]
        alpha = self.sorted_alphas[m]
        beta = self.beta(m)
        res = (
            self.epsilon(n) * beta /
            (np.sqrt(np.pi) * scipy.special.jn(n, alpha))
        ) * (
            scipy.special.jn(n, alpha * r) * np.cos(n * phi)
        )
        res[r > 1] = 0
        return res

    def U(self, m):
        '''
        U matrix for uniform sampling
        '''
        ret = np.zeros_like(m, dtype=float)
        n, k = self.sorted_nk[m].T
        ret[(n == 0) * (k == 0)] = 1.
        return ret

    def Lambda(self, m, n):
        ret = np.zeros_like(m, dtype=float)
        diag = m == n
        ret[diag] = self.lambdas(m[diag])
        return ret

    def lambdas(self, m):
        return (self.sorted_alphas[m] ** 2)

    def beta(self, m):
        m = np.atleast_1d(m)
        res = np.empty_like(m, dtype=float)
        n, _ = self.sorted_nk[m].T
        res[m == 0] = np.sqrt(3 / 2)

        mask = m > 0
        n_ = n[mask]
        lambdas = self.lambdas(m)[mask]

        res[mask] = np.sqrt(
            (2 * n_ + 1) * lambdas /
            (lambdas - n_ * (n_ + 1))
        )

        return res

    def B(self, m, m_):
        n, k = self.sorted_nk[m].T
        n_, k_ = self.sorted_nk[m_].T

        mask = (n == n_ + 1) + (n == n_ - 1)
        res = (
            mask *
            (n * n_ + 1) /
            ((2 * n + 1) * (2 * n_ + 1))
        )
        res[mask] *= self.beta(m)[mask] * self.beta(m_)[mask]

        lambdas = self.lambdas(m)[mask]
        lambdas_ = self.lambdas(m_)[mask]
        n = n[mask]
        n_ = n_[mask]

        res[mask] *= (
            (
                lambdas + lambdas_ -
                (n * (n_ + 1)) - (n_ * (n + 1))
            ) / (lambdas - lambdas_) ** 2
        )

        return res

    def propagator(self, t, r, theta, domain_discretization=100):
        raise
        r = np.atleast_2d(r)
        theta = np.atleast_2d(theta)

        x_delta = 2. / domain_discretization
        phi_delta = 2 * np.pi / domain_discretization
        x, phi = np.mgrid[
            0:2 + x_delta:x_delta,
            -np.pi:np.pi + phi_delta:phi_delta
        ]

        n, k = self.sorted_nk.T
        k = k[n == 0]
        Ms = np.arange(len(n))[n == 0]

        prop = np.zeros((len(x), len(r)))
        arrivals = r[None, ...] + x[..., None]
        arrivals_angle = theta[None, ...] + phi[..., None]

        for m in Ms:
            u1 = self.u(m, arrivals, arrivals_angle)
            u = self.u(m, x, theta)[:, None]
            prop += (
                np.exp(-1 * self.Lambda_matrix[m, m] * t) *
                u * u1 * x[:, None]
            ) * 4 * np.pi ** 2

        prop = np.trapz(prop.T, x=x)
        return prop


def calculate_p_and_qg(time, length, diffusion_constant, gyromagnetic_ratio):
    '''
    Grebenkov's dimensionless p and q factors
    '''
    p = diffusion_constant * time / (length ** 2)
    # Careful q here is not the q value it's Grebenkov's notation
    q = float(
        gyromagnetic_ratio *
        length * time
    )

    return p, q


def calculate_p_and_qg_length_derivatives(
    time, length, diffusion_constant, gyromagnetic_ratio
):
    '''
    Grebenkov's dimensionless p and q factors differentiated
    with respect to length
    '''
    p = -diffusion_constant * time / (length ** 3)
    # Careful q here is not the q value it's Grebenkov's notation
    q = float(gyromagnetic_ratio * time)

    return p, q


def defaults_from_class(func):
    def wrapper(self, **kwargs):
        new_kwargs = {}
        for k in self.default_protocol_vars:
            new_kwargs[k] = kwargs.setdefault(k, getattr(self, k, None))
        return func(self, **new_kwargs)
    return wrapper


class GradientStrengthEcho:
    '''
    Different Gradient Strength Echo protocols
    '''

    def __init__(
        self, Time=None, gradient_strength=None, delta=None,
        geometry=None,
        length=None,
        diffusion_constant=CONSTANTS['water_diffusion_constant'],
        gyromagnetic_ratio=CONSTANTS['water_gyromagnetic_ratio'],
    ):
        self.Time = Time
        self.gradient_strength = gradient_strength
        self.delta = delta
        self.geometry = geometry
        self.length = length
        self.diffusion_constant = diffusion_constant
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.default_protocol_vars = list(locals().keys())
        self.default_protocol_vars.remove('self')

    def attenuation_steady(self, **kwargs):
        '''
        Steady profile: no time between the two pulses (Delta=0)
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        p, qg = calculate_p_and_qg(
            kwargs['Time'], kwargs['length'],
            kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
        )
        geometry = kwargs['geometry']
        pLambda = (
            geometry.Lambda_matrix *
            p
        )

        jqgB = 1j * geometry.B_matrix * qg

        E = np.empty_like(gradient_strength, dtype=complex)
        for i, g in enumerate(gradient_strength):
            E[i] = np.dot(
                scipy.linalg.expm(-.5 * (
                    pLambda +
                    jqgB * g
                )),
                scipy.linalg.expm(-.5 * (
                    pLambda -
                    jqgB * g
                ))
            )[0, 0]

        return E

    def attenuation_constant_gradient(self, **kwargs):
        '''
        Constant gradient pulse
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        p, qg = calculate_p_and_qg(
            kwargs['Time'], kwargs['length'],
            kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
        )
        geometry = kwargs['geometry']

        pLambda = (
            geometry.Lambda_matrix *
            p
        )

        jqgB = 1j * geometry.B_matrix * qg

    #    E = np.empty_like(gradient_strength, dtype=complex)
        E = np.empty((len(gradient_strength),), dtype=complex)
        for i, g in enumerate(gradient_strength):
            E[i] = scipy.linalg.expm(-.5 * (
                pLambda +
                jqgB * g
            ))[0, 0]

        return E

    def attenuation_length_derivative_constant_gradient(self, **kwargs):
        '''
        Constant gradient pulse
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        length = kwargs['length']
        p, qg = calculate_p_and_qg(
            kwargs['Time'], length,
            kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
        )
        geometry = kwargs['geometry']

        pLambda = (
            geometry.Lambda_matrix *
            p
        )
        pLambda_L = - 2 * pLambda / length

        jqgB = 1j * geometry.B_matrix * qg
        jqgB_L = jqgB / length

    #    E = np.empty_like(gradient_strength, dtype=complex)
        E = np.empty((len(gradient_strength),), dtype=complex)
        for i, g in enumerate(gradient_strength):
            E_ = scipy.linalg.expm(-.5 * (
                pLambda +
                jqgB * g
            ))

            E[i] = -.5 * np.dot(
                E_, pLambda_L + jqgB_L * g
            )[0, 0]
        return E

    def attenuation_two_square_pulses_TE(self, **kwargs):
        '''
        Two square pulses separated by a time:
        Time = 2 * delta + Delta + TE
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))
            if kwargs[k] is None:
                del kwargs[k]

        if sum([
            int(k in kwargs) for k in ('TE', 'Delta', 'delta', 'Time')
        ]) != 3:
            raise ValueError(
                'Exactly 3 of delta; Delta; TE; and Time must be specified'
            )

        if 'delta' not in kwargs:
            kwargs['delta'] = (
                kwargs['Time'] -
                (kwargs['Delta'] + kwargs['TE']) / 2
            )
        elif 'Delta' not in kwargs:
            kwargs['Delta'] = (
                kwargs['Time'] -
                (2 * kwargs['delta'] + kwargs['TE'])
            )
        elif 'TE' not in kwargs:
            kwargs['TE'] = (
                kwargs['Time'] -
                (2 * kwargs['delta'] + kwargs['Delta'])
            )
        elif 'Time' not in kwargs:
            kwargs['Time'] = (
                kwargs['TE'] +
                (2 * kwargs['delta'] + kwargs['Delta'])
            )

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        # p, qg = calculate_p_and_qg(
        #    kwargs['Time'], kwargs['length'],
        #    kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
        # )
        geometry = kwargs['geometry']

        gradient_strength = np.atleast_1d(gradient_strength)
        T = kwargs['Time']
        delta = kwargs['delta']
        Delta = kwargs['Delta']
        TE = kwargs['TE']

        p, qg = calculate_p_and_qg(
            T, kwargs['length'],
            kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
        )
        pLambda = (
            geometry.Lambda_matrix * p
        )

        jqB = 1j * geometry.B_matrix * qg

        # The matrix exponential of a diagonal matrix
        # is the exponential of each element in the
        # diagonal
        if np.isinf(kwargs['Time']):
            raise ValueError('Total time must be finite')
        else:
            second_segment = np.diag(np.exp(-(
                np.diag(pLambda) * Delta / T
            )))
            fourth_segment = np.diag(np.exp(-(
                np.diag(pLambda) * TE / T
            )))

        attenuation = np.empty_like(gradient_strength, dtype=complex)
        for i, g in enumerate(gradient_strength):
            first_segment = np.nan_to_num(scipy.linalg.expm(-(
                pLambda +
                jqB * g
            ) * delta / T))

            third_segment = np.nan_to_num(scipy.linalg.expm(-(
                pLambda -
                jqB * g
            ) * delta / T))

            attenuation_mat = np.dot(
                np.dot(
                    np.dot(
                        first_segment,
                        second_segment
                    ),
                    third_segment
                ),
                fourth_segment
            )

            # The pre multiplication of
            # U and U* cancels

            attenuation[i] = np.dot(
                geometry.U_matrix.T,
                np.dot(
                    attenuation_mat,
                    geometry.U_matrix
                )
            )[0, 0]

        return attenuation

    def attenuation_two_square_pulses(self, **kwargs):
        '''
        Two square pulses separated by a time: 2 * delta - Time
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        # p, qg = calculate_p_and_qg(
        #    kwargs['Time'], kwargs['length'],
        #    kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
        # )
        geometry = kwargs['geometry']

        gradient_strength = np.atleast_1d(gradient_strength)
        T = kwargs['Time']
        delta = kwargs['delta']
        if np.isinf(T):
            p, qg = calculate_p_and_qg(
                1, kwargs['length'],
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            T = 1
        else:
            p, qg = calculate_p_and_qg(
                kwargs['Time'], kwargs['length'],
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            Delta = T - 2 * kwargs['delta']
            if Delta < 0:
                raise ValueError(
                    'Total time shorter than two pulse lengths (delta)'
                )
        pLambda = (
            geometry.Lambda_matrix * p
        )

        jqB = 1j * geometry.B_matrix * qg

        # The matrix exponential of a diagonal matrix
        # is the exponential of each element in the
        # diagonal
        if np.isinf(kwargs['Time']):
            second_segment = np.zeros_like(geometry.B_matrix)
            second_segment[0, 0] = 1
        else:
            second_segment = np.diag(np.exp(-(
                np.diag(pLambda) * Delta / T
            )))

        attenuation = np.empty_like(gradient_strength, dtype=complex)
        for i, g in enumerate(gradient_strength):
            first_segment = np.nan_to_num(scipy.linalg.expm2(-(
                pLambda +
                jqB * g
            ) * delta / T))

            third_segment = np.nan_to_num(scipy.linalg.expm2(-(
                pLambda -
                jqB * g
            ) * delta / T))

            attenuation_mat = np.dot(
                np.dot(
                    first_segment,
                    second_segment
                ),
                third_segment
            )

            # The pre multiplication of
            # U and U* cancels

            attenuation[i] = np.dot(
                geometry.U_matrix.T,
                np.dot(
                    attenuation_mat,
                    geometry.U_matrix
                )
            )[0, 0]

        return attenuation

    def attenuation_length_derivative_two_square_pulses(self, **kwargs):
        '''
        Two square pulses separated by a time: 2 * delta - Time
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        geometry = kwargs['geometry']

        gradient_strength = np.atleast_1d(gradient_strength)

        T = kwargs['Time']
        delta = kwargs['delta']
        length = kwargs['length']
        if np.isinf(T):
            p, qg = calculate_p_and_qg(
                1, length,
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            p_L, qg_L = calculate_p_and_qg_length_derivatives(
                1, length,
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )

            T = 1
        else:
            p, qg = calculate_p_and_qg(
                kwargs['Time'], length,
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            p_L, qg_L = calculate_p_and_qg_length_derivatives(
                kwargs['Time'], length,
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            Delta = T - 2 * kwargs['delta']

        pLambda = (
            geometry.Lambda_matrix * p
        )

        pLambda_L = (
            - 2 * pLambda / length
        )

        jqB = 1j * geometry.B_matrix * qg

        jqB_L = jqB / length

        # The matrix exponential of a diagonal matrix
        # is the exponential of each element in the
        # diagonal
        if np.isinf(kwargs['Time']):
            second_segment = np.zeros_like(geometry.B_matrix)
            second_segment[0, 0] = 1
        else:
            second_segment = np.exp(-np.diag(pLambda) * Delta / T)
            second_segment_L = -(
                second_segment *
                np.diag(pLambda_L) *
                Delta / T
            )

            second_segment = np.diag(second_segment)
            second_segment_L = np.diag(second_segment_L)

        attenuation = np.empty_like(gradient_strength, dtype=complex)
        for i, g in enumerate(gradient_strength):
            first_segment = np.nan_to_num(scipy.linalg.expm(-(
                pLambda +
                jqB * g
            ) * delta / T))
            first_segment_L = -np.dot(
                first_segment,
                pLambda_L + jqB_L * g
            ) * delta / T

            third_segment = np.nan_to_num(scipy.linalg.expm(-(
                pLambda -
                jqB * g
            ) * delta / T))
            third_segment_L = -np.dot(
                third_segment,
                pLambda_L - jqB_L * g
            ) * delta / T

            attenuation_mat = (
                dot3(first_segment_L, second_segment, third_segment) +
                dot3(first_segment, second_segment_L, third_segment) +
                dot3(first_segment, second_segment, third_segment_L)
            )

            # The pre multiplication of
            # U and U* cancels

            attenuation[i] = np.dot(
                geometry.U_matrix.T,
                np.dot(
                    attenuation_mat,
                    geometry.U_matrix
                )
            )[0, 0]

        return attenuation

    def attenuation_two_square_pulses_long_diffusion(
        self, **kwargs
    ):
        '''
        Two square pulses separated by an infinite time
        '''
        if 'Time' in kwargs:
            warn('Acquisition time ignored in the long diffusion time regime')

        kwargs['Time'] = np.inf
        return self.attenuation_two_square_pulses(**kwargs)

    def attenuation_stejkal_tanner_long_diffusion(self, **kwargs):
        '''
        Two pulses of impulse length separated by an infinite time.
        The parameter delta is taken in account in the
        Stejkal-Tanner approximation scheme
        '''
        if 'Time' in kwargs:
            warn('Acquisition time ignored in the long diffusion time regime')
        kwargs['Time'] = np.inf
        return self.attenuation_stejkal_tanner(**kwargs)

    def attenuation_stejkal_tanner(self, **kwargs):
        '''
        Two pulses of impulse length separated by a time time.
        The parameter delta is taken in account in the
        Stejkal-Tanner approximation scheme
        '''
        for k in self.default_protocol_vars:
            kwargs.setdefault(k, getattr(self, k, None))

        gradient_strength = np.atleast_1d(kwargs['gradient_strength'])
        geometry = kwargs['geometry']
        T = kwargs['Time']
        Time = kwargs['Time']
        delta = kwargs['delta']

        if np.isinf(Time):
            p, qg = calculate_p_and_qg(
                1, kwargs['length'],
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            T = 1
        else:
            p, qg = calculate_p_and_qg(
                Time, kwargs['length'],
                kwargs['diffusion_constant'], kwargs['gyromagnetic_ratio']
            )
            Delta = T - 2 * delta

        pLambda = (
            geometry.Lambda_matrix * p
        )

        jqB = 1j * geometry.B_matrix * qg

        # The matrix exponential of a diagonal matrix
        # is the exponential of each element in the
        # diagonal

        if np.isinf(Time):
            second_segment = np.zeros_like(geometry.B_matrix)
            second_segment[0, 0] = 1
        else:
            second_segment = np.diag(np.exp(-(
                np.diag(pLambda) * Delta / T
            )))

        attenuation = np.empty_like(gradient_strength, dtype=complex)
        for i, g in enumerate(gradient_strength):

            first_segment = np.nan_to_num(scipy.linalg.expm(
                (-jqB) * g * delta / T
            ))

            third_segment = np.nan_to_num(scipy.linalg.expm(
                (+jqB) * g * delta / T
            ))

            attenuation_mat = np.dot(
                np.dot(
                    first_segment,
                    second_segment
                ),
                third_segment
            )

            # The pre multiplication of
            # U and U* cancels

            attenuation[i] = np.dot(
                geometry.U_matrix.T,
                np.dot(
                    attenuation_mat,
                    geometry.U_matrix
                )
            )[0, 0]

        return attenuation
