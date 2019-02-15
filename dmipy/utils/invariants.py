import numpy as np
from scipy.special import lpmv, factorial
from os.path import join as pjoin, dirname
import pkg_resources
from os.path import join
from scipy.sparse import coo_matrix, load_npz
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_mrtrix
from dipy.reconst.shm import real_sph_harm
import itertools

DATA_PATH = pkg_resources.resource_filename(
    'dmipy', 'data'
)


def _get_index(l, m):
    if l == 0:
        idx = 0
    else:
        i0 = ((l-1) * l)//2

        idx = i0 + (m+l)

    return int(idx)


def _LegendreP(m, l, x):
    return (-1)**m * lpmv(m, l, x)


def sh_matrix(sh_order, vecs, sh_base='mrtrix'):
    r""" Compute Spherical Harmonics (SH) matrix M
    Parameters
    ----------
    sh_order : int, even
        Truncation order of the SH.
    vecs : array, Nx3
        array unit directions [X,Y,Z].
    sh_base : string,
        the base of the SH.

    Returns
    --------
    M : array,
        The base of the SH sampled according to vecs.
    """
    if not(sh_base in ['mrtrix', 'dipy']):
        raise ValueError(
            'sh_base must be either "mrtrix" or "dipy"')
    r, theta, phi = cart2sphere(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    theta[np.isnan(theta)] = 0

    n_c = (sh_order + 1) * (sh_order + 2) // 2

    if sh_base == 'mrtrix':
        M, m, l = real_sym_sh_mrtrix(sh_order, theta, phi)
    else:
        if sh_base == 'dipy':
            shm = real_sph_harm
        else:
            shm = gaunt_real_sph_harm

        M = np.zeros((vecs.shape[0], n_c))
        counter = 0
        for l in range(0, sh_order + 1, 2):
            for m in range(-l, l + 1):
                M[:, counter] = shm(m, l, theta, phi)
                counter += 1
    return M


def _fit_sh(shell, M, L):
    if np.all(shell == 0):
        coef = np.zeros(M.shape[1])
    else:
        pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + L), M.T)
        coef = np.dot(pseudoInv, shell)
    return coef


def _laplace_beltrami(sh_order):
    "Returns the Laplace-Beltrami regularisation matrix for SH basis"
    n_c = (sh_order + 1)*(sh_order + 2)//2
    diagL = np.zeros(n_c)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = (l * (l + 1)) ** 2
            counter += 1

    return np.diag(diagL)


def fit_sh_to_data(data, vecs, sh_order=4, lambda_lb=1e-03, sh_base='mrtrix'):
    r""" Fit Spherical Harmonics (SH) to data
    Parameters
    ----------
    data : array,
        the data to be fitted using SH. Admitted shape are N, AxN, AxBxN, and
        AxBxCxN
    vecs : array, Nx3
        array unit directions [X,Y,Z] corresponding to the sampling of the data.
    sh_order : int, even
        Truncation order of the SH.
    lambda_lb : float, positive
        Laplace-Beltrami regualarization parameters.
    sh_base : string,
        the base of the SH.

    Returns
    --------
    coef : array,
        The SH coefficients that better fit the data.
    """
    if not(sh_base in ['mrtrix', 'dipy']):
        raise ValueError(
            'sh_base must be either "mrtrix" or "dipy"')
    M = sh_matrix(sh_order, vecs, sh_base=sh_base)
    L = _laplace_beltrami(sh_order) * lambda_lb
    shape = data.shape
    if data.ndim > 4:
        raise ValueError(
            'data.ndim must be lower than 5')
    n_c = (sh_order + 1)*(sh_order + 2)//2
    coef = np.zeros(list(shape[:-1])+[n_c])

    if len(shape) == 1:
        coef = _fit_sh(data, M, L)

    if len(shape) == 2:
        for i in range(shape[0]):
            coef[i, :] = _fit_sh(data[i, :], M, L)

    if len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                coef[i, j, :] = _fit_sh(data[i, j, :], M, L)

    if len(shape) == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    coef[i, j, k, :] = _fit_sh(data[i, j, k, :], M, L)
    return coef


def gaunt_sph_harm(m, l, theta, phi):
    r""" Compute spherical harmonics as defined in [1]
    Parameters
    ----------
    m : int,
        The order of the harmonic.
    l : int,
        The degree of the harmonic.
    theta : float,
        The azimuthal coordinate.
    phi : float,
        The polar coordinate.

    Returns
    --------
    val : float,
        The SH $y^m_l$ sampled at `theta` and `phi`.

    See Also
    --------
    [1] Homeier, Herbert HH, and E. Otto Steinborn. 'Some properties of the
    coupling coefficients of real spherical harmonics and their relation to
    Gaunt coefficients.' Journal of Molecular Structure: THEOCHEM 368 (1996):
    31-37.
    """
    x = np.cos(theta)
    am = np.abs(m)
    val = (1j)**(m+am)
    if l+am == 0:
        f = 1.0
    else:
        f = factorial(l-am)/factorial(l+am)

    val *= np.sqrt((2*l+1)/(4*np.pi) * f).astype(complex)
    val *= _LegendreP(am, l, x).astype(complex)
    val *= np.exp(1j*m*phi)
    return val


def gaunt_real_sph_harm(m, l, theta, phi):
    r""" Compute real spherical harmonics as defined in [1]
    Parameters
    ----------
    m : int,
        The order of the harmonic.
    l : int,
        The degree of the harmonic.
    theta : float,
        The azimuthal coordinate.
    phi : float,
        The polar coordinate.

    Returns
    --------
    val : float,
        The real SH $Y^m_l$ sampled at `theta` and `phi`.

    See Also
    --------
    [1] Homeier, Herbert HH, and E. Otto Steinborn. 'Some properties of the
    coupling coefficients of real spherical harmonics and their relation to
    Gaunt coefficients.' Journal of Molecular Structure: THEOCHEM 368 (1996):
    31-37.
    """
    if m > 0:
        val = np.sqrt(2) * np.real(gaunt_sph_harm(m, l, theta, phi))
    if m == 0:
        val = np.real(gaunt_sph_harm(0, l, theta, phi))
    if m < 0:
        val = np.sqrt(2) * np.imag(gaunt_sph_harm(np.abs(m), l, theta, phi))

    return val


def convert_sh(coef, in_base='mrtrix'):
    r"""Converts SH coefficients from 'mrtrix' or 'dipy' base to the Gaunt
    coefficients SH base
    """
    rad2 = np.sqrt(2)
    shape = coef.shape
    N = coef.shape[-1]
    sh_order = int((-3 + np.sqrt(1+8*N))/2.0)
    new_coef = np.copy(coef)

    if in_base == 'mrtrix':
        c = 0
        for l in range(0, sh_order + 1, 2):
            for m in range(-l, l + 1):
                if m != 0:
                    new_coef[..., c] = coef[..., c]*rad2
                else:
                    new_coef[..., c] = coef[..., c]
                c += 1

    if in_base == 'dipy':
        start = 0
        for l in range(0, sh_order + 1, 2):
            stop = start + 2*l+1
            new_coef[..., start:stop] = new_coef[..., start:stop][..., ::-1]
            start += 2*l+1

    return new_coef


def _kappa(l2, l3, m2, m3):
    r = max(abs(l2-l3), min(abs(m2+m3), abs(m2-m3)))
    return r


def _condition(l1, l2, l3, m1, m2, m3):
    m_cond = False
    l_cond = False
    n_cond = False
    if m1 in [m2+m3, m2-m3, -m2+m3, -m2-m3]:
        m_cond = True
    l_max = l2+l3

    k = _kappa(l2, l3, m2, m3)
    if (k+l_max) % 2 == 0:
        l_min = k
    else:
        l_min = k+1

    bl = (l1+l2+l3) % 2 == 0

    bl1 = False
    for ll in range(l_min, l_max+1, 2):
        if l1 == ll:
            bl1 = True

    l_cond = bl and bl1

    bm = 0
    bm += m1 < 0
    bm += m2 < 0
    bm += m3 < 0

    if bm % 2 == 0:
        n_cond = True

    r = m_cond and l_cond and n_cond

    return r


def _get_gaunt_index(l1, l2, l3, m1, m2, m3, sh_order=16):
    n_c = (sh_order+1)**2
    i1 = l1**2
    i2 = l2**2
    i3 = l3**2
    return (n_c**2) * (i1 + l1 + m1) + n_c*(i2 + l2 + m2) + i3 + l3 + m3


def real_gaunt(l1,  l2,  l3,  mu1,  mu2,  mu3,  RG):
    r"""Returns the real SH gaunt coefficients, namely the integral of
    three SH with the same arguments
    """
    c = _get_gaunt_index(l1, l2, l3, mu1, mu2, mu3)
    return RG[c]


def real_gaunt4(l1,  l2,  l3,  l4,  mu1,  mu2,  mu3,  mu4,  RG):
    r"""Returns the four SH real gaunt coefficients, namely the integral of
    four SH with the same arguments
    """
    N = 2*np.max([l1, l2, l3, l4])
    r = 0

    for l in range(0, N+1):
        for m in range(-l, l+1):
            c1 = _get_gaunt_index(l, l2, l3, m, mu2, mu3)
            c2 = _get_gaunt_index(l1, l, l4, mu1, m, mu4)
            r += RG[c1] * RG[c2]

    return r


def real_gaunt5(l1,  l2,  l3,  l4, l5,  mu1,  mu2,  mu3,  mu4, mu5, RG):
    r"""Returns the five SH real gaunt coefficients, namely the integral of
    five SH with the same arguments
    """
    N = 2*np.max([l1, l2, l3, l4, l5])
    r = 0
    for l in range(0, N+1):
        for m in range(-l, l+1):
            if _condition(l, l1, l2, m, mu1, mu2):
                c1 = _get_gaunt_index(l, l1, l2, m, mu1, mu2)
                for lp in range(0, N+1):
                    for mp in range(-lp, lp+1):
                        if _condition(lp, l3, l4, mp, mu3, mu4) and _condition(l, lp, l5, m, mp, mu5):
                            c2 = _get_gaunt_index(lp, l3, l4, mp, mu3, mu4)
                            c3 = _get_gaunt_index(l, lp, l5, m, mp, mu5)
                            r += RG[c1] * RG[c2] * RG[c3]

    return r


def get_invariants_list(sh_order):
    r""" Returns the list of algebraically independent invariants for a given
    SH degree.

    Parameters
    ----------
    sh_order : int, even
        the truncation order of the SH

    Returns
    -------
    l_list : list,
        list of lists of degrees representing the algebraically independent
        invariants
    """
    if sh_order > 8:
        raise ValueError(
            'SH order is greater than 8')
    if sh_order % 2 != 0:
        raise ValueError(
            'SH order must be even')
    l_list = []
    if sh_order == 0:
        l_list = [[0]]
    if sh_order == 2:
        l_list = [[0], [2, 2], [2, 2, 2]]
    if sh_order == 4:
        l_list = [[0],
                  [2, 2],
                  [4, 4],
                  [2, 2, 2],
                  [2, 2, 4],
                  [2, 4, 4],
                  [4, 4, 4],
                  [2, 2, 2, 4],
                  [2, 2, 4, 4],
                  [2, 4, 4, 4],
                  [4, 4, 4, 4],
                  [2, 2, 2, 2, 4]]
    if sh_order == 6:
        l_list = [[0],
                  [2, 2],
                  [4, 4],
                  [6, 6],
                  [2, 2, 2],
                  [2, 2, 4],
                  [2, 4, 4],
                  [2, 4, 6],
                  [2, 6, 6],
                  [4, 4, 4],
                  [4, 4, 6],
                  [4, 6, 6],
                  [6, 6, 6],
                  [2, 2, 2, 4],
                  [2, 2, 2, 6],
                  [2, 2, 4, 4],
                  [2, 2, 4, 6],
                  [2, 2, 6, 6],
                  [2, 4, 4, 4],
                  [2, 4, 4, 6],
                  [2, 4, 6, 6],
                  [2, 6, 6, 6],
                  [4, 4, 4, 4],
                  [4, 4, 4, 6],
                  [4, 4, 6, 6]]
    if sh_order == 8:
        l_list = [[0],
                  [2, 2],
                  [4, 4],
                  [6, 6],
                  [8, 8],
                  [2, 2, 2],
                  [2, 2, 4],
                  [2, 4, 4],
                  [2, 4, 6],
                  [2, 6, 6],
                  [2, 6, 8],
                  [2, 8, 8],
                  [4, 4, 4],
                  [4, 4, 6],
                  [4, 4, 8],
                  [4, 6, 6],
                  [4, 6, 8],
                  [4, 8, 8],
                  [6, 6, 6],
                  [6, 6, 8],
                  [6, 8, 8],
                  [8, 8, 8],
                  [2, 2, 2, 4],
                  [2, 2, 2, 6],
                  [2, 2, 4, 4],
                  [2, 2, 4, 6],
                  [2, 2, 4, 8],
                  [2, 2, 6, 6],
                  [2, 2, 6, 8],
                  [2, 2, 8, 8],
                  [2, 4, 4, 4],
                  [2, 4, 4, 6],
                  [2, 4, 4, 8],
                  [2, 4, 6, 6],
                  [2, 4, 6, 8],
                  [2, 4, 8, 8],
                  [2, 6, 6, 6],
                  [2, 6, 6, 8],
                  [2, 6, 8, 8],
                  [2, 8, 8, 8],
                  [4, 4, 4, 4],
                  [4, 4, 4, 6]]
    return l_list


def get_invariants(sh_coef, l_list, sh_base='mrtrix', normalize=True):
    r""" Returns the rotation invariants features specified in l_list for the
    given SH coefficients

    Parameters
    ----------
    sh_coef : array,
        array containing the SH coefficients. It can be multidimensional but
        the last dimension must contain the SH coefficients. E.g. a three
        dimensional image with dimension x, y, and z the shape of sh_coef
        must be [x,y,z,n_c] with n_c the number of SH coefficients.
    l_list : list,
        the list of lists of desired invariants
    sh_base : string,
        the base in which the sh_coef are represented. Admitted values are
        'mrtrix', 'dipy', and 'gaunt'
    normalize : boolean,
        True if the invariants have to be normalized between [-1,1]

    Returns
    -------
    invariants : array
        array containing the desired invariants
    """
    if not(sh_base in ['mrtrix', 'dipy']):
        raise ValueError(
            'sh_base must be either "mrtrix" or "dipy"')

    l_max = np.max([np.max(l) for l in l_list])
    if l_max > 8:
        raise ValueError(
            'maximum degree in l_list is greater than 8')

    len_max = np.max([len(l) for l in l_list])
    if len_max > 5:
        raise ValueError(
            'maximum power in l_list is greater than 5')

    odd_number = np.sum([np.sum(np.array(l) % 2) for l in l_list])
    if odd_number > 0:
        raise ValueError(
            'all of the degrees in l_list must be even')

    N = sh_coef.shape[-1]
    sh_order = int((-3 + np.sqrt(1+8*N))/2.0)

    if sh_order < l_max:
        raise ValueError(
            'SH order of the coefficients is lower than the degree of the required invariants')

    RG = load_npz(join(DATA_PATH, 'real_gaunt_coefficients.npz')).toarray()[0, :]
    coef = convert_sh(sh_coef, in_base=sh_base)

    inv_number = len(l_list)

    if len(coef.shape) > 1:
        shape = list(coef.shape[:-1])+[inv_number]
    else:
        shape = inv_number

    invariants = np.zeros(shape)

    n_c = (sh_order + 1) * (sh_order + 2) // 2
    g_sh = np.zeros(n_c)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            g_sh[counter] = gaunt_real_sph_harm(m, l, 0, 0)
            counter += 1

    for i, l_i in enumerate(l_list):
        len_l = len(l_i)
        if len_l == 1:
            if l_i[0] == 0:
                mtv = 1
                invariants[..., i] = coef[..., 0]*np.sqrt(4*np.pi)

        if len_l == 2:
            if l_i[0] == l_i[1]:
                ll = l_i[0]
                mtv = (2*ll+1)/(4*np.pi)
                start = (ll-1)*(ll)//2
                stop = (ll+1)*(ll+2)//2
                invariants[..., i] = np.sum(coef[..., start:stop]**2, -1)

        if len_l == 3:
            l1 = l_i[0]
            l2 = l_i[1]
            l3 = l_i[2]
            mtv = 0
            m_list = itertools.product(range(-l1, l1+1), range(-l2, l2+1), range(-l3, l3+1))
            for m1,m2,m3 in m_list:
                rg = real_gaunt(l1, l2, l3, m1, m2, m3, RG)
                mtv += g_sh[_get_index(l1, m1)] * g_sh[_get_index(l2, m2)] * g_sh[_get_index(l3, m3)] * rg
                invariants[..., i] += coef[..., _get_index(l1, m1)]*coef[..., _get_index(l2, m2)]*coef[..., _get_index(l3, m3)]*rg

        if len_l == 4:
            l1 = l_i[0]
            l2 = l_i[1]
            l3 = l_i[2]
            l4 = l_i[3]
            mtv = 0
            m_list = itertools.product(range(-l1, l1+1), range(-l2, l2+1), range(-l3, l3+1), range(-l4, l4+1))
            for m1,m2,m3,m4 in m_list:
                rg = real_gaunt4(l1, l2, l3, l4, m1, m2, m3, m4, RG)
                mtv += g_sh[_get_index(l1, m1)] * g_sh[_get_index(l2, m2)] * g_sh[_get_index(l3, m3)] * g_sh[_get_index(l4, m4)] * rg
                invariants[..., i] += coef[..., _get_index(l1, m1)]*coef[..., _get_index(l2, m2)]*coef[..., _get_index(l3, m3)]*coef[..., _get_index(l4, m4)]*rg

        if len_l == 5:
            l1 = l_i[0]
            l2 = l_i[1]
            l3 = l_i[2]
            l4 = l_i[3]
            l5 = l_i[4]
            mtv = 0
            m_list = itertools.product(range(-l1, l1+1), range(-l2, l2+1), range(-l3, l3+1), range(-l4, l4+1), range(-l5, l5+1))
            for m1,m2,m3,m4,m5 in m_list:
                rg = real_gaunt5(l1, l2, l3, l4, l5, m1, m2, m3, m4, m5, RG)
                mtv += g_sh[_get_index(l1, m1)] * g_sh[_get_index(l2, m2)] * g_sh[_get_index(l3, m3)] * g_sh[_get_index(l4, m4)] * g_sh[_get_index(l5, m5)] * rg
                invariants[..., i] += coef[..., _get_index(l1, m1)]*coef[..., _get_index(l2, m2)]*coef[..., _get_index(l3, m3)]*coef[..., _get_index(l4, m4)]*coef[..., _get_index(l5, m5)]*rg
        if normalize:
            invariants[..., i] = invariants[..., i]/mtv

    return invariants
