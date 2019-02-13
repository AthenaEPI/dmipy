import numpy as np
from numpy.testing import assert_array_almost_equal
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_mrtrix
from dmipy.utils.invariants import  get_invariants_list, get_invariants

def test_invariants():
    sh_order = 8
    v = np.random.randn(100,3)
    v = v / np.linalg.norm(v, axis=1)[:,None]
    r, theta, phi = cart2sphere(v[:,0], v[:,1], v[:,2])
    coef, m, l = real_sym_sh_mrtrix(sh_order, theta, phi)
    for l in range(0,sh_order+1,2):
        n_c = (l+1)*(l+2)//2
        l_list = get_invariants_list(l)
        invariants = get_invariants(coef[:,0:n_c], l_list)
        inv_test = np.ones((100, len(l_list)))
        assert_array_almost_equal(invariants, inv_test, decimal=6)
    return invariants, inv_test
