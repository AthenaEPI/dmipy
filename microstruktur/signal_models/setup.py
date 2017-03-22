#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    # if len(parent_package):  # prevent installation of subpackages
    #    raise Exception()
    from numpy.distutils.misc_util import Configuration
    config = Configuration('signal', parent_package, top_path)
    config.add_data_files(
        'sphere_alphas.npy',
        'sphere_alphas_nk.npy',
        'MCF_Bcl.npz',
        'MCF_Bcp.npz',
        'MCF_Bpl.npz',
        'MCF_Bpp.npz',
        'MCF_Bsl.npz',
        'MCF_Bsp.npz',
        'MCF_Lcl.npz',
        'MCF_Lcp.npz',
        'MCF_Lpl.npz',
        'MCF_Lpp.npz',
        'MCF_Lsl.npz',
        'MCF_Lsp.npz',
    )

    print(config.todict())
    return config

if __name__ == '__main__':
    from setuptools import setup
    setup(**configuration(top_path='').todict())
