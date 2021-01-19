from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 1
_version_minor = 0
_version_micro = 4  # use '' for first of series, number for 1 and above
_version_extra = ''
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "dmipy: diffusion microstructure imaging in python"
# Long description will go up on the pypi page
long_description_content_type = 'text/markdown'

NAME = "dmipy"
MAINTAINER = "Rutger Fick"
MAINTAINER_EMAIL = "fick.rutger@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION_CONTENT_TYPE = long_description_content_type
URL = "https://github.com/AthenaEPI/dmipy"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Rutger Fick"
AUTHOR_EMAIL = "fick.rutger@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {
    'dmipy': [
        pjoin('data', 'gradient_tables', '*'),
        pjoin('data', 'spheres', '*'),
        pjoin('data', 'bingham_normalization_splinefit.npz'),
        pjoin('data', 'wu_minn_hcp_subjects.txt'),
        pjoin('data', 'wu_minn_hcp_coronal_slice.nii.gz'),
        pjoin('data', 'tanguy_cat_spinal_cord', '*'),
        pjoin('data', 'camino', '*'),
        pjoin('data', 'de_santis_camino', '*'),
        pjoin('data', 'isbi2015_white_matter_challenge', '*')
    ]
}
REQUIRES = ["dipy", "scipy", "numpy (>=1.13)"]
