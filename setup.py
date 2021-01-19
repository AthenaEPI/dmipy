import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in dmipy/version.py
ver_file = os.path.join('dmipy', 'version.py')
with open(ver_file) as f:
    exec(f.read())

with open ('requirements.txt', "r") as f:
    requirements=f.read().splitlines()[::-1]

with open('README.md', 'r') as f:
    long_description = f.read()

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=long_description,
            long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=requirements,
            requires=requirements)


if __name__ == '__main__':
    setup(**opts)
