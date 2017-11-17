#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
from setuptools import setup

setup(
    name='pyoof',
    version='0.1',
    description='pyoof - OOF holography package',
    author='Tomas Cassanelli',
    author_email='tcassanelli@gmail.com',
    # url='http://',
    install_requires=[
        'setuptools',
        'numpy>=1.8',
        'scipy>=0.15',
        'astropy>=1.1',
        'matplotlib>=2.0',
        ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=['pyoof', 'pyoof.zernike', 'pyoof.telgeometry', 'pyoof.aperture'],
    package_dir={
        'pyoof': 'pyoof',
        'pyoof.zernike': 'pyoof/zernike',
        'pyoof.telgeometry': 'pyoof/telgeometry',
        'pyoof.aperture': 'pyoof/aperture'
        },
    include_package_data=True,  # see MANIFEST.in
    long_description='''pyoof is a Python pacakge which performs out-of-focus (OOF) holography n astronomical beam maps for single-dish radio telescopes.'''
    )
