#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup


setup(
    name='pyoof',
    version='0.1',
    description='pyoof - out-of-focus holography package',
    author='Tomas Cassanelli',
    author_email='tcassanelli@gmail.com',
    # url='http://www3.mpifr-bonn.mpg.de/staff/bwinkel/',
    install_requires=[
        'setuptools',
        'numpy>=1.8',
        'scipy>=0.15',
        'astropy>=1.1',
        'matplotlib>=2.0',
    ],
    packages=['pyoof', 'pyoof.zernike', 'pyoof/telgeometry', 'pyoof/aperture'],
    package_dir={
        'pyoof': 'pyoof',
        'pyoof.zernike': 'pyoof/zernike',
        'pyoof.telgeometry': 'pyoof/telgeometry',
        'pyoof.aperture': 'pyoof/aperture'
        },
    package_data={
        '': ['pyoof.mplstyle'],
        '': ['config_params.yaml']},
    long_description='''
        pyoof is a Python pacakge that contains all needed tools to perform
        out-of-focus (OOF) holography on astronomical beam maps for single
        dish radio telescopes.
    '''
    )
