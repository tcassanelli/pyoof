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
    packages=['pyoof', 'pyoof.zernike'],
    package_dir={
        'pyoof': 'pyoof',
        'pyoof.zernike': 'pyoof/zernike',
        },
    package_data={
        '': ['pyoof.mplstyle'],
        '': ['config_params.yaml']},
    long_description='''
    pyoof ... out-of-focus holography package
    '''
    )
