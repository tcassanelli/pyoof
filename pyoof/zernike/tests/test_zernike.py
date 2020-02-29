#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
import pyoof

# Basic mesh for polynomials
x = np.linspace(-1, 1, 1000)
xx, yy = np.meshgrid(x, x)
r, t = pyoof.cart2pol(xx, yy)


def test_R():

    _R = pyoof.zernike.R(n=6, m=4, rho=r)

    R_true = np.load(get_pkg_data_filename('data/R.npy'))

    assert_allclose(_R, R_true)


def test_U():

    _U_cos = pyoof.zernike.U(n=7, l=-5, rho=r, theta=t)
    _U_sin = pyoof.zernike.U(n=7, l=5, rho=r, theta=t)

    U_cos_true = np.load(get_pkg_data_filename('data/Ucos.npy'))
    U_sin_true = np.load(get_pkg_data_filename('data/Usin.npy'))

    with pytest.raises(TypeError):
        pyoof.zernike.U(n=7.1, l=-5, rho=r, theta=t)

    with pytest.raises(TypeError):
        pyoof.zernike.U(n=7, l=-5.1, rho=r, theta=t)

    assert_allclose(_U_cos, U_cos_true)
    assert_allclose(_U_sin, U_sin_true)
