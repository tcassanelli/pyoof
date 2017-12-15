#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
import pyoof

# Basic mesh for polynomials
x = np.linspace(-1, 1, 1e3)
xx, yy = np.meshgrid(x, x)
r, t = pyoof.cart2pol(xx, yy)


def test_R():

    _R = pyoof.zernike.R(n=6, m=4, rho=r)

    R_true = np.load(get_pkg_data_filename('data/R.npy'))

    assert_allclose(_R, R_true)


def test_U():

    _U = pyoof.zernike.U(n=7, l=-5, rho=r, theta=t)

    U_true = np.load(get_pkg_data_filename('data/U.npy'))

    assert_allclose(_U, U_true)
