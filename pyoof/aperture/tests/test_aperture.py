#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
from numpy.testing import assert_equal, assert_allclose
import pyoof


def test_e_rse():

    with NumpyRNGContext(0):
        phase = np.sort(np.random.uniform(-2.5, 2.5, (5, 5)))

    e_rse = pyoof.aperture.e_rse(phase)

    assert_allclose(e_rse, 0.110794937514)


def test_illum_pedestal():

    pr = 50
    x = np.linspace(-pr, pr)
    y = x
    xx, yy = np.meshgrid(x, y)
    I_coeff = [1, -14, 0, 0]

    _illum_pedestal = pyoof.aperture.illum_pedestal(xx, yy, I_coeff, pr)
    illum_pedestal_true = np.load(
        get_pkg_data_filename('data/illum_pedestal.npy')
        )

    assert_allclose(_illum_pedestal, illum_pedestal_true)


# def test_wavefront(rho, theta, K_coeff):

#     x = np.linspace(-50, 50, 1e3)
#     xx, yy = np.meshgrid(x, x)
#     rho, theta = pyoof.cart2pol(xx, yy)

#     with NumpyRNGContext(0):
#         K_coeff = np.sort(np.random.uniform(-2.5, 2.5, 21)


#     W = aperture.wavefront(rho, theta, K_coeff)
#     W_true = 1
