#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_equal, assert_allclose
import pyoof


def test_cart2pol():

    with NumpyRNGContext(0):
        x = np.random.uniform(-1, 1, 5)
        y = np.random.uniform(-1, 1, 5)

    rho, theta = pyoof.cart2pol(x, y)

    rho_true = np.array([
        0.30768718, 0.44811525, 0.81005283, 0.93166014, 0.27867163
        ])
    theta_true = np.array([
        1.24792264, -0.28229117, 1.31427197, 1.47429564, -2.15067483
        ])

    assert_allclose(rho, rho_true)
    assert_allclose(theta, theta_true)


def test_wavevector2degrees():

    wavel = np.linspace(0.005, 0.014, 10)

    with NumpyRNGContext(0):
        u = np.sort(np.random.uniform(-2.5, 2.5, 10))

    wavevector_degrees = pyoof.wavevector2degrees(u, wavel)

    wavevector_degrees_true = np.array([
        -0.16695773, -0.13122773, -0.12515963, 0.10286468, 0.12585635,
        0.29439539, 0.45975143, 0.73976655, 1.45905107, 1.85961435
        ])

    assert_allclose(wavevector_degrees, wavevector_degrees_true)


def test_wavevector2radians():

    wavel = np.linspace(0.005, 0.014, 10)

    with NumpyRNGContext(0):
        u = np.sort(np.random.uniform(-2.5, 2.5, 10))

    wavevector_radians = pyoof.wavevector2radians(u, wavel)

    wavevector_radians_true = np.array([
        -0.002913962029, -0.00229035602, -0.002184447606, 0.00179532732,
        0.002196607677, 0.005138168804, 0.008024176219, 0.012911361982,
        0.025465245051, 0.032456393235
        ])

    assert_allclose(wavevector_radians, wavevector_radians_true)


def test_co_matrices():

    # importing tests files
    res = np.load(
        get_pkg_data_filename('data/res_co_matrices.npy')
        )
    jac = np.load(
        get_pkg_data_filename('data/jac_co_matrices.npy')
        )
    cov_true = np.load(
        get_pkg_data_filename('data/cov_co_matrices.npy')
        )
    corr_true = np.load(
        get_pkg_data_filename('data/corr_co_matrices.npy')
        )

    cov, corr = pyoof.co_matrices(res, jac, 1)

    assert_allclose(cov, cov_true)
    assert_allclose(corr, corr_true)


def test_line_equation():

    x = np.linspace(-1, 1, 10)
    y1 = pyoof.line_equation((0, 0), (1, 1,), x)
    y1_true = x

    with NumpyRNGContext(0):
        p1 = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        p2 = (np.random.uniform(0, 1), np.random.uniform(0, 1))

    y2 = pyoof.line_equation(p1, p2, x)
    y2_true = np.array([
        5.604404226951, 4.902904538006, 4.20140484906, 3.499905160114,
        2.798405471169, 2.096905782223, 1.395406093278, 0.693906404332,
        -0.007593284613, -0.709092973559
        ])

    assert_allclose(y1, y1_true)
    assert_allclose(y2, y2_true)


def test_rms():

    x1 = np.array([1] * 10 + [-1] * 10)
    rms1 = pyoof.rms(x1)
    rms1_true = 1.0

    with NumpyRNGContext(0):
        x2 = np.random.uniform(-20, 20, 5)
    rms2 = pyoof.rms(x2)
    rms2_true = 4.6335342124813295

    assert_equal(rms1, rms1_true)
    assert_equal(rms2, rms2_true)
