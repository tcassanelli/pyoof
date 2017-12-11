#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_equal, assert_allclose
import pyoof


# Basic data for the telescope
pr = 50.  # m
x = np.linspace(-pr, pr, 1e3)
xx, yy = np.meshgrid(x, x)
d_z = .022  # m

# Other configurations
Fp = 20
F = 200
sr = 3.25
a = 1
L = pr - sr


def test_opd_effelsberg():

    _opd_effelsberg = pyoof.telgeometry.opd_effelsberg(x=xx, y=yy, d_z=d_z)

    opd_effelsberg_true = np.load(
        get_pkg_data_filename('data/opd_effelsberg.npy')
        )

    assert_allclose(_opd_effelsberg, opd_effelsberg_true)


def test_opd_manual():

    _opd_manual = pyoof.telgeometry.opd_manual(Fp=Fp, F=F)(x=xx, y=yy, d_z=d_z)

    opd_manual_true = np.load(
        get_pkg_data_filename('data/opd_manual.npy')
        )

    assert_allclose(_opd_manual, opd_manual_true)


def test_block_effelsberg():

    _block_effelsberg = pyoof.telgeometry.block_effelsberg(x=xx, y=yy)

    block_effelsberg_true = np.load(get_pkg_data_filename(
        'data/block_effelsberg.npy')
        )

    assert_allclose(_block_effelsberg, block_effelsberg_true)


def test_block_manual():

    _block_manual = pyoof.telgeometry.block_manual(
        pr=pr, sr=sr, a=a, L=L
        )(x=xx, y=yy)

    block_manual_true = np.load(
        get_pkg_data_filename('data/block_manual.npy')
        )

    assert_allclose(_block_manual, block_manual_true)
