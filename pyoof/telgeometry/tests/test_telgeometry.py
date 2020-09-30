#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy import units as apu
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
import pyoof


# Basic data for the telescope
pr = 50. * apu.m
x = np.linspace(-pr, pr, 1000)
xx, yy = np.meshgrid(x, x)
d_z = 2.2 * apu.cm

# Other configurations
Fp = 20 * apu.m
F = 200 * apu.m
sr = 3.25 * apu.m
a = 1 * apu.m
L = pr - sr


def test_opd_effelsberg():

    _opd_effelsberg = pyoof.telgeometry.opd_effelsberg(x=xx, y=yy, d_z=d_z)

    opd_effelsberg_true = np.load(
        get_pkg_data_filename('data/opd_effelsberg.npy')
        )

    assert_quantity_allclose(
        _opd_effelsberg, Quantity(opd_effelsberg_true, apu.m)
        )


def test_opd_manual():
    _opd_manual = pyoof.telgeometry.opd_manual(Fp=Fp, F=F)(x=xx, y=yy, d_z=d_z)
    opd_manual_true = np.load(get_pkg_data_filename('data/opd_manual.npy'))
    assert_quantity_allclose(_opd_manual, Quantity(opd_manual_true, apu.m))


def test_block_effelsberg():

    _block_effelsberg = pyoof.telgeometry.block_effelsberg(
        alpha=20 * apu.deg)(x=xx, y=yy)

    block_effelsberg_true = np.load(
        get_pkg_data_filename('data/block_effelsberg.npy')
        )

    assert_allclose(_block_effelsberg, block_effelsberg_true)


def test_block_manual():

    _block_manual = pyoof.telgeometry.block_manual(
        pr=pr, sr=sr, a=a, L=L
        )(x=xx, y=yy)

    block_manual_true = np.load(get_pkg_data_filename('data/block_manual.npy'))
    assert_allclose(_block_manual, block_manual_true)
