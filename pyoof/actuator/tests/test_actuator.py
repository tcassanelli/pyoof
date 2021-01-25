#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import os
import numpy as np
from astropy import units as apu
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
from numpy.testing import assert_allclose
from pyoof.actuator import EffelsbergActuator

# Basic data for the telescope
resolution = 1000
wavel = 7 * apu.mm
n = 5

actuator = EffelsbergActuator(
    frequency=34.75 * apu.GHz,
    nrot=1,
    sign=-1,
    order=n,
    sr=3.25 * apu.m,
    pr=50 * apu.m,
    resolution=resolution,
    )


@pytest.fixture()
def pyoof_tmp_dir(tmpdir_factory):

    tdir = str(tmpdir_factory.mktemp('actuator'))

    alpha_lookup, actuator_sr_lookup = actuator.read_lookup(interp=False)
    actuator.write_lookup(
        fname=os.path.join(tdir, 'lookup_test.data'),
        actuator_sr=actuator_sr_lookup
        )

    return tdir


def test_read_lookup():
    alpha_lookup_true = [7, 10, 20, 30, 32, 40, 50, 60, 70, 80, 90] * apu.deg
    assert_quantity_allclose(actuator.alpha_lookup, alpha_lookup_true)


def test_write_lookup(pyoof_tmp_dir):
    print('temp directory:', os.listdir(pyoof_tmp_dir))

    fh_lookup_true = open(
        get_pkg_data_filename('../../data/lookup_effelsberg.data')
        )
    fh_lookup = open(os.path.join(pyoof_tmp_dir, 'lookup_test.data'))

    read_lines_true = fh_lookup_true.readlines()
    read_lines = fh_lookup.readlines()

    assert len(read_lines_true) == len(read_lines)

    for line, line_true in zip(read_lines, read_lines_true):
        assert line == line_true
