#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.misc import NumpyRNGContext
from astropy import units as apu
import pyoof


def test_str2LaTeX():
    LaTeX_string = pyoof.str2LaTeX('1_2_3_4_5')
    assert LaTeX_string == '1\\_2\\_3\\_4\\_5'


def test_uv_ratio():

    width_true, height_true = 5.65260911, 5
    with NumpyRNGContext(0):
        u = np.random.uniform(-1, 1, 5) * apu.deg
        v = np.random.uniform(-1, 1, 5) * apu.deg

    width, height = pyoof.uv_ratio(u, v)

    assert_allclose(width, width_true)
    assert_allclose(height, height_true)

