#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
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


def test_read_lookup():

    alpha_lookup_true = [7, 10, 20, 30, 32, 40, 50, 60, 70, 80, 90] * apu.deg

    assert_quantity_allclose(actuator.alpha_lookup, alpha_lookup_true)


# this tests is too long!
# def test_interpolation():

#     G_coeff_lookup = actuator.fit_all(
#         phase_pr=actuator.phase_pr_lookup,
#         alpha=actuator.alpha_lookup
#         )[0]

#     # cheking whether the software solves G and writes correctly
#     phase_pr_lookup_retreived = actuator.generate_phase_pr(
#         G_coeff=G_coeff_lookup,
#         alpha=actuator.alpha_lookup
#         ).to_value(apu.rad)

#     phase_pr_lookup_true = actuator.phase_pr_lookup.to_value(apu.rad)

#     assert_allclose(
#         actual=phase_pr_lookup_retreived,
#         desired=phase_pr_lookup_true,
#         rtol=1e-5,
#         atol=3
#         )
