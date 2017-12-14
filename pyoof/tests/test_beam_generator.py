#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_equal, assert_allclose
import pyoof


# Initial fits file configuration
n = 7                                    # initial order
N_K_coeff = (n + 1) * (n + 2) // 2       # total numb. polynomials
c_dB = -14                               # illumination taper
I_coeff = np.array([1, c_dB, 0, 0])      # illumination coefficients
K_coeff = np.array([0.1] * N_K_coeff)    # random Zernike circle coeff.
wavel = 0.0093685143125                  # wavelenght in m
d_z = [0.022, 0, -0.022]                 # radial offset

# Making example for the Effelsberg telescope
effelsberg_telescope = [
    pyoof.telgeometry.block_effelsberg,  # blockage distribution
    pyoof.telgeometry.opd_effelsberg,    # OPD function
    50.,                                 # primary reflector radius m
    'effelsberg'                         # telescope name
    ]


def test_beam_generator():

    hdulist = pyoof.beam_generator(
        params=np.hstack((I_coeff, K_coeff)),
        wavel=wavel,
        d_z=d_z,
        telgeo=effelsberg_telescope[:-1],
        illum_func=pyoof.aperture.illum_pedestal,
        noise=0,
        resolution=2 ** 8,
        box_factor=5,
        save=False,
        work_dir=None
        )

    beam_data = [hdulist[i].data['BEAM'] for i in range(1, 4)]
    u_data = [hdulist[i].data['U'] for i in range(1, 4)]
    v_data = [hdulist[i].data['V'] for i in range(1, 4)]

    data_obs_true = pyoof.extract_data_pyoof(
        get_pkg_data_filename('data/beam_generator.fits')
        )[1]

    [beam_data_true, u_data_true, v_data_true] = data_obs_true

    assert_allclose(beam_data, beam_data_true)
    assert_allclose(u_data, u_data_true)
    assert_allclose(v_data, v_data_true)
