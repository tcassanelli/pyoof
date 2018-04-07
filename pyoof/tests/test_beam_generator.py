#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import os
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
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


# Generating temp file with pyoof fits
@pytest.fixture
def oof_work_dir(tmpdir_factory):

    tdir = str(tmpdir_factory.mktemp('pyoof'))

    pyoof.beam_generator(
        params=np.hstack((I_coeff, K_coeff)),
        wavel=wavel,
        d_z=d_z,
        telgeo=effelsberg_telescope[:-1],
        illum_func=pyoof.aperture.illum_pedestal,
        noise=0,
        resolution=2 ** 8,
        box_factor=5,
        work_dir=tdir
        )

    print('files directory: ', tdir)

    return tdir


def test_beam_generator(oof_work_dir):

    data_info, data_obs = pyoof.extract_data_pyoof(
        os.path.join(oof_work_dir, 'data_generated', 'test000.fits')
        )

    data_info_true, data_obs_true = pyoof.extract_data_pyoof(
        get_pkg_data_filename('data/beam_generator.fits')
        )

    assert_allclose(data_obs, data_obs_true)
    assert data_info[2:] == data_info_true[2:]
