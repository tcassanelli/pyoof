#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import os
import numpy as np
from astropy import units as apu
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
import pyoof

# Initial fits file configuration
n = 5                                         # initial order
N_K_coeff = (n + 1) * (n + 2) // 2            # total number of polynomials
c_dB = -14 * apu.dB                           # illumination taper
I_coeff = [1, c_dB, 2, 0 * apu.m, 0 * apu.m]  # illumination coefficients
K_coeff = np.array([0.1] * N_K_coeff)         # random Zernike circle coeff.
wavel = 0.0093685143125 * apu.m               # wavelenght
d_z = [-2.2, 0, 2.2] * apu.cm                 # radial offset

# Making example for the Effelsberg telescope
effelsberg_telescope = [
    pyoof.telgeometry.block_effelsberg(alpha=20 * apu.deg),  # blockage
    pyoof.telgeometry.opd_effelsberg,    # OPD function
    50. * apu.m,                         # primary reflector radius
    'effelsberg'                         # telescope name
    ]


# Generating temp file with pyoof fits
@pytest.fixture
def oof_work_dir(tmpdir_factory):

    tdir = str(tmpdir_factory.mktemp('pyoof'))

    pyoof.simulate_data_pyoof(
        K_coeff=K_coeff,
        I_coeff=I_coeff,
        wavel=wavel,
        d_z=d_z,
        illum_func=pyoof.aperture.illum_parabolic,
        telgeo=effelsberg_telescope[:-1],
        noise=0,
        resolution=2 ** 8,
        box_factor=5,
        work_dir=tdir
        )

    print('files directory: ', tdir)

    return tdir


def test_simulate_data_pyoof(oof_work_dir):

    data_info, data_obs = pyoof.extract_data_pyoof(
        os.path.join(oof_work_dir, 'data_generated', 'test000.fits')
        )
    [beam_data, u_data, v_data] = data_obs

    data_info_true, data_obs_true = pyoof.extract_data_pyoof(
        get_pkg_data_filename('data/data_simulated.fits')
        )
    [beam_data_true, u_data_true, v_data_true] = data_obs_true

    assert_allclose(beam_data, beam_data_true)
    assert_quantity_allclose(u_data, u_data_true)
    assert_quantity_allclose(v_data, v_data_true)

    for i in range(4, 7):
        assert_quantity_allclose(data_info[i], data_info_true[i])

    assert data_info[2:4] == data_info_true[2:4]
    assert data_info[7] == data_info_true[7]
