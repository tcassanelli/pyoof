#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import os
import numpy as np
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
from numpy.testing import assert_array_almost_equal, assert_allclose
import pyoof


# Initial fits file configuration
n = 4                                           # initial order
N_K_coeff = (n + 1) * (n + 2) // 2 - 1          # total numb. polynomials
c_dB = np.random.randint(-21, -10)              # illumination taper
I_coeff = np.array([1, c_dB, 0, 0])             # illumination coefficients

# Zernike circle polynomial coeff, first one kept fixed.
K_coeff = np.hstack((0, np.random.normal(0., .08, N_K_coeff)))
wavel = 0.0093685143125                         # wavelenght in m
d_z = [0.022, 0, -0.022]                        # radial offset
params_true = np.hstack((I_coeff, K_coeff))

noise_level = 1e1                               # noise added to gen data

effelsberg_telescope = [
    pyoof.telgeometry.block_effelsberg,         # blockage distribution
    pyoof.telgeometry.opd_effelsberg,           # OPD function
    50.,                                        # primary reflector radius m
    'effelsberg'                                # telescope name
    ]

# Least squares minimization
method = 'trf'
resolution = 2 ** 8
box_factor = 5


# Generating temp file with pyoof fits and pyoof_out
@pytest.fixture(scope='module')
def oof_work_dir(tmpdir_factory):

    tdir = str(tmpdir_factory.mktemp('OOF'))

    pyoof.beam_generator(
        params=params_true,
        wavel=wavel,
        d_z=d_z,
        telgeo=effelsberg_telescope[:-1],
        illum_func=pyoof.aperture.illum_pedestal,
        noise=noise_level,
        resolution=resolution,
        box_factor=box_factor,
        work_dir=tdir
        )

    print('files directory: ', tdir)

    # Reading the generated data
    pathfits = os.path.join(tdir, 'data_generated/test000.fits')
    data_info, data_obs = pyoof.extract_data_pyoof(pathfits)

    pyoof.fit_beam(
        data_info=data_info,
        data_obs=data_obs,
        method=method,
        order_max=n,
        illum_func=pyoof.aperture.illum_pedestal,
        telescope=effelsberg_telescope,
        fit_previous=True,
        resolution=resolution,
        box_factor=box_factor,
        config_params_file=None,
        make_plots=False,
        verbose=0,
        work_dir=tdir
        )

    return tdir


def test_fit_beam(oof_work_dir):

    # To see if we are in the right temp directory
    print(os.listdir(oof_work_dir))

    # lets compare the params from the last order
    params = ascii.read(
        os.path.join(
            oof_work_dir, 'pyoof_out', 'test000-000',
            'fitpar_n{}.csv'.format(n))
        )

    parfit = params['parfit']

    assert_array_almost_equal(parfit, params_true, decimal=2)
