#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import os
import numpy as np
from astropy import units as apu
from astropy.io import ascii
from numpy.testing import assert_allclose
import pyoof


# Initial fits file configuration
n = 5                                           # initial order
N_K_coeff = (n + 1) * (n + 2) // 2 - 1          # total numb. polynomials
c_dB = np.random.randint(-21, -10) * apu.dB     # illumination taper
wavel = 0.0093685143125 * apu.m                 # wavelength

plus_minus = np.random.normal(0.025, 0.005)
d_z = [plus_minus, 0, -plus_minus] * apu.m      # radial offset

noise_level = .01                               # noise added to gen data

effelsberg_telescope = [
    pyoof.telgeometry.block_effelsberg,         # blockage distribution
    pyoof.telgeometry.opd_effelsberg,           # OPD function
    50. * apu.m,                                # primary reflector radius m
    'effelsberg'                                # telescope name
    ]

# Least squares minimization
resolution = 2 ** 8
box_factor = 5

# True values to be compared at the end
K_coeff_true = np.hstack((0, np.random.normal(0., .08, N_K_coeff)))
I_coeff_true = [1, c_dB, 0 * apu.m, 0 * apu.m]  # illumination coefficients

I_coeff_true_dimensionless = [
    I_coeff_true[0]] + [I_coeff_true[i].value for i in range(1, 4)]


# Generating temp file with pyoof fits and pyoof_out
@pytest.fixture()
def oof_work_dir(tmpdir_factory):

    tdir = str(tmpdir_factory.mktemp('pyoof'))

    pyoof.simulate_data_pyoof(
        K_coeff=K_coeff_true,
        I_coeff=I_coeff_true,
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
    pathfits = os.path.join(tdir, 'data_generated', 'test000.fits')
    data_info, data_obs = pyoof.extract_data_pyoof(pathfits)

    for i in range(3):
        print('snr:', data_obs[0][i].max() / data_obs[0][i].std())

    pyoof.fit_zpoly(
        data_info=data_info,
        data_obs=data_obs,
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
    print('temp directory: ', os.listdir(oof_work_dir))

    # lets compare the params from the last order
    fit_pars = os.path.join(
        oof_work_dir, 'pyoof_out', 'test000-000', 'fitpar_n{}.csv'.format(n)
        )

    params = ascii.read(fit_pars)['parfit']
    assert_allclose(params[4:], K_coeff_true, rtol=1e-7, atol=1e-1)
    assert_allclose(params[:4], I_coeff_true_dimensionless, rtol=1e-5, atol=1)
