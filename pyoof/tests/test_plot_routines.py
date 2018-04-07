#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import os
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
from astropy.utils.misc import NumpyRNGContext
import pyoof


# Initial fits file configuration
n = 6                                    # initial order
N_K_coeff = (n + 1) * (n + 2) // 2       # total numb. polynomials
wavel = 0.0093685143125                  # wavelenght in m

with NumpyRNGContext(0):
    # random Zernike circle coeff.
    K_coeff = np.hstack((np.random.normal(0., .08, N_K_coeff)))
    plus_minus = np.random.normal(0.025, 0.005)
    c_dB = np.random.randint(-21, -10)

I_coeff = np.array([1, c_dB, 0, 0])
d_z = [plus_minus, 0, -plus_minus]       # radial offset

effelsberg_telescope = [
    pyoof.telgeometry.block_effelsberg,  # blockage distribution
    pyoof.telgeometry.opd_effelsberg,    # OPD function
    50.,                                 # primary reflector radius m
    'effelsberg'                         # telescope name
    ]

print('K_coeff: ', K_coeff)
print('d_z: ', d_z)

# Least squares minimization
resolution = 2 ** 8
box_factor = 5


@pytest.mark.mpl_image_compare(
    baseline_dir='data',
    filename='plot_phase.pdf'
    )
def test_plot_phase():

    fig_phase = pyoof.plot_phase(
        K_coeff=K_coeff, notilt=False, pr=50, title='plot phase'
        )
    return fig_phase


@pytest.mark.mpl_image_compare(
    baseline_dir='data',
    filename='plot_beam.pdf'
    )
def test_plot_beam():

    fig_beam = pyoof.plot_beam(
        params=np.hstack((I_coeff, K_coeff)),
        d_z=d_z,
        wavel=wavel,
        illum_func=pyoof.aperture.illum_pedestal,
        telgeo=effelsberg_telescope[:-1],
        resolution=resolution,
        box_factor=box_factor,
        plim_rad=None,
        angle='degrees',
        title='plot beam'
        )

    return fig_beam


@pytest.mark.mpl_image_compare(
    baseline_dir='data',
    filename='plot_data.pdf'
    )
def test_plot_data():

    [beam, u, v] = pyoof.extract_data_pyoof(
        get_pkg_data_filename('data/beam_generator.fits')
        )[1]

    fig_data = pyoof.plot_data(
        u_data=u,
        v_data=v,
        beam_data=beam,
        d_z=d_z,
        angle='degrees',
        title='plot data',
        res_mode=False
        )

    return fig_data


@pytest.fixture()
def oof_work_dir(tmpdir_factory):

    tdir = str(tmpdir_factory.mktemp('pyoof'))

    pathfits = get_pkg_data_filename('data/beam_generator.fits')
    data_info, data_obs = pyoof.extract_data_pyoof(pathfits)

    pyoof.fit_beam(
        data_info=data_info,
        data_obs=data_obs,
        method='trf',
        order_max=3,
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

    print('files directory: ', tdir)

    return tdir


@pytest.mark.mpl_image_compare(
    baseline_dir='data',
    filename='plot_variance.pdf'
    )
def test_plot_variance(oof_work_dir):

    path_pyoof = os.path.join(oof_work_dir, 'pyoof_out/beam_generator-000')
    fig_variance = pyoof.plot_variance(
        matrix=np.genfromtxt(os.path.join(path_pyoof, 'cov_n3.csv')),
        order=3,
        diag=True,
        illumination='pedestal',
        cbtitle='plot variance',
        title='plot variance'
        )

    return fig_variance
