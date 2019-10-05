#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from astropy import units as u
import pyoof
from pyoof import (
    aperture, telgeometry, fit_zpoly, extract_data_effelsberg, actuator
    )

# telescope = [blockage, delta, pr, name]
telescope = dict(
    effelsberg=[
        telgeometry.block_effelsberg,
        telgeometry.opd_effelsberg,
        50. * u.m,  # primary reflector radius
        'effelsberg'
        ],
    manual=[
        telgeometry.block_manual(
            pr=50 * u.m, sr=0 * u.m, a=0 * u.m, L=0 * u.m),
        telgeometry.opd_manual(Fp=30 * u.m, F=387.39435 * u.m),
        50. * u.m,  # primary reflector radius
        'effelsberg partial blockage'
        ]
    )


def fit_beam_effelsberg(pathfits):
    """
    Fits the beam from the OOF holography observations specifically for the
    Effelsberg telescope.
    """

    data_info, data_obs = extract_data_effelsberg(pathfits)

    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    fit_zpoly(
        data_info=data_info,
        data_obs=[beam_data, u_data, v_data],
        order_max=6,                         # it'll fit from 1 to order_max
        illum_func=aperture.illum_pedestal,  # or illum_gauss
        telescope=telescope['effelsberg'],
        fit_previous=True,                   # True is recommended
        resolution=2 ** 8,                   # standard is 2 ** 8
        box_factor=5,              # box_size = 5 * pr, better pixel resolution
        config_params_file=None,   # default or add path config_file.yaml
        make_plots=True,           # for now testing only the software
        verbose=2,
        work_dir=None
        # work_dir='/scratch/v/vanderli/cassane'
        )


if __name__ == '__main__':

    # natasha
    pth2data = '/home/tcassanelli/data/pyoof'
    fit_beam_effelsberg(
        pathfits=os.path.join(pth2data, 'S9mm_3824-3843_3C84_72deg_H6_BW.fits')
        )

    # mac
    # pth2data = '/Users/tomascassanelli/MPIfR/OOF/data/S9mm_noFEM/'
    # fit_beam_effelsberg(
    #     pathfits=os.path.join(pth2data, 'S9mm_3824-3843_3C84_72deg_H6_BW.fits')
    #     )

    # for n in range(1, 8):
    #     pyoof.plot_fit_path(
    #         path_pyoof='/Users/tomascassanelli/MPIfR/OOF/data/norm_test/norm',
    #         order=n,
    #         illum_func=aperture.illum_pedestal,
    #         telgeo=telescope['effelsberg'][:-1],
    #         resolution=2 ** 8,
    #         box_factor=5,
    #         angle=u.deg,
    #         plim=None,
    #         save=True
    #         )
    #     plt.close('all')
    # plt.show()
    
    # fit_beam_effelsberg('/Users/tomascassanelli/MPIfR/OOF/data/S9mm_noFEM/S9mm_3824-3843_3C84_72deg_H6_BW.fits')

    # path_pyoof_out= '/Users/tomascassanelli/MPIfR/OOF/data/S9mm_noFEM/pyoof_out/S9mm_3800-3807_3C84_48deg_H6_LON-073'

    # actuator.actuator_displacement(path_pyoof_out=path_pyoof_out, order=2)

    # fig = pyoof.actuator.plot_actuator_displacement(path_pyoof_out, 2, '', actuators=True, act_data=None)
    # plt.show()
    