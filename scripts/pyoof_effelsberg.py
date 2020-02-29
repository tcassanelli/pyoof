#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import glob
from astropy import units as u
from pyoof import (
    aperture, telgeometry, fit_zpoly, extract_data_effelsberg, actuator
    )

# telescope = [blockage, delta, pr, name]
telescope = dict(
    effelsberg=[
        telgeometry.block_effelsberg,
        telgeometry.opd_effelsberg,
        50. * u.m,
        'effelsberg'
        ],
    manual=[
        telgeometry.block_manual(
            pr=50 * u.m, sr=0 * u.m, a=0 * u.m, L=0 * u.m),
        telgeometry.opd_manual(Fp=30 * u.m, F=387.39435 * u.m),
        50. * u.m,
        'effelsberg partial blockage'
        ]
    )


def compute_phase_error(pathfits, order_max):
    """
    Uses fit_zpoly and calculates the actuators at the Effelsberg telescope.
    """

    data_info, data_obs = extract_data_effelsberg(pathfits)

    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    fit_zpoly(
        data_info=data_info,
        data_obs=[beam_data, u_data, v_data],
        order_max=order_max,                 # it'll fit from 1 to order_max
        illum_func=aperture.illum_pedestal,  # or illum_gauss
        telescope=telescope['effelsberg'],
        fit_previous=True,                   # True is recommended
        resolution=2 ** 8,                   # standard is 2 ** 8
        box_factor=5,              # box_size = 5 * pr, better pixel resolution
        config_params_file=None,   # default or add path config_file.yaml
        make_plots=True,           # for now testing only the software
        verbose=2,
        # work_dir=None
        work_dir='/scratch/v/vanderli/cassane'
        )

    num_list = ["%03d" % i for i in range(101)]
    for j in range(len(num_list)):
        path_pyoof_out = os.path.join(
            pthto, 'pyoof_out', name + '-' + num_list[j]
            )
        if not os.path.exists(path_pyoof_out):
            path_pyoof_out = os.path.join(
                pthto, 'pyoof_out', name + '-' + num_list[j - 1]
                )
            break

    for order in range(1, order_max + 1):
        actuator.actuator_displacement(
            path_pyoof_out=path_pyoof_out,
            order=order,
            edge=None,
            make_plots=True
            )


if __name__ == '__main__':

    # pth2data = '/home/tcassanelli/data/pyoof'                    # natasha
    # pth2data = '/Users/tomascassanelli/MPIfR/OOF/data/S7mm_FEM'  # local
    pth2data = '/home/v/vanderli/cassane/data/pyoof/*.fits'        # scinet
    files = glob.glob(pth2data)

    for _f in file:
        compute_phase_error(
            pathfits=files,
            order_max=6
            )
