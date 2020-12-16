#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import glob
from astropy import units as u
from pyoof import aperture, telgeometry, fit_zpoly, extract_data_effelsberg

# telescope = [blockage, delta, pr, name]
pr = 50 * u.m
telescope = dict(
    effelsberg_20deg=[
        telgeometry.block_effelsberg(alpha=20 * u.deg),
        telgeometry.opd_effelsberg,
        pr,
        'effelsberg (20 deg blockage)'
        ],
    effelsberg_10deg=[
        telgeometry.block_effelsberg(alpha=10 * u.deg),
        telgeometry.opd_effelsberg,
        pr,
        'effelsberg (10 deg blockage)'
        ],
    effelsberg_0deg=[
        telgeometry.block_effelsberg(alpha=0 * u.deg),
        telgeometry.opd_effelsberg,
        pr,
        'effelsberg (0 deg blockage)'
        ],
    effelsberg_sr_only=[
        telgeometry.block_manual(
            pr=50 * u.m, sr=3.25 * u.m, a=0 * u.m, L=0 * u.m),
        telgeometry.opd_effelsberg,
        pr,
        'effelsberg (sub-reflector only blockage)'
        ],
    effelsberg_empty=[
        telgeometry.block_manual(
            pr=50 * u.m, sr=0 * u.m, a=0 * u.m, L=0 * u.m),
        telgeometry.opd_effelsberg,
        pr,
        'effelsberg (sub-reflector only blockage)'
        ]
    )


def compute_phase_error(pathfits, order_max):
    """
    Uses fit_zpoly and calculates the actuators at the Effelsberg telescope.
    """

    data_info, data_obs = extract_data_effelsberg(pathfits)

    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    for configuration in telescope.keys():
    # for configuration in ['effelsberg_empty']:
        fit_zpoly(
            data_info=data_info,
            data_obs=[beam_data, u_data, v_data],
            order_max=order_max,
            illum_func=aperture.illum_pedestal,
            # illum_func=aperture.illum_gauss,
            telescope=telescope[configuration],
            fit_previous=True,                   # True is recommended
            resolution=2 ** 8,         # standard is 2 ** 8
            box_factor=5,              # box_size = 5 * pr
            config_params_file=None,   # default or add path config_file.yaml
            make_plots=True,           # for now testing only the software
            verbose=2,
            work_dir=None
            # work_dir='/scratch/v/vanderli/cassane'
            )


if __name__ == '__main__':
    pth2data = '/home/v/vanderli/cassane/data/pyoof_Dec2020/*.fits'
    # pth2data = '/Users/tomascassanelli/MPIfR/OOF/data2020/Sep2020/*.fits'
    # pth2data = '/scratch/v/vanderli/cassane/pyoof_data/Sep2020/*.fits'
    files = glob.glob(pth2data)

    for _f in files:
        compute_phase_error(pathfits=_f, order_max=6)
