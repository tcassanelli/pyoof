#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from scipy.constants import c as light_speed
from astropy.io import fits


# Auxiliar functions to handle files, strings and data


__all__ = [
    'extract_data_fits', 'str2LaTeX',
    ]


def find_name_path(path):
    head, tail = os.path.split(path)
    return head, tail


def extract_data_fits(pathfits):
    # Opening fits file with astropy
    hdulist = fits.open(pathfits)

    # Observation frequency
    freq = hdulist[0].header['FREQ']  # Hz
    wavel = light_speed / freq

    # Mean elevation
    meanel = hdulist[0].header['MEANEL']  # Degrees

    # name of the fit file to fit
    name = find_name_path(pathfits)[1][:-5]

    beam_data = [hdulist[i].data['fnu'] for i in range(1, 4)][::-1]
    u_data = [hdulist[i].data['DX'] for i in range(1, 4)][::-1]
    v_data = [hdulist[i].data['DY'] for i in range(1, 4)][::-1]
    d_z_m = [hdulist[i].header['DZ'] for i in range(1, 4)][::-1]

    # Permuting the position to provide same as main_functions
    beam_data.insert(1, beam_data.pop(2))
    u_data.insert(1, u_data.pop(2))
    v_data.insert(1, v_data.pop(2))
    d_z_m.insert(1, d_z_m.pop(2))

    # path or directory where the fits file is located
    pthto = find_name_path(pathfits)[0]

    return name, freq, wavel, d_z_m, meanel, pthto, [beam_data, u_data, v_data]


def str2LaTeX(name):
    # LaTeX problem with underscore _ -> \_
    string_list = list(name)
    for idx, string in enumerate(string_list):
        if string_list[idx] == '_':
            string_list[idx] = '\_'
    name_LaTeX = ''.join(string_list)
    return name_LaTeX


def line_func(P1, P2, x):
    (x1, y1) = P1
    (x2, y2) = P2
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1
