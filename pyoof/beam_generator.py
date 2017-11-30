#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import os
from scipy.constants import c as light_speed
from astropy.io import fits
from .aperture import radiation_pattern
from .math_functions import wavevector2radians

__all__ = ['beam_generator']


def beam_generator(
    params, freq, d_z, telgeo, illum_func, noise, resolution, box_factor
        ):
    """
    Routine to generate data and test the pyoof package algorithm. It has the
    default setting for the pyoof package OOF holography input.
    """

    wavel = light_speed / freq  # wavelength
    bw = 1.22 * wavel / (2 * telgeo[2])  # Beamwidth
    size_in_bw = 8 * bw
    plim_u = [-size_in_bw, size_in_bw]  # radians
    plim_v = [-size_in_bw, size_in_bw]

    I_coeff, K_coeff = params[:4], params[4:]

    # Generating power pattern and spatial frequencies
    u, v, P = [], [], []
    for i in range(3):
        _u, _v, _radiation = radiation_pattern(
            K_coeff=K_coeff,
            I_coeff=I_coeff,
            d_z=d_z[i],
            wavel=wavel,
            illum_func=illum_func,
            telgeo=telgeo,
            resolution=resolution,
            box_factor=box_factor
            )

        _u = wavevector2radians(_u, wavel)  # radians
        _v = wavevector2radians(_v, wavel)

        uu, vv = np.meshgrid(_u, _v)
        power_pattern = np.abs(_radiation) ** 2

        # trim the power pattern
        power_pattern[plim_u[0] > uu] = np.nan
        power_pattern[plim_u[1] < uu] = np.nan
        power_pattern[plim_v[0] > vv] = np.nan
        power_pattern[plim_v[1] < vv] = np.nan

        power_trim_1d = power_pattern[~np.isnan(power_pattern)]
        size_trim = int(np.sqrt(power_trim_1d.size))  # new size of the box

        # Box to be trimmed in uu and vv meshed arrays
        box = [(plim_u[0] < uu) & (plim_u[1] > uu) & (plim_v[0] < vv) & (
            plim_v[1] > vv)]

        # reshaping trimmed arrys
        power_trim = power_trim_1d.reshape(size_trim, size_trim)
        u_trim = uu[box].reshape(size_trim, size_trim)
        v_trim = vv[box].reshape(size_trim, size_trim)

        u.append(u_trim)
        v.append(v_trim)
        P.append(power_trim)

    # adding noise!
    if noise == 0:
        power_noise = np.array(P)
    else:
        power_noise = (
            np.array(P) + np.random.normal(0., noise, np.array(P).shape)
            )
    # power_norm = [P[i] / P[i].max() for i in range(3)]
    # power_noise = (
    #     np.array(power_norm) +
    #     np.random.normal(0., noise, np.array(power_norm).shape)
    #     )

    u_to_save = [u[i].flatten() for i in range(3)]
    v_to_save = [v[i].flatten() for i in range(3)]
    p_to_save = [power_noise[i].flatten() for i in range(3)]

    # Writing default fits file for OOF observations
    table_hdu0 = fits.BinTableHDU.from_columns([
        fits.Column(name='U', format='E', array=u_to_save[0]),
        fits.Column(name='V', format='E', array=v_to_save[0]),
        fits.Column(name='BEAM', format='E', array=p_to_save[0])
        ])

    table_hdu1 = fits.BinTableHDU.from_columns([
        fits.Column(name='U', format='E', array=u_to_save[1]),
        fits.Column(name='V', format='E', array=v_to_save[1]),
        fits.Column(name='BEAM', format='E', array=p_to_save[1])
        ])

    table_hdu2 = fits.BinTableHDU.from_columns([
        fits.Column(name='U', format='E', array=u_to_save[2]),
        fits.Column(name='V', format='E', array=v_to_save[2]),
        fits.Column(name='BEAM', format='E', array=p_to_save[2])
        ])

    table_hdu0.update_ext_name('MINUS OOF')
    table_hdu1.update_ext_name('ZERO OOF')
    table_hdu2.update_ext_name('PLUS OOF')

    # storing data
    current_path = os.getcwd()
    if not os.path.exists(current_path + '/data_generated'):
        os.makedirs(current_path + '/data_generated')

    for j in ["%03d" % i for i in range(101)]:
        name_file = current_path + '/data_generated/test{}.fits'.format(j)
        if not os.path.exists(name_file):

            prihdr = fits.Header()
            prihdr['FREQ'] = freq
            prihdr['WAVEL'] = wavel
            prihdr['MEANEL'] = 0
            prihdr['OBJECT'] = 'test{}'.format(j)
            prihdr['DATE_OBS'] = 'test{}'.format(j)
            prihdr['COMMENT'] = 'Generated data pyoof package'
            prihdr['AUTHOR'] = 'Tomas Cassanelli'
            prihdu = fits.PrimaryHDU(header=prihdr)
            pyoof_fits = fits.HDUList(
                [prihdu, table_hdu0, table_hdu1, table_hdu2]
                )

            for i in range(3):
                pyoof_fits[i + 1].header['DZ'] = d_z[i]

            pyoof_fits.writeto(name_file, clobber=True)

            break

    return pyoof_fits
