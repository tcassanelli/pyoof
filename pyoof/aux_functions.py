#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
from scipy.constants import c as light_speed
from astropy.io import fits
import numpy as np
from astropy.io import ascii

__all__ = [
    'extract_data_effelsberg', 'str2LaTeX', 'store_csv', 'store_ascii'
    ]


def extract_data_effelsberg(pathfits):
    """
    Extracts the necessary data for the pyoof package, for Effelsberg
    telescope data only.

    Parameters
    ----------
    pathfits : str
        Path to the fits file that contains the three beam maps precalibrated,
        plus some other important parameter for the fit.

    Returns
    -------
    data_info : list
        It contains all extra data besides the beam map.
        data_info = [name, pthto, freq, wavel, d_z, meanel]

    data_obs : list
        It contains the main data for the leat squares optimisation.
        data_obs = [beam_data, u_data, v_data]
    """

    # Opening fits file with astropy
    hdulist = fits.open(pathfits)

    # Observation frequency
    freq = hdulist[0].header['FREQ']  # Hz
    wavel = light_speed / freq

    # Mean elevation
    meanel = hdulist[0].header['MEANEL']  # Degrees

    # name of the fit file to fit
    name = os.path.split(pathfits)[1][:-5]

    beam_data = [hdulist[i].data['fnu'] for i in range(1, 4)][::-1]
    u_data = [hdulist[i].data['DX'] for i in range(1, 4)][::-1]
    v_data = [hdulist[i].data['DY'] for i in range(1, 4)][::-1]
    d_z = [hdulist[i].header['DZ'] for i in range(1, 4)][::-1]

    # Permuting the position to provide same as main_functions
    beam_data.insert(1, beam_data.pop(2))
    u_data.insert(1, u_data.pop(2))
    v_data.insert(1, v_data.pop(2))
    d_z.insert(1, d_z.pop(2))

    # path or directory where the fits file is located
    pthto = os.path.split(pathfits)[0]

    data_info = [name, pthto, freq, wavel, d_z, meanel]
    data_obs = [beam_data, u_data, v_data]

    return data_info, data_obs


def str2LaTeX(python_string):
    """
    Function that solves the underscore problem in a python string to LaTeX
    it changes it from _ -> \_ symbol. Useful in matplotlib plots.

    Parameters
    ----------
    python_string : str
        String that needs to be changed.

    Returns
    -------
    LaTeX_string : str
        Sring with the new underscore symbol.
    """

    string_list = list(python_string)

    for idx, string in enumerate(string_list):
        if string_list[idx] == '_':
            string_list[idx] = '\_'

    LaTeX_string = ''.join(string_list)

    return LaTeX_string


def store_csv(name, order, name_dir, save_to_csv):
    """
    Function that stores all important information in a CSVs files after the
    least squares optimisation has finished. It will be saved in the
    'OOF_out/name' directory

    Parameters
    ----------
    name : str
        File name of the fits file to be optimised.
    order : int
        Maximum order for the optimization in the Zernike circle polynomials
        coefficients.
    name_dir : str
        Directory of the fits file.
    save_to_csv : list
        It contains all data that will be stored.
        save_to_csv = [
        beam_data, u_data, v_data, res_optim, jac_optim, grad_optim, phase,
        cov_ptrue, corr_ptrue
            ]
    """

    headers = [
        'Normalised beam', 'u vector radians', 'v vector radians', 'Residual',
        'Jacobian', 'Gradient', 'Phase primary reflector radians',
        'Variance-Covariance matrix (first row fitted parameters idx)',
        'Correlation matrix (first row fitted parameters idx)'
        ]

    fnames = [
        '/beam_data.csv', '/u_data.csv', '/v_data.csv',
        '/res_n' + str(order) + '.csv', '/jac_n' + str(order) + '.csv',
        '/grad_n' + str(order) + '.csv', '/phase_n' + str(order) + '.csv',
        '/cov_n' + str(order) + '.csv', '/corr_n' + str(order) + '.csv'
        ]

    for fname, header, file in zip(fnames, headers, save_to_csv):
        np.savetxt(
            fname=name_dir + fname,
            X=file,
            header=header + ' ' + name
            )


def store_ascii(name, order, name_dir, params_to_save, info_to_save):
    """
    Function that stores all information in a ascii files after the
    least squares optimisation has finished. It will be saved in the
    'OOF_out/name' directory

    Parameters
    ----------
    name : str
        File name of the fits file to be optimised.
    order : int
        Order of the fit or opmisation done to the fits maps
    name_dir : str
        Directory of the fits file.
    params_to_save : list
        Stores the parameters found after the optimisation. The keywords for
        the fitpar_n#.dat ascii file are ['parname', 'parfit', 'parinit'].
    info_to_save : list
        Stores string and some other important information from the
        observations and optimisation. The keywords for the fitinfo.dat ascii
        file are ['telescope', 'name', 'd_z-', 'd_z0', 'd_z+', 'wavel',
        'freq', 'illumination', 'meanel', 'fft_resolution'].
    """

    ascii.write(
        table=params_to_save,
        output=name_dir + '/fitpar_n' + str(order) + '.dat',
        names=['parname', 'parfit', 'parinit'],
        comment='Fitted parameters ' + name
        )

    ascii.write(
        table=info_to_save,
        output=name_dir + '/fitinfo.dat',
        names=[
            'telescope', 'name', 'd_z-', 'd_z0', 'd_z+', 'wavel', 'freq',
            'illumination', 'meanel', 'fft_resolution'
            ],
        comment='Fit information ' + name
        )
