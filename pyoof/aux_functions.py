#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import yaml
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, QTable
from astropy import units as apu
from astropy.constants import c as light_speed
from .math_functions import rms
from .aperture import e_rs

__all__ = [
    'extract_data_pyoof', 'extract_data_effelsberg', 'str2LaTeX',
    'store_data_csv', 'uv_ratio', 'store_data_ascii', 'table_pyoof_out'
    ]


def extract_data_pyoof(pathfits):
    """
    Extracts data from the `~pyoof` default fits file OOF holography
    observations, ready to use for the least squares minimization (see
    `~pyoof.fit_zpoly`). The fits file has to have the following keys on its
    PrimaryHDU header: ``'FREQ'``, ``'WAVEL'``, ``'MEANEL'``, ``'OBJECT'`` and
    ``'DATE_OBS'``. Besides this three BinTableHDU are required for the data
    itself; ``MINUS OOF``, ``ZERO OOF`` and ``PLUS OOF``. The BinTableHDU
    header has to have the ``'DZ'`` key which includes the radial offset,
    :math:`d_z`. Finally the BinTableHDU has the data files ``'U'``, ``'V'``
    and ``'BEAM'``, which is the :math:`x`- and :math:`y`-axis position in
    radians and the ``'BEAM'`` in a flat array, in mJy.

    Parameters
    ----------
    pathfits : `str`
        Path to the fits file that contains the three beam maps pre-calibrated,
        using the correct PrimaryHDU and the three BinTableHDU (``MINUS OOF``,
        ``ZERO OOF`` and ``PLUS OOF``).

    Returns
    -------
    data_info : `list`
        It contains all extra data besides the beam map. The output
        corresponds to a list,
        ``[name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]``.
        These are, name of the fits file, paht of the fits file, observed
        object, observation date, frequency, wavelength, radial offset and
        mean elevation, respectively.
    data_obs : `list`
        It contains beam maps and :math:`x`-, and :math:`y`-axis
        (:math:`uv`-plane in Fourier space) data for the least squares
        minimization (see `~pyoof.fit_zpoly`). The list has the following order
        ``[beam_data, u_data, v_data]``. ``beam_data`` is the three beam
        observations, minus, zero and plus out-of-focus, in a flat array.
        ``u_data`` and ``v_data`` are the beam axes in a flat array.
    """

    hdulist = fits.open(pathfits)  # open fits file, pyoof format
    # path or directory where the fits file is located
    pthto = os.path.split(pathfits)[0]
    # name of the fit file to fit
    name = os.path.split(pathfits)[1][:-5]

    if not all(
            k in hdulist[0].header
            for k in ['FREQ', 'WAVEL', 'MEANEL', 'OBJECT', 'DATE_OBS']
            ):
        raise TypeError('Not all necessary keys found in FITS header.')

    freq = hdulist[0].header['FREQ'] * apu.Hz
    wavel = hdulist[0].header['WAVEL'] * apu.m
    meanel = hdulist[0].header['MEANEL'] * apu.deg
    obs_object = hdulist[0].header['OBJECT']
    obs_date = hdulist[0].header['DATE_OBS']

    beam_data = np.array([hdulist[i].data['BEAM'] for i in range(1, 4)])
    u_data = np.array([hdulist[i].data['U'] for i in range(1, 4)]) * apu.rad
    v_data = np.array([hdulist[i].data['V'] for i in range(1, 4)]) * apu.rad
    d_z = np.array([hdulist[i].header['DZ'] for i in range(1, 4)]) * apu.m

    data_file = [name, pthto]
    data_info = data_file + [obs_object, obs_date, freq, wavel, d_z, meanel]
    data_obs = [beam_data, u_data, v_data]

    return data_info, data_obs


def extract_data_effelsberg(pathfits):
    """
    Extracts data from the Effelsberg OOF holography observations, ready to
    use for the least squares minimization. This function will only work for
    the Effelsberg telescope beam maps.

    Parameters
    ----------
    pathfits : `str`
        Path to the fits file that contains the three beam maps pre-calibrated,
        from the Effelsberg telescope.

    Returns
    -------
    data_info : `list`
        It contains all extra data besides the beam map. The output
        corresponds to a list,
        ``[name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]``.
        These are, name of the fits file, paht of the fits file, observed
        object, observation date, frequency, wavelength, radial offset and
        mean elevation, respectively.
    data_obs : `list`
        It contains beam maps and :math:`x`-, and :math:`y`-axis
        (:math:`uv`-plane in Fourier space) data for the least squares
        minimization (see `~pyoof.fit_zpoly`). The list has the following order
        ``[beam_data, u_data, v_data]``. ``beam_data`` is the three beam
        observations, minus, zero and plus out-of-focus, in a flat array.
        ``u_data`` and ``v_data`` are the beam axes in a flat array.
    """

    pos = [3, 1, 2]  # Positions for OOF holography observations at Effelsberg
    hdulist = fits.open(pathfits)  # main fits file OOF holography format

    # Observation frequency
    freq = hdulist[0].header['FREQ'] * apu.Hz
    wavel = light_speed / freq

    # Mean elevation
    meanel = hdulist[0].header['MEANEL'] * apu.deg
    obs_object = hdulist[0].header['OBJECT']        # observed object
    obs_date = hdulist[0].header['DATE_OBS']        # observation date
    d_z = np.array([hdulist[i].header['DZ'] for i in pos]) * apu.m

    beam_data = np.array([hdulist[i].data['fnu'] for i in pos])
    u_data = np.array([hdulist[i].data['DX'] for i in pos]) * apu.rad
    v_data = np.array([hdulist[i].data['DY'] for i in pos]) * apu.rad

    # path or directory where the fits file is located
    pthto = os.path.split(pathfits)[0]
    # name of the fit file to fit
    name = os.path.split(pathfits)[1][:-5]

    data_info = [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]
    data_obs = [beam_data, u_data, v_data]

    return data_info, data_obs


def str2LaTeX(python_string):
    """
    Function that solves the underscore problem in a python string to
    :math:`\\LaTeX` string.

    Parameters
    ----------
    python_string : `str`
        String that needs to be changed.

    Returns
    -------
    LaTeX_string : `str`
        String with the new underscore symbol.
    """

    string_list = list(python_string)
    for idx, string in enumerate(string_list):
        if string_list[idx] == '_':
            string_list[idx] = '\\_'

    LaTeX_string = ''.join(string_list)

    return LaTeX_string


def store_data_csv(name, name_dir, order, save_to_csv):
    """
    Stores all important information in a csv file after the least squares
    minimization has finished, `~pyoof.fit_zpoly`. All data will be stores in
    the ``pyoof_out/name`` directory, with ``name`` the name of the fits file.

    Parameters
    ----------
    name : `str`
        File name of the fits file to be optimized.
    name_dir : `str`
        Path to store all the csv files. The files will depend on the order of
        the Zernike circle polynomial.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.
    save_to_csv : `list`
        It contains all data that will be stored. The list must have the
        following order, ``[beam_data, u_data, v_data, res_optim, jac_optim,
        grad_optim, phase, cov_ptrue, corr_ptrue]``.
    """

    headers = [
        'Normalized beam', 'u vector radians', 'v vector radians', 'Residual',
        'Jacobian', 'Gradient', 'Phase-error radians',
        'Variance-Covariance matrix (first row fitted parameters idx)',
        'Correlation matrix (first row fitted parameters idx)'
        ]

    fnames = [
        f'beam_data.csv', f'u_data.csv', f'v_data.csv',
        f'res_n{order}.csv', f'jac_n{order}.csv',
        f'grad_n{order}.csv', f'phase_n{order}.csv',
        f'cov_n{order}.csv', f'corr_n{order}.csv'
        ]

    if order != 1:
        headers = headers[3:]
        fnames = fnames[3:]
        save_to_csv = save_to_csv[3:]

    for fname, header, file in zip(fnames, headers, save_to_csv):
        np.savetxt(
            fname=os.path.join(name_dir, fname),
            X=file,
            header=' '.join((header, name))
            )


def store_data_ascii(name, name_dir, order, params_solution, params_init):
    """
    Stores in an ascii format the parameters found by the least squares
    minimization (see `~pyoof.fit_zpoly`).

    Parameters
    ----------
    name : `str`
        File name of the fits file to be optimized.
    name_dir : `str`
        Path to store all the csv files. The files will depend on the order of
        the Zernike circle polynomial.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.
    params_solution : `~numpy.ndarray`
        Contains the best fitted parameters, the illumination function
        coefficients, ``I_coeff`` and the Zernike circle polynomial
        coefficients, ``K_coeff`` in one array.
    params_init : `~numpt.ndarray`
        Contains the initial parameters used in the least squares minimization
        to start finding the best fitted combination of them.
    """

    n = order
    N_K_coeff = (n + 1) * (n + 2) // 2

    # Making nice table :)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    params_names = ['i_amp', 'c_dB', 'q', 'x_0', 'y_0']
    for i in range(N_K_coeff):
        params_names.append(f'K({N[i]}, {L[i]})')

    # To store fit information and found parameters in ascii file
    tab = Table(
        data=[params_names, params_solution, params_init],
        names=['parname', 'parfit', 'parinit'],
        )

    tab.write(
        os.path.join(name_dir, f'fitpar_n{n}.csv'),
        overwrite=True
        )


def uv_ratio(u, v):
    """
    Calculates the aspect ratio for the 3 power pattern plots, plus some
    corrections for the text on it. Used in the `function` `~pyoof.plot_beam`
    and `~pyoof.plot_data`

    Parameters
    ----------
    u : `~astropy.units.quantity.Quantity`
        Spatial frequencies from the power pattern, usually in degrees.
    v : `~astropy.units.quantity.Quantity`
        Spatial frequencies from the power pattern, usually in degrees.

    Returns
    -------
    width : `float`
        Width for the power pattern figure.
    height : `float`
        Height for the power pattern figure.
    """

    ratio = (u.max() - u.min()) / (v.max() - v.min())

    height = 5
    width = ratio * 2.25 * height

    return width, height


def table_pyoof_out(path_pyoof_out, order):
    """
    Auxiliary function to tabulate all data from a series of observations
    gathered in a common pyoof_out directory.

    Parameters
    ----------
    path_pyoof_out : `str`
        Path to the directory ``pyoof_out/`` or where the output from the
        `~pyoof` package is located.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.

    Returns
    -------
    qt : `~astropy.table.table.QTable`
        Table with units of the most important quantities from the `~pyoof`
        package.
    """

    qt = QTable(
        names=[
        'name', 'tel_name', 'obs-object', 'meanel', 'beam-snr', 'obs-date',
        'i_amp', 'c_dB', 'q', 'phase-rms', 'e_rs'
            ],
        dtype=[
            np.string_, np.string_, np.string_, np.float, np.float, np.string_,
            np.float, np.float, np.float, np.float, np.float
            ]
        )

    for p, pyoof_out in enumerate(path_pyoof_out):

        with open(os.path.join(pyoof_out, 'pyoof_info.yml'), 'r') as inputfile:
            pyoof_info = yaml.load(inputfile, Loader=yaml.Loader)

        phase = np.genfromtxt(os.path.join(pyoof_out, f'phase_n{order}.csv'))
        phase_rms = rms(phase, circ=True)
        phase_e_rs = e_rs(phase, circ=True)

        params = Table.read(
            os.path.join(pyoof_out, f'fitpar_n{order}.csv'),
            format='ascii'
            )

        I_coeff = params['parfit'][:5]

        qt.add_row([
            pyoof_info['name'], pyoof_info['tel_name'],
            pyoof_info['obs_object'], pyoof_info['meanel'], pyoof_info['snr'],
            pyoof_info['obs_date'], I_coeff[0], I_coeff[1], I_coeff[2],
            phase_rms, phase_e_rs
            ])

    # updating units
    qt['phase-rms'] *= apu.rad
    qt['meanel'] *= apu.deg
    qt['obs-date'] = Time(qt['obs-date'], format='isot', scale='utc')
    qt['c_dB'] *= apu.dB

    qt.meta = {
    'order': order
        }

    return qt
