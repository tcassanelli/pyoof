#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import numpy as np
from astropy.io import ascii, fits
from astropy import units as apu
from astropy.constants import c as light_speed
from .aperture import illum_gauss, illum_pedestal

__all__ = [
    'extract_data_pyoof', 'extract_data_effelsberg', 'str2LaTeX',
    'store_data_csv', 'uv_ratio', 'illum_strings', 'store_data_ascii'
    ]


def illum_strings(illum_func):
    """
    It assigns string labels to the illumination function. The `~pyoof` package
    has two standard illumination functions, `~pyoof.aperture.illum_pedestal`
    and `~pyoof.aperture.illum_gauss`.

    Parameters
    ----------
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with ``I_coeff``. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.

    Returns
    -------
    illum_name : `str`
        String with the illumination function name.
    taper_name : `str`
        String with the illumination function taper.
    """

    # adding illumination function information
    if illum_func == illum_pedestal:
        illum_name = 'pedestal'
        taper_name = 'c_dB'
    elif illum_func == illum_gauss:
        illum_name = 'gauss'
        taper_name = 'sigma_dB'
    else:
        illum_name = 'manual'
        taper_name = 'taper_dB'

    return illum_name, taper_name


def extract_data_pyoof(pathfits):
    """
    Extracts data from the `~pyoof` default fits file OOF holography
    observations, ready to use for the least squares minimization (see
    `~pyoof.fit_beam`). The fits file has to have the following keys on its
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
        minimization (see `~pyoof.fit_beam`). The list has the following order
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

    beam_data = [hdulist[i].data['BEAM'] for i in range(1, 4)]
    u_data = [hdulist[i].data['U'] * apu.rad for i in range(1, 4)]
    v_data = [hdulist[i].data['V'] * apu.rad for i in range(1, 4)]
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
        minimization (see `~pyoof.fit_beam`). The list has the following order
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

    beam_data = [hdulist[i].data['fnu'] for i in pos]
    u_data = [hdulist[i].data['DX'] * apu.rad for i in pos]
    v_data = [hdulist[i].data['DY'] * apu.rad for i in pos]

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
    minimization has finished, `~pyoof.fit_beam`. All data will be stores in
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
        'Jacobian', 'Gradient', 'Phase primary reflector radians',
        'Variance-Covariance matrix (first row fitted parameters idx)',
        'Correlation matrix (first row fitted parameters idx)'
        ]

    fnames = [
        '/beam_data.csv', '/u_data.csv', '/v_data.csv',
        '/res_n{}.csv'.format(order), '/jac_n{}.csv'.format(order),
        '/grad_n{}.csv'.format(order), '/phase_n{}.csv'.format(order),
        '/cov_n{}.csv'.format(order), '/corr_n{}.csv'.format(order)
        ]

    if order != 1:
        headers = headers[3:]
        fnames = fnames[3:]
        save_to_csv = save_to_csv[3:]

    for fname, header, file in zip(fnames, headers, save_to_csv):
        np.savetxt(
            fname=name_dir + fname,
            X=file,
            header=header + ' ' + name
            )


def store_data_ascii(
    name, name_dir, taper_name, order, params_solution, params_init
        ):
    """
    Stores in an ascii format the parameters found by the least squares
    minimization (see `~pyoof.fit_beam`).

    Parameters
    ----------
    name : `str`
        File name of the fits file to be optimized.
    name_dir : `str`
        Path to store all the csv files. The files will depend on the order of
        the Zernike circle polynomial.
    taper_name : `str`
        Name of the illumination function taper.
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

    params_names = ['i_amp', taper_name, 'x_0', 'y_0']
    for i in range(N_K_coeff):
        params_names.append('K({}, {})'.format(N[i], L[i]))

    # To store fit information and found parameters in ascii file
    ascii.write(
        table=[params_names, params_solution, params_init],
        output=name_dir + '/fitpar_n{}.csv'.format(n),
        names=['parname', 'parfit', 'parinit'],
        comment='Fitted parameters ' + name,
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

    ratio = (v.max() - v.min()) / (u.max() - u.min()) * 30

    width = ratio
    height = width / 5

    return width, height
