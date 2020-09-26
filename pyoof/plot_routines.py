#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from astropy.io import ascii
from astropy import units as apu
from astropy.utils.data import get_pkg_data_filename
import warnings
import os
import yaml
from .aperture import radiation_pattern, phase
from .aux_functions import uv_ratio
from .math_functions import norm

__all__ = [
    'plot_beam', 'plot_beam_data', 'plot_phase', 'plot_phase_data',
    'plot_variance', 'plot_fit_path'
    ]

# Plot style added from relative path
plt.style.use(get_pkg_data_filename('data/pyoof.mplstyle'))


def plot_beam(
    I_coeff, K_coeff, d_z, wavel, illum_func, telgeo, resolution, box_factor,
    plim, angle, title
        ):
    """
    Beam maps, :math:`P_\\mathrm{norm}(u, v)`, figure given fixed
    ``I_coeff`` coefficients and ``K_coeff`` set of coefficients. It is the
    straight forward result from a least squares minimization
    (`~pyoof.fit_zpoly`). There will be three maps, for three radial offsets,
    :math:`d_z^-`, :math:`0` and :math:`d_z^+` (in meters).

    Parameters
    ----------
    I_coeff : `list`
        List which contains 4 parameters, the illumination amplitude,
        :math:`A_{E_\\mathrm{a}}`, the illumination taper,
        :math:`c_\\mathrm{dB}` and the two coordinate offset, :math:`(x_0,
        y_0)`. The illumination coefficients must be listed as follows,
        ``I_coeff = [i_amp, c_dB, x0, y0]``.
    K_coeff : `~numpy.ndarray`
        Constants coefficients, :math:`K_{n\\ell}`, for each of them there is
        only one Zernike circle polynomial, :math:`U^\\ell_n(\\varrho,
        \\varphi)`.
    d_z : `~astropy.units.quantity.Quantity`
        Radial offset :math:`d_z`, added to the sub-reflector in length units.
        This characteristic measurement adds the classical interference
        pattern to the beam maps, normalized squared (field) radiation
        pattern, which is an out-of-focus property. The radial offset list
        must be as follows, ``d_z = [d_z-, 0., d_z+]`` all of them in length
        units.
    wavel : `~astropy.units.quantity.Quantity`
        Wavelength, :math:`\\lambda`, of the observation in length units.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telgeo : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (`float`) in meters. The list
        must have the following order, ``telego = [block_dist, opd_func, pr]``.
    resolution : `int`
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and with
        power of 2 for faster FFT processing. It is recommended a value higher
        than ``resolution = 2 ** 8``.
    box_factor : `int`
        Related to the FFT resolution (**resolution** key), defines the image
        pixel size level. It depends on the primary radius, ``pr``, of the
        telescope, e.g. a ``box_factor = 5`` returns ``x = np.linspace(-5 *
        pr, 5 * pr, resolution)``, an array to be used in the FFT2
        (`~numpy.fft.fft2`).
    plim : `~astropy.units.quantity.Quantity`
        Contains the maximum values for the :math:`u` and :math:`v`
        wave-vectors in angle units. The `~astropy.units.quantity.Quantity`
        must be in the following order, ``plim = [umin, umax, vmin, vmax]``.
    angle : `~astropy.units.quantity.Quantity`
        Angle unit. Axes for the power pattern.
    title : `str`
        Figure title.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The three beam maps plotted from the input parameters. Each map with a
        different offset :math:`d_z` value. From left to right, :math:`d_z^-`,
        :math:`0` and :math:`d_z^+`.
    """

    u, v, F = [], [], []
    for _d_z in d_z:

        _u, _v, _F = radiation_pattern(
            K_coeff=K_coeff,
            I_coeff=I_coeff,
            d_z=_d_z,
            wavel=wavel,
            illum_func=illum_func,
            telgeo=telgeo,
            resolution=resolution,
            box_factor=box_factor
            )

        u.append(_u)  # radians
        v.append(_v)  # radians
        F.append(_F)

    power_pattern = np.abs(F) ** 2
    power_norm = [norm(power_pattern[i]) for i in range(3)]

    # Limits, they need to be transformed to degrees
    if plim is None:
        pr = telgeo[2]                          # primary reflector radius
        bw = 1.22 * apu.rad * wavel / (2 * pr)  # Beamwidth radians
        s_bw = bw * 8                           # size-beamwidth ratio radians

        # Finding central point for shifted maps
        uu, vv = np.meshgrid(_u, _v)
        u_offset = uu[power_norm[1] == power_norm[1].max()][0]
        v_offset = vv[power_norm[1] == power_norm[1].max()][0]

        plim = [
            (-s_bw + u_offset).to_value(apu.rad),
            (s_bw + u_offset).to_value(apu.rad),
            (-s_bw + v_offset).to_value(apu.rad),
            (s_bw + v_offset).to_value(apu.rad)
            ] * apu.rad

    plim = plim.to_value(angle)
    plim_u, plim_v = plim[:2], plim[2:]

    subtitle = [
        '$P_{\\textrm{\\scriptsize{norm}}}(u,v)$ $d_z=' +
        str(round(d_z[i].to_value(apu.cm), 3)) + '$ cm' for i in range(3)
        ]

    fig = plt.figure(figsize=uv_ratio(plim_u, plim_v), constrained_layout=True)
    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        width_ratios=[1] * 3,
        height_ratios=[1, 0.03],
        wspace=0.03
        )
    ax = [plt.subplot(gs[i]) for i in range(6)]

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[0].set_ylabel('$v$ {}'.format(angle))

    cax = [ax[i + 3] for i in range(3)]

    for i in range(3):
        vmin, vmax = power_norm[i].min(), power_norm[i].max()

        extent = [
            u[i].to_value(angle).min(), u[i].to_value(angle).max(),
            v[i].to_value(angle).min(), v[i].to_value(angle).max()
            ]
        levels = np.linspace(vmin, vmax, 10)

        im = ax[i].imshow(X=power_norm[i], extent=extent, vmin=vmin, vmax=vmax)
        ax[i].contour(
            u[i].to_value(angle),
            v[i].to_value(angle),
            power_norm[i],
            levels=levels,
            colors='k',
            linewidths=0.4
            )

        ax[i].set_title(subtitle[i])
        ax[i].set_xlabel('$u$ {}'.format(angle))

        # limits don't work with astropy units
        ax[i].set_ylim(*plim_v)
        ax[i].set_xlim(*plim_u)
        ax[i].grid(False)

        plt.colorbar(
            im, cax=cax[i], orientation='horizontal', use_gridspec=True
            )
        cax[i].set_xlabel('Amplitude [arb]')
        cax[i].set_yticklabels([])
        cax[i].yaxis.set_ticks_position('none')
    fig.suptitle(title)
    # fig.tight_layout()

    return fig


def plot_beam_data(u_data, v_data, beam_data, d_z, angle, title, res_mode):
    """
    Real data beam maps, :math:`P^\\mathrm{obs}(x, y)`, figures given
    given 3 out-of-focus radial offsets, :math:`d_z`.

    Parameters
    ----------
    u_data : `list`
        :math:`x` axis value for the 3 beam maps in radians. The values have
        to be flatten, in one dimension, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    v_data : `list`
        :math:`y` axis value for the 3 beam maps in radians. The values have
        to be flatten, one dimensional, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    beam_data : `~numpy.ndarray`
        Amplitude value for the beam map in mJy. The values have to be
        flatten, one dimensional, and stacked in the same order as the ``d_z =
        [d_z-, 0., d_z+]`` values from each beam map. If ``res_mode = False``,
        the beam map will be normalized.
    d_z : `~astropy.units.quantity.Quantity`
        Radial offset :math:`d_z`, added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. The radial offset list must be as follows,
        ``d_z = [d_z-, 0., d_z+]`` all of them in length units.
    angle : `~astropy.units.quantity.Quantity`
        Angle unit. Axes for the power pattern.
    title : `str`
        Figure title.
    res_mode : `bool`
        If `True` the beam map will not be normalized. This feature is used
        to compare the residual outputs from the least squares minimization
        (`~pyoof.fit_zpoly`).

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure from the three observed beam maps. Each map with a different
        offset :math:`d_z` value. From left to right, :math:`d_z^-`, :math:`0`
        and :math:`d_z^+`.
    """

    if not res_mode:
        # Power pattern normalization
        beam_data = [norm(beam_data[i]) for i in range(3)]

    subtitle = [
        '$P_{\\textrm{\\scriptsize{norm}}}(u,v)$ $d_z=' +
        str(round(d_z[i].to_value(apu.cm), 3)) + '$ cm' for i in range(3)
        ]

    fig = plt.figure(
        figsize=uv_ratio(u_data[0], v_data[0]),
        constrained_layout=True
        )

    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        width_ratios=[1] * 3,
        height_ratios=[1, 0.03],
        wspace=0.03
        )
    ax = [plt.subplot(gs[i]) for i in range(6)]

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[0].set_ylabel('$v$ {}'.format(angle))

    cax = [ax[i + 3] for i in range(3)]

    for i in range(3):
        # new grid for beam_data
        u_ng = np.linspace(
            u_data[i].to(angle).min(), u_data[i].to(angle).max(), 300)
        v_ng = np.linspace(
            v_data[i].to(angle).min(), v_data[i].to(angle).max(), 300)

        beam_ng = interpolate.griddata(
            # coordinates of grid points to interpolate from.
            points=(u_data[i].to(angle), v_data[i].to(angle)),
            values=beam_data[i],
            # coordinates of grid points to interpolate to.
            xi=tuple(np.meshgrid(u_ng, v_ng)),
            method='cubic'
            )

        vmin, vmax = beam_ng.min(), beam_ng.max()
        levels = np.linspace(vmin, vmax, 10)
        extent = [
            u_ng.to_value(angle).min(), u_ng.to_value(angle).max(),
            v_ng.to_value(angle).min(), v_ng.to_value(angle).max()
            ]

        im = ax[i].imshow(X=beam_ng, extent=extent, vmin=vmin, vmax=vmax)
        ax[i].contour(
            u_ng.to_value(angle),
            v_ng.to_value(angle),
            beam_ng,
            levels=levels,
            colors='k',
            linewidths=0.4
            )

        ax[i].set_xlabel('$u$ {}'.format(angle))
        ax[i].set_title(subtitle[i])
        ax[i].grid(False)

        plt.colorbar(
            im, cax=cax[i], orientation='horizontal', use_gridspec=True
            )
        cax[i].set_xlabel('Amplitude [arb]')
        cax[i].set_yticklabels([])
        cax[i].yaxis.set_ticks_position('none')

    fig.suptitle(title)

    return fig


def plot_phase(K_coeff, tilt, pr, title):
    """
    Aperture phase distribution (phase-error), :math:`\\varphi(x, y)`, figure,
    given the Zernike circle polynomial coefficients, ``K_coeff``, solution
    from the least squares minimization.

    Parameters
    ----------
    K_coeff : `~numpy.ndarray`
        Constants coefficients, :math:`K_{n\\ell}`, for each of them there is
        only one Zernike circle polynomial, :math:`U^\\ell_n(\\varrho,
        \\varphi)`.
    tilt : `bool`
        Boolean to include or exclude the tilt coefficients in the aperture
        phase distribution. The Zernike circle polynomials are related to tilt
        through :math:`U^{-1}_1(\\varrho, \\varphi)` and
        :math:`U^1_1(\\varrho, \\varphi)`.
    pr : `astropy.units.quantity.Quantity`
        Primary reflector radius in length units.
    title : `str`
        Figure title.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Aperture phase distribution parametrized in terms of the Zernike
        circle polynomials, and represented for the telescope's primary
        reflector.
    """

    if not tilt:
        cbartitle = (
            '$\\varphi_{\\scriptsize{\\textrm{no-tilt}}}(x,y)$ amplitude rad'
            )
    else:
        cbartitle = '$\\varphi(x, y)$ amplitude rad'

    extent = [-pr.to_value(apu.m), pr.to_value(apu.m)] * 2
    levels = np.linspace(-2, 2, 9)  # radians
    _x, _y, _phase = phase(K_coeff=K_coeff, tilt=tilt, pr=pr)

    fig, ax = plt.subplots(figsize=(6, 5.8))

    im = ax.imshow(X=_phase.to_value(apu.rad), extent=extent)

    # Partial solution for contour Warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(
            _x.to_value(apu.m),
            _y.to_value(apu.m),
            _phase.to_value(apu.rad),
            levels=levels,
            colors='k',
            alpha=0.3
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(cbartitle)

    ax.set_title(title)
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    ax.grid(False)

    fig.tight_layout()

    return fig


def plot_phase_data(phase_data, pr, title):
    """
    Aperture phase distribution (phase-error), :math:`\\varphi(x, y)`, figure.
    The plot is made by giving the phase_data in radians and the primary
    reflector in length units. Notice that if the tilt term is not required
    this has to be removed manually from the ``phase_data`` array.

    Parameters
    ----------
    phase_data : `astropy.units.quantity.Quantity`
        Aperture phase distribution data in angle or radian units.
    pr : `astropy.units.quantity.Quantity`
        Primary reflector radius in length units.
    title : `str`
        Figure title.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Aperture phase distribution represented for the telescope's primary
        reflector.
    """

    _x = np.linspace(-pr, pr, phase_data.shape[0])
    _y = np.linspace(-pr, pr, phase_data.shape[0])

    extent = [-pr.to_value(apu.m), pr.to_value(apu.m)] * 2
    levels = np.linspace(-2, 2, 9)  # radians

    fig, ax = plt.subplots(figsize=(6, 5.8))

    im = ax.imshow(X=phase_data.to_value(apu.rad), extent=extent)

    # Partial solution for contour Warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(
            _x.to_value(apu.m),
            _y.to_value(apu.m),
            phase_data.to_value(apu.rad),
            levels=levels,
            colors='k',
            alpha=0.3
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(im, cax=cax)
    cbartitle = '$\\varphi(x, y)$ amplitude rad'
    cb.ax.set_ylabel(cbartitle)

    ax.set_title(title)
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    ax.grid(False)

    fig.tight_layout()

    return fig


def plot_variance(matrix, order, diag, cbtitle, title):
    """
    Variance-Covariance matrix or Correlation matrix figure. It returns
    the triangle figure with a color amplitude value for each element. Used to
    check/compare the correlation between the fitted parameters in a least
    squares minimization.

    Parameters
    ----------
    matrix : `~numpy.ndarray`
        Two dimensional array containing the Variance-Covariance or
        Correlation function. Output from the fit procedure.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.
    diag : `bool`
        If `True` it will plot the matrix diagonal.
    cbtitle : `str`
        Color bar title.
    title : `str`
        Figure title.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Triangle figure representing Variance-Covariance or Correlation matrix.
    """
    n = order
    N_K_coeff = (n + 1) * (n + 2) // 2
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    params_names = ['$A_{E_\mathrm{a}}$', '$c_\\mathrm{dB}$', '$x_0$', '$y_0$']
    for i in range(N_K_coeff):
        params_names.append('$K_{' + str(N[i]) + '\,' + str(L[i]) + '}$')
    params_names = np.array(params_names)
    params_used = [int(i) for i in matrix[:1][0]]
    _matrix = matrix[1:]

    x_ticks, y_ticks = _matrix.shape

    extent = [0, x_ticks, 0, y_ticks]

    if diag:
        k = -1
        # idx represents the ignored elements
        labels_x = params_names[params_used]
        labels_y = labels_x[::-1]
    else:
        k = 0
        # idx represents the ignored elements
        labels_x = params_names[params_used][:-1]
        labels_y = labels_x[::-1][:-1]

    # selecting half covariance
    mask = np.tri(_matrix.shape[0], k=k)
    matrix_mask = np.ma.array(_matrix, mask=mask).T
    # mask out the lower triangle

    fig, ax = plt.subplots()

    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    im = ax.imshow(
        X=matrix_mask,
        extent=extent,
        vmax=_matrix.max(),
        vmin=_matrix.min(),
        cmap=plt.cm.Reds,
        interpolation='nearest',
        origin='upper'
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(im, cax=cax)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    cb.ax.set_ylabel(cbtitle)

    ax.set_title(title)
    ax.set_xticks(np.arange(x_ticks) + 0.5)
    ax.set_xticklabels(labels_x, rotation='vertical')
    ax.set_yticks(np.arange(y_ticks) + 0.5)
    ax.set_yticklabels(labels_y)
    ax.grid(False)

    fig.tight_layout()

    return fig


def plot_fit_path(
    path_pyoof, order, illum_func, telgeo, resolution, box_factor, angle,
    plim, save
        ):
    """
    Plot all important figures after a least squares minimization.

    Parameters
    ----------
    path_pyoof : `str`
        Path to the pyoof output, ``'pyoof_out/directory'``.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telgeo : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (`float`) in meters. The list
        must have the following order, ``telego = [block_dist, opd_func, pr]``.
    resolution : `int`
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and with
        power of 2 for faster FFT processing. It is recommended a value higher
        than ``resolution = 2 ** 8``.
    box_factor : `int`
        Related to the FFT resolution (**resolution** key), defines the image
        pixel size level. It depends on the primary radius, ``pr``, of the
        telescope, e.g. a ``box_factor = 5`` returns ``x = np.linspace(-5 *
        pr, 5 * pr, resolution)``, an array to be used in the FFT2
        (`~numpy.fft.fft2`).
    angle : `~astropy.units.quantity.Quantity`
        Angle unit. Axes for the power pattern.
    plim : `~astropy.units.quantity.Quantity`
        Contains the maximum values for the :math:`u` and :math:`v`
        wave-vectors in angle units. The `~astropy.units.quantity.Quantity`
        must be in the following order, ``plim = [umin, umax, vmin, vmax]``.
    save : `bool`
        If `True`, it stores all plots in the ``'pyoof_out/directory'``
        directory.

    Returns
    -------
    fig_beam : `~matplotlib.figure.Figure`
        The three beam maps plotted from the input parameters. Each map with a
        different offset :math:`d_z` value. From left to right, :math:`d_z^-`,
        :math:`0` and :math:`d_z^+`.
    fig_phase : `~matplotlib.figure.Figure`
        Aperture phase distribution for the Zernike circle polynomials for the
        telescope primary reflector.
    fig_res : `~matplotlib.figure.Figure`
        Figure from the three observed beam maps residual. Each map with a
        different offset :math:`d_z` value. From left to right, :math:`d_z^-`,
        :math:`0` and :math:`d_z^+`.
    fig_data : `~matplotlib.figure.Figure`
        Figure from the three observed beam maps. Each map with a different
        offset :math:`d_z` value. From left to right, :math:`d_z^-`, :math:`0`
        and :math:`d_z^+`.
    fig_cov : `~matplotlib.figure.Figure`
        Triangle figure representing Variance-Covariance matrix.
    fig_corr : `~matplotlib.figure.Figure`
        Triangle figure representing Correlation matrix.
    """

    try:
        path_pyoof
    except NameError:
        print('pyoof directory does not exist: ' + path_pyoof)
    else:
        pass

    path_plot = os.path.join(path_pyoof, 'plots')

    if not os.path.exists(path_plot):
        os.makedirs(path_plot)

    # Reading least squares minimization output
    n = order
    fitpar = ascii.read(os.path.join(path_pyoof, 'fitpar_n{}.csv'.format(n)))

    with open(os.path.join(path_pyoof, 'pyoof_info.yml'), 'r') as inputfile:
        pyoof_info = yaml.load(inputfile, Loader=yaml.Loader)

    obs_object = pyoof_info['obs_object']
    meanel = round(pyoof_info['meanel'], 2)

    # Beam and residual
    beam_data = np.genfromtxt(os.path.join(path_pyoof, 'beam_data.csv'))
    res = np.genfromtxt(os.path.join(path_pyoof, 'res_n{}.csv'.format(n)))

    # fixing astropy units
    u_data = np.genfromtxt(os.path.join(path_pyoof, 'u_data.csv')) * apu.rad
    v_data = np.genfromtxt(os.path.join(path_pyoof, 'v_data.csv')) * apu.rad

    wavel = pyoof_info['wavel'] * apu.m
    d_z = np.array(pyoof_info['d_z']) * apu.m

    # Covariance and Correlation matrix
    cov = np.genfromtxt(os.path.join(path_pyoof, 'cov_n{}.csv'.format(n)))
    corr = np.genfromtxt(os.path.join(path_pyoof, 'corr_n{}.csv'.format(n)))

    if n == 1:
        fig_data = plot_beam_data(
            u_data=u_data,
            v_data=v_data,
            beam_data=beam_data,
            d_z=d_z,
            title='{} observed power pattern $\\alpha={}$ deg'.format(
                obs_object, meanel),
            angle=angle,
            res_mode=False
            )

    fig_beam = plot_beam(
        I_coeff=fitpar['parfit'][:4],
        K_coeff=fitpar['parfit'][4:],
        title='{} fit power pattern  $n={}$ $\\alpha={}$ degrees'.format(
            obs_object, n, meanel),
        d_z=d_z,
        wavel=wavel,
        illum_func=illum_func,
        telgeo=telgeo,
        plim=plim,
        angle=angle,
        resolution=resolution,
        box_factor=box_factor
        )

    fig_phase = plot_phase(
        K_coeff=fitpar['parfit'][4:],
        title=(
            '{} phase-error $d_z=\\pm {}$ cm ' +
            '$n={}$ $\\alpha={}$ deg'
            ).format(obs_object, round(d_z[2].to_value(apu.cm), 3), n, meanel),
        tilt=False,
        pr=telgeo[2]
        )

    fig_res = plot_beam_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=res,
        d_z=d_z,
        title='{} residual $n={}$'.format(obs_object, n),
        angle=angle,
        res_mode=True
        )

    fig_cov = plot_variance(
        matrix=cov,
        order=n,
        title='{} variance-covariance matrix $n={}$'.format(obs_object, n),
        cbtitle='$\\sigma_{ij}^2$',
        diag=True,
        )

    fig_corr = plot_variance(
        matrix=corr,
        order=n,
        title='{} correlation matrix $n={}$'.format(obs_object, n),
        cbtitle='$\\rho_{ij}$',
        diag=True,
        )

    if save:
        fig_beam.savefig(os.path.join(path_plot, 'fitbeam_n{}.pdf'.format(n)))
        fig_phase.savefig(
            os.path.join(path_plot, 'fitphase_n{}.pdf'.format(n)))
        fig_res.savefig(os.path.join(path_plot, 'residual_n{}.pdf'.format(n)))
        fig_cov.savefig(os.path.join(path_plot, 'cov_n{}.pdf'.format(n)))
        fig_corr.savefig(os.path.join(path_plot, 'corr_n{}.pdf'.format(n)))

        if n == 1:
            fig_data.savefig(os.path.join(path_plot, 'obsbeam.pdf'))
