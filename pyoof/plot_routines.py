#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.io import fits, ascii
from .aperture import angular_spectrum, phase
from .math_functions import wavevector2degrees, wavevector2radians
from .aux_functions import extract_data_effelsberg, str2LaTeX

__all__ = [
    'plot_beam', 'plot_data', 'plot_phase', 'plot_variance',
    'plot_data_effelsberg', 'plot_fit_path', 'plot_fit_nikolic',
    ]

# Plot style added from relative path
plotstyle_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(plotstyle_dir, 'pyoof.mplstyle'))


def plot_beam(
    params, d_z, wavel, illum_func, telgeo, resolution, plim_rad, angle,
    title
        ):
    """
    Plot of the beam maps given fixed I_coeff coefficients and K_coeff
    coeffcients. It is the straight forward result from a least squares fit
    procedure. There will be three maps, for three out-of-focus values, d_z,
    given.

    Parameters
    ----------
    params : ndarray
        An stack of the illumination and Zernike circle polynomaisl
        coefficients. params = np.hstack([I_coeff, K_coeff])
    d_z : list
        Distance between the secondary and primary refelctor measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end. d_z = [-d_z, 0, +d_z].
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with parameters (x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage function, optical path difference
        (delta function), and the primary radius.
        telego = [blockage, delta, int].
    resolution : int
        Fast Fourier Transform resolution for a rectancular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    plim_rad : ndarray
        Contains the maximum values for the u and v wave-vectors, it can be in
        degrees or radias depending which one is chosen in angle function
        parameter. plim_rad = np.array([umin, umax, vmin, vmax]).
    angle : str
        Choose angle unit, it can be 'degrees' or 'radians'.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The three beam maps plotted from the input parameters. Each map with a
        different offset d_z value. From left to right, -d_z, 0 and +d_z.
    """

    I_coeff = params[:4]
    K_coeff = params[4:]

    # Selection between radians or degrees plotting
    uv_title = angle
    if angle == 'degrees':
        wavevector_change = wavevector2degrees
    else:
        wavevector_change = wavevector2radians

    u, v, aspectrum = [], [], []
    for _d_z in d_z:

        _u, _v, _aspectrum = angular_spectrum(
            K_coeff=K_coeff,
            I_coeff=I_coeff,
            d_z=_d_z,
            wavel=wavel,
            illum_func=illum_func,
            telgeo=telgeo,
            resolution=resolution
            )

        u.append(_u)
        v.append(_v)
        aspectrum.append(_aspectrum)

    beam = np.abs(aspectrum) ** 2
    beam_norm = np.array([beam[i] / beam[i].max() for i in range(3)])

    # Limits, they need to be transformed to degrees
    if plim_rad is None:
        pr = telgeo[2]  # primary reflector radius
        b_factor = 1.22 * wavel / (2 * pr)  # Beamwidth
        plim_u = [-600 * b_factor, 600 * b_factor]
        plim_v = [-600 * b_factor, 600 * b_factor]
        figsize = (14, 4.5)
        shrink = 0.88

    else:
        if angle == 'degrees':
            plim_angle = np.degrees(plim_rad)
        else:
            plim_angle = plim_rad
        plim_u, plim_v = plim_angle[:2], plim_angle[2:]
        figsize = (14, 3.3)
        shrink = 0.77

    fig, ax = plt.subplots(ncols=3, figsize=figsize)

    levels = 10  # number of colour lines

    subtitle = [
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z[0], 3)) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(d_z[1]) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z[2], 3)) + '$ m'
        ]

    for i in range(3):
        u_angle = wavevector_change(u[i], wavel)
        v_angle = wavevector_change(v[i], wavel)

        extent = [u_angle.min(), u_angle.max(), v_angle.min(), v_angle.max()]

        # make sure it is set to origin='lower', see plot style
        im = ax[i].imshow(beam_norm[i], extent=extent, vmin=0, vmax=1)
        ax[i].contour(u_angle, v_angle, beam_norm[i], levels)
        cb = fig.colorbar(im, ax=ax[i], shrink=shrink)

        ax[i].set_title(subtitle[i])
        ax[i].set_ylabel('$v$ ' + uv_title)
        ax[i].set_xlabel('$u$ ' + uv_title)
        ax[i].set_ylim(*plim_v)
        ax[i].set_xlim(*plim_u)
        ax[i].grid('off')

        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

    # fig.set_tight_layout(True)
    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.06, bottom=0.1, right=1,
        top=0.91, wspace=0.13, hspace=0.2
        )

    return fig


def plot_data(u_data, v_data, beam_data, d_z, angle, title):
    """
    Plot of the real data beam maps given given 3 out-of-focus values of d_z.

    Parameters
    ----------
    u_data : ndarray
        x axis value for the 3 beam maps in radians. The values have to be
        flatten, one dimentional, and stacked in the same order as the
        d_z = [-d_z, 0, +d_z] values from each beam map.
    v_data : ndarray
        y axis value for the 3 beam maps in radians. The values have to be
        flatten, one dimentional, and stacked in the same order as the
        d_z = [-d_z, 0, +d_z] values from each beam map.
    beam_data : ndarray
        Amplitude value for the beam map in any unit (it will be normalised).
        The values have to be flatten, one dimentional, and stacked in the
        same order as the d_z=[-d_z, 0, +d_z] values from each beam map.
    d_z : list
        Distance between the secondary and primary refelctor measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end. d_z = [-d_z, 0, +d_z].
    wavel : float
        Wavelength of the observation in meters.
    angle : str
        Choose angle unit, it can be 'degrees' or 'radians'.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Data figure from the three observed beam maps. Each map with a
        different offset d_z value. From left to right, -d_z, 0 and +d_z.
    """

    # Power pattern normalisation
    beam_data = [beam_data[i] / beam_data[i].max() for i in range(3)]
    # input u and v are in radians

    uv_title = angle

    if angle == 'degrees':
        u_data, v_data = np.degrees(u_data), np.degrees(v_data)

    fig, ax = plt.subplots(ncols=3, figsize=(14, 3.3))

    levels = 10  # number of colour lines
    shrink = 0.77

    vmin = np.min(beam_data)
    vmax = np.max(beam_data)

    subtitle = [
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z[0], 3)) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(d_z[1]) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z[2], 3)) + '$ m'
        ]

    for i in range(3):
        # new grid for beam_data
        u_ng = np.linspace(u_data[i].min(), u_data[i].max(), 300)
        v_ng = np.linspace(v_data[i].min(), v_data[i].max(), 300)

        beam_ng = interpolate.griddata(
            # coordinates of grid points to interpolate from.
            points=(u_data[i], v_data[i]),
            values=beam_data[i],
            # coordinates of grid points to interpolate to.
            xi=tuple(np.meshgrid(u_ng, v_ng)),
            method='cubic'
            )

        extent = [u_ng.min(), u_ng.max(), v_ng.min(), v_ng.max()]
        im = ax[i].imshow(beam_ng, extent=extent, vmin=vmin, vmax=vmax)
        ax[i].contour(u_ng, v_ng, beam_ng, levels)
        cb = fig.colorbar(im, ax=ax[i], shrink=shrink)

        ax[i].set_ylabel('$v$ ' + uv_title)
        ax[i].set_xlabel('$u$ ' + uv_title)
        ax[i].set_title(subtitle[i])
        ax[i].grid('off')

        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.06, bottom=0.1, right=1,
        top=0.91, wspace=0.13, hspace=0.2
        )

    return fig


def plot_phase(K_coeff, d_z, notilt, pr, title):
    """
    Plot of the phase or wavefront (aberration) distribution given the Zernike
    circle polynomials coefficients.

    Parameters
    ----------
    K_coeff : ndarray
        Constants coefficients for each of them there is only one Zernike
        circle polynomial.
    d_z : float
        Distance between the secondary and primary reflector measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end.
    notilt : bool
        True or False boolean to include or exclude the tilt coefficients in
        the aperture phase distribution. The Zernike circle polynomials are
        related to tilt through U(l=-1, n=1) and U(l=1, n=1).
    pr : float
        Primary reflector radius.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Phase distribution for the Zernike circle polynomials for a telescope
        primary dish.
    """

    if notilt:
        subtitle = (
            '$\\varphi_\mathrm{no\,tilt}(x, y)$  $d_z=\pm' +
            str(round(d_z, 3)) + '$ m'
            )
        bartitle = '$\\varphi_\mathrm{no\,tilt}(x, y)$ amplitude rad'
    else:
        subtitle = '$\\varphi(x, y)$  $d_z=\pm' + str(round(d_z, 3)) + '$ m'
        bartitle = '$\\varphi(x, y)$ amplitude rad'

    extent = [-pr, pr, -pr, pr]
    [x, y, _phase] = phase(K_coeff=K_coeff, notilt=notilt, pr=pr)

    fig, ax = plt.subplots()

    levels = np.linspace(-2, 2, 9)
    # as used in Nikolic software

    shrink = 1

    # to rotate and sign invert
    # phase = -phase[::-1, ::-1]

    im = ax.imshow(_phase, extent=extent)
    ax.contour(x, y, _phase, levels=levels, colors='k', alpha=0.3)
    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    ax.set_title(subtitle)
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    ax.grid('off')
    cb.ax.set_ylabel(bartitle)
    fig.suptitle(title)

    return fig


def plot_variance(matrix, params_names, diag, cbtitle, title):
    """
    Plot for the Variance-Covariance matrix or Correlation matrix. It returns
    the triangle figure with a color amplitude value for each elelemt. Used to
    check the correlation between the fitted parameters in a least squares
    optimisation.

    Parameters
    ----------
    matrix : ndarray
        Two dimensional array containing the Variance-Covariance or
        Correlation function. Output from the fit procedure.
    params_names : ndarray
        One dimensional array containing all the string names of the
        coefficients used, see store_ascii function.
    diag : bool
        If True it will plot the matrix diagonal.
    cbtitle : str
        Colorbar title.
    title : str
        Figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Triangle figure representing Variance-Covariance or Correlation matrix.
    """

    params_used = [int(i) for i in matrix[:1][0]]
    _matrix = matrix[1:]

    x_ticks, y_ticks = _matrix.shape
    shrink = 1

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

    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()
    cb.ax.set_ylabel(cbtitle)

    ax.set_title(title)

    ax.set_xticks(np.arange(x_ticks) + 0.5)
    ax.set_xticklabels(labels_x, rotation='vertical')
    ax.set_yticks(np.arange(y_ticks) + 0.5)
    ax.set_yticklabels(labels_y)
    ax.grid('off')

    return fig


# move function to a effelsberg script
def plot_data_effelsberg(pathfits, save, angle):
    """
    Plot all data from an OOF Effelsberg observation given the path.
    """

    data_info, data_obs = extract_data_effelsberg(pathfits)
    [name, _, _, d_z, _, pthto] = data_info
    [beam_data, u_data, v_data] = data_obs

    fig_data = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        title=name + ' observed beam',
        d_z=d_z,
        angle=angle
        )

    if not os.path.exists(pthto + '/OOF_out'):
        os.makedirs(pthto + '/OOF_out')
    if not os.path.exists(pthto + '/OOF_out/' + name):
        os.makedirs(pthto + '/OOF_out/' + name)
    if save:
        fig_data.savefig(pthto + '/OOF_out/' + name + '/obsbeam.pdf')

    return fig_data


def plot_fit_path(
    pathoof, order, telgeo, illum_func, resolution, angle, plim_rad, save
        ):
    """
    Plot all important figures after a least squares optimisation.

    Parameters
    ----------
    pathoof : str
        Path to the pyoof output, 'OOF_out/directory'.
    order : int
        Maximum order for the optimization in the Zernike circle polynomials
        coefficients.
    telgeo : list
        List that contains the blockage function, optical path difference
        (delta function), and the primary radius.
        telego = [blockage, delta, int].
    illum_func : function
        Illumination function with parameters (x, y, I_coeff, pr).
    resolution : int
        Fast Fourier Transform resolution for a rectancular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    angle : str
        Choose angle unit, it can be 'degrees' or 'radians'.
    plim_rad : ndarray
        Contains the maximum values for the u and v wave-vectors, it can be in
        degrees or radias depending which one is chosen in angle function
        parameter. plim_rad = np.array([umin, umax, vmin, vmax]).
    save : bool
        If True, it stores all plots in the 'OOF_out/name' directory.

    Returns
    -------
    fig_beam : matplotlib.figure.Figure
        The three beam maps plotted from the input parameters. Each map with a
        different offset d_z value. From left to right, -d_z, 0 and +d_z.
    fig_phase : matplotlib.figure.Figure
        Phase distribution for the Zernike circle polynomials for a telescope
        primary dish.
    fig_res : matplotlib.figure.Figure
        Residual from the three beam maps from the last residual evaluation in
        the optimisation procedure.
    fig_data : matplotlib.figure.Figure
        Data figure from the three observed beam maps. Each map with a
        different offset d_z value. From left to right, -d_z, 0 and +d_z
    fig_cov : matplotlib.figure.Figure
        Triangle figure representing Variance-Covariance matrix.
    fig_corr : matplotlib.figure.Figure
        Triangle figure representing Correlation matrix.
    """

    n = order
    fitpar = ascii.read(pathoof + '/fitpar_n' + str(n) + '.dat')
    fitinfo = ascii.read(pathoof + '/fitinfo.dat')

    # Residual
    res = np.genfromtxt(pathoof + '/res_n' + str(n) + '.csv')

    # Data
    u_data = np.genfromtxt(pathoof + '/u_data.csv')
    v_data = np.genfromtxt(pathoof + '/v_data.csv')
    beam_data = np.genfromtxt(pathoof + '/beam_data.csv')

    # Covariance and Correlation matrix
    cov = np.genfromtxt(pathoof + '/cov_n' + str(n) + '.csv')
    corr = np.genfromtxt(pathoof + '/corr_n' + str(n) + '.csv')

    d_z = np.array(
        [fitinfo['d_z-'][0], fitinfo['d_z0'][0], fitinfo['d_z+'][0]]
        )
    name = fitinfo['name'][0]

    # LaTeX problem with underscore _ -> \_
    name = str2LaTeX(name)

    fig_beam = plot_beam(
        params=np.array(fitpar['parfit']),
        title=name + ' fitted power pattern  $n=' + str(n) + '$',
        d_z=d_z,
        wavel=fitinfo['wavel'][0],
        illum_func=illum_func,
        telgeo=telgeo,
        plim_rad=plim_rad,
        angle=angle,
        resolution=resolution
        )

    fig_phase = plot_phase(
        K_coeff=np.array(fitpar['parfit'])[4:],
        d_z=d_z[2],  # only one function for the three beam maps
        title=name + ' Aperture phase distribution  $n=' + str(n) + '$',
        notilt=True,
        pr=telgeo[2]
        )

    fig_res = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=res,
        d_z=d_z,
        title=name + ' residual  $n=' + str(n) + '$',
        angle=angle,
        )

    fig_data = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        d_z=d_z,
        title=name + ' observed power pattern',
        angle=angle
        )

    fig_cov = plot_variance(
        matrix=cov,
        params_names=fitpar['parname'],
        title=name + ' Variance-covariance matrix $n=' + str(n) + '$',
        cbtitle='$\sigma_{ij}^2$',
        diag=True
        )

    fig_corr = plot_variance(
        matrix=corr,
        params_names=fitpar['parname'],
        title=name + ' Correlation matrix $n=' + str(n) + '$',
        cbtitle='$\\rho_{ij}$',
        diag=True
        )

    if save:
        fig_beam.savefig(pathoof + '/fitbeam_n' + str(n) + '.pdf')
        fig_phase.savefig(
            filename=pathoof + '/fitphase_n' + str(n) + '.pdf',
            bbox_inches='tight'
            )
        fig_res.savefig(pathoof + '/residual_n' + str(n) + '.pdf')
        fig_cov.savefig(pathoof + '/cov_n' + str(n) + '.pdf')
        fig_corr.savefig(pathoof + '/corr_n' + str(n) + '.pdf')
        fig_data.savefig(pathoof + '/obsbeam.pdf')

    return fig_beam, fig_phase, fig_res, fig_data, fig_cov, fig_corr


# I will erase this soon, need to be added to other script.
def plot_fit_nikolic(pathoof, order, d_z, title, plim_rad, angle):
    """
    Designed to plot Bojan Nikolic solutions given a path to its oof output.
    """

    n = order
    path = pathoof + '/z' + str(n)
    params_nikolic = np.array(
        fits.open(path + '/fitpars.fits')[1].data['ParValue'])

    wavelength = fits.open(path + '/aperture-notilt.fits')[0].header['WAVE']

    fig_beam = plot_beam(
        params=params_nikolic,
        title=title + ' phase distribution Nikolic fit $n=' + str(n) + '$',
        d_z=d_z,
        wavel=wavelength,
        illum='nikolic',
        plim_rad=plim_rad,
        angle=angle
        )

    fig_phase = plot_phase(
        params=params_nikolic,
        d_z=d_z[2],
        title=title + ' Nikolic fit $n=' + str(n) + '$',
        notilt=True
        )

    return fig_beam, fig_phase
