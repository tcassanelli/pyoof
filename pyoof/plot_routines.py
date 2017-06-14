#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from astropy.io import fits, ascii
from .aperture import angular_spectrum, phi
from .math_functions import cart2pol, angle_selection
from .aux_functions import extract_data_effelsberg, str2LaTeX
from .telgeometry import telescope_geo

__all__ = [
    'plot_beam', 'plot_data', 'plot_phase', 'plot_variance',
    'plot_data_effelsberg', 'plot_fit_path', 'plot_fit_nikolic',
    ]

# Plot style added from relative path
plotstyle_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(plotstyle_dir, 'pyoof.mplstyle'))


def plot_beam(
    params, d_z_m, lam, illum, telescope, resolution, plim_rad, title, angle
        ):

    I_coeff = params[:4]
    K_coeff = params[4:]

    # Selection between radians or degrees plotting
    wavevector_change, uv_title = angle_selection(angle)

    d_z = np.array(d_z_m) * 2 * np.pi / lam

    u, v, aspectrum = [], [], []
    for _d_z in d_z:

        _u, _v, _aspectrum = angular_spectrum(
            K_coeff=K_coeff,
            d_z=_d_z,
            I_coeff=I_coeff,
            illum=illum,
            telescope=telescope,
            resolution=resolution
            )

        u.append(_u)
        v.append(_v)
        aspectrum.append(_aspectrum)

    beam = np.abs(aspectrum) ** 2
    beam_norm = np.array([beam[i] / beam[i].max() for i in range(3)])

    # Limits, they need to be transformed to degrees
    if plim_rad is None:
        pr = telescope_geo(telescope)[1]  # primary reflector radius
        b_factor = 2 * pr / lam  # D / lambda
        plim_u = [-700 / b_factor, 700 / b_factor]
        plim_v = [-700 / b_factor, 700 / b_factor]
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
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z_m[0], 3)) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(d_z_m[1]) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z_m[2], 3)) + '$ m'
        ]

    for i in range(3):
        u_angle = wavevector_change(u[i], lam)
        v_angle = wavevector_change(v[i], lam)

        extent = [u_angle.min(), u_angle.max(), v_angle.min(), v_angle.max()]

        # make sure it is set to origin='lower', see plot style
        im = ax[i].imshow(beam_norm[i], extent=extent, vmin=0, vmax=1)
        ax[i].contour(u_angle, v_angle, beam_norm[i], levels)
        cb = fig.colorbar(im, ax=ax[i], shrink=shrink)

        ax[i].set_title(subtitle[i])
        ax[i].set_ylabel('$v$ ' + uv_title)
        ax[i].set_xlabel('$u$ ' + uv_title)
        ax[i].set_ylim(plim_v[0], plim_v[1])
        ax[i].set_xlim(plim_u[0], plim_u[1])
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


def plot_data(u_data, v_data, beam_data, d_z_m, title, angle):

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
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z_m[0], 3)) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(d_z_m[1]) + '$ m',
        '$P_\mathrm{norm}(u,v)$ $d_z=' + str(round(d_z_m[2], 3)) + '$ m']

    for i in range(3):
        # new grid for beam_data
        u_ng = np.linspace(u_data[i].min(), u_data[i].max(), 300)
        v_ng = np.linspace(v_data[i].min(), v_data[i].max(), 300)

        beam_ng = griddata(
            # coordinates of grid points to interpolate from.
            x=u_data[i],
            y=v_data[i],
            z=beam_data[i],
            # coordinates of grid points to interpolate to.
            xi=u_ng,
            yi=v_ng,
            interp='linear'
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


def plot_phase(K_coeff, d_z_m, title, notilt, telescope):

    _K_coeff = K_coeff.copy()

    if notilt:
        _K_coeff[1] = 0  # For value K(-1, 1) = 0
        _K_coeff[2] = 0  # For value K(1, 1) = 0
        subtitle = (
            '$\phi_\mathrm{no\,tilt}(x, y)$  $d_z=\pm' + str(round(d_z_m, 3)) +
            '$ m'
            )
        bartitle = '$\phi_\mathrm{no\,tilt}(x, y)$ amplitude rad'
    else:
        subtitle = '$\phi(x, y)$  $d_z=\pm' + str(round(d_z_m, 3)) + '$ m'
        bartitle = '$\phi(x, y)$ amplitude rad'

    pr = telescope_geo(telescope)[1]
    x = np.linspace(-pr, pr, 1e3)
    y = x

    x_grid, y_grid = np.meshgrid(x, y)

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / pr

    extent = [x.min(), x.max(), y.min(), y.max()]
    phase = phi(theta=t, rho=r_norm, K_coeff=_K_coeff)
    phase[(x_grid ** 2 + y_grid ** 2 > pr ** 2)] = 0

    fig, ax = plt.subplots()

    levels = np.linspace(-2, 2, 9)
    # as used in Nikolic software

    shrink = 1

    # to rotate and sign invert
    # phase = -phase[::-1, ::-1]

    im = ax.imshow(phase, extent=extent)
    ax.contour(x_grid, y_grid, phase, levels=levels, colors='k', alpha=0.5)
    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    ax.set_title(subtitle)
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    ax.grid('off')
    cb.ax.set_ylabel(bartitle)
    fig.suptitle(title)

    return fig


def plot_variance(matrix, params_names, title, cbtitle, diag):

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


# not sure to keep this function
def plot_data_effelsberg(pathfits, save, angle):
    """
    Plot all data from an OOF Effelsberg observation given the path.
    """

    data_info, data_obs = extract_data_effelsberg(pathfits)
    [name, _, _, d_z_m, _, pthto] = data_info
    [beam_data, u_data, v_data] = data_obs

    fig_data = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        title=name + ' observed beam',
        d_z_m=d_z_m,
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
    pathoof, order, telescope, plim_rad, save, angle, resolution
        ):

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

    d_z_m = np.array(
        [fitinfo['d_z-'][0], fitinfo['d_z0'][0], fitinfo['d_z+'][0]])
    name = fitinfo['name'][0]

    # LaTeX problem with underscore _ -> \_
    name = str2LaTeX(name)

    fig_beam = plot_beam(
        params=np.array(fitpar['parfit']),
        title=name + ' fitted power pattern  $n=' + str(n) + '$',
        d_z_m=d_z_m,
        lam=fitinfo['wavel'][0],
        illum=fitinfo['illum'][0],
        telescope=telescope,
        plim_rad=plim_rad,
        angle=angle,
        resolution=resolution
        )

    fig_phase = plot_phase(
        K_coeff=np.array(fitpar['parfit'])[4:],
        d_z_m=d_z_m[2],  # only one function for the three beam maps
        title=name + ' Aperture phase distribution  $n=' + str(n) + '$',
        notilt=True,
        telescope=telescope
        )

    fig_res = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=res,
        d_z_m=d_z_m,
        title=name + ' residual  $n=' + str(n) + '$',
        angle=angle,
        )

    fig_data = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        d_z_m=d_z_m,
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


def plot_fit_nikolic(pathoof, order, d_z_m, title, plim_rad, angle):
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
        d_z_m=d_z_m,
        lam=wavelength,
        illum='nikolic',
        plim_rad=plim_rad,
        angle=angle
        )

    fig_phase = plot_phase(
        params=params_nikolic,
        d_z_m=d_z_m[2],
        title=title + ' Nikolic fit $n=' + str(n) + '$',
        notilt=True
        )

    return fig_beam, fig_phase
