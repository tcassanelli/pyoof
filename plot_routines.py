# Author: Tomas Cassanelli
import matplotlib.pyplot as plt
import numpy as np
from main_functions import angular_spectrum, wavevector_to_degree, \
    cart2pol, phi, antenna_shape, aperture
from scipy.constants import c as light_speed
from matplotlib.mlab import griddata
from astropy.io import fits, ascii
import ntpath
import os

import matplotlib
# Standard parameters plot functions
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['lines.linewidth'] = 0.7
matplotlib.rcParams['image.cmap'] = 'viridis'
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['figure.titlesize'] = 12


def find_name_path(path):
    head, tail = ntpath.split(path)
    return head, tail


def plot_beam(params, d_z_m, lam, illum, plim_rad, title, rad):

    I_coeff = params[:4]
    K_coeff = params[4:]

    if rad:
        angle_coeff = np.pi / 180
        uv_title = 'radians'
    else:
        angle_coeff = 1
        uv_title = 'degrees'

    d_z = np.array(d_z_m) * 2 * np.pi / lam

    u, v, aspectrum = [], [], []
    for _d_z in d_z:

        _u, _v, _aspectrum = angular_spectrum(
            K_coeff=K_coeff,
            d_z=_d_z,
            I_coeff=I_coeff,
            illum=illum
            )

        u.append(_u)
        v.append(_v)
        aspectrum.append(_aspectrum)

    beam = np.abs(aspectrum) ** 2
    beam_norm = np.array([beam[i] / beam[i].max() for i in range(3)])

    # Limits, they need to be transformed to degrees
    if plim_rad is None:
        pr = 50  # primary reflector radius
        b_factor = 2 * pr / lam  # D / lambda
        plim_u = [-700 / b_factor, 700 / b_factor]
        plim_v = [-700 / b_factor, 700 / b_factor]
        figsize = (14, 4.5)
        shrink = 0.88

    else:
        plim_deg = plim_rad * 180 / np.pi
        plim_u, plim_v = plim_deg[:2], plim_deg[2:]
        figsize = (14, 3.3)
        shrink = 0.77

    fig, ax = plt.subplots(ncols=3, figsize=figsize)

    levels = 10  # number of colour lines

    subtitle = [
        '$|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z_m[0], 3)) + '$ m',
        '$|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(d_z_m[1]) + '$ m',
        '$|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z_m[2], 3)) + '$ m'
        ]

    for i in range(3):
        u_deg = wavevector_to_degree(u[i], lam) * angle_coeff
        v_deg = wavevector_to_degree(v[i], lam) * angle_coeff

        u_grid, v_grid = np.meshgrid(u_deg, v_deg)

        extent = [u_deg.min(), u_deg.max(), v_deg.min(), v_deg.max()]

        im = ax[i].imshow(beam_norm[i], extent=extent)
        ax[i].contour(u_grid, v_grid, beam_norm[i], levels)
        cb = fig.colorbar(im, ax=ax[i], shrink=shrink)

        ax[i].set_title(subtitle[i])
        ax[i].set_ylabel('$v$ ' + uv_title)
        ax[i].set_xlabel('$u$ ' + uv_title)
        ax[i].set_ylim(plim_v[0] * angle_coeff, plim_v[1] * angle_coeff)
        ax[i].set_xlim(plim_u[0] * angle_coeff, plim_u[1] * angle_coeff)

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


def plot_data(u_data, v_data, beam_data, d_z_m, title, rad, norm=False):

    if rad:
        angle_coeff = 1
        uv_title = 'radians'
    else:
        angle_coeff = 180 / np.pi
        uv_title = 'degrees'

    fig, ax = plt.subplots(ncols=3, figsize=(14, 3.3))

    levels = 10  # number of colour lines
    shrink = 0.77

    if norm:
        vmin = np.min(beam_data)
        vmax = np.max(beam_data)
    else:
        vmin = None
        vmax = None

    subtitle = [
        '$|P(u,v)|^2$ $d_z=' + str(round(d_z_m[0], 3)) + '$ m',
        '$|P(u,v)|^2$ $d_z=' + str(d_z_m[1]) + '$ m',
        '$|P(u,v)|^2$ $d_z=' + str(round(d_z_m[2], 3)) + '$ m']

    for i in range(3):
        # new grid for beam_data
        u_ng = np.linspace(u_data[i].min(), u_data[i].max(), 300) * angle_coeff
        v_ng = np.linspace(v_data[i].min(), v_data[i].max(), 300) * angle_coeff

        beam_ng = griddata(
            u_data[i] * angle_coeff, v_data[i] * angle_coeff,
            beam_data[i], u_ng, v_ng, interp='linear'
            )

        extent = [u_ng.min(), u_ng.max(), v_ng.min(), v_ng.max()]
        im = ax[i].imshow(beam_ng, extent=extent, vmin=vmin, vmax=vmax)
        ax[i].contour(u_ng, v_ng, beam_ng, levels, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax[i], shrink=shrink)

        ax[i].set_ylabel('$v$ ' + uv_title)
        ax[i].set_xlabel('$u$ ' + uv_title)
        ax[i].set_title(subtitle[i])

        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.06, bottom=0.1, right=1,
        top=0.91, wspace=0.13, hspace=0.2
        )

    return fig


def plot_phase(params, d_z_m, title, notilt):

    K_coeff = params[4:]

    if notilt:
        K_coeff[0] = 0
        K_coeff[1] = 0
        subtitle = (
            '$\phi_\mathrm{no\,tilt}(x, y)$  $d_z=\pm' + str(round(d_z_m, 3)) +
            '$ m'
            )
        bartitle = '$\phi_\mathrm{no\,tilt}(x, y)$ amplitude rad'
    else:
        subtitle = '$\phi(x, y)$  $d_z=\pm' + str(round(d_z_m, 3)) + '$ m'
        bartitle = '$\phi(x, y)$ amplitude rad'

    pr = 50
    x = np.linspace(-pr, pr, 1e3)
    y = np.linspace(-pr, pr, 1e3)

    x_grid, y_grid = np.meshgrid(x, y)

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / pr

    extent = [x.min(), x.max(), y.min(), y.max()]
    _phi = phi(theta=t, rho=r_norm, K_coeff=K_coeff) * antenna_shape(
        x_grid, y_grid)

    fig, ax = plt.subplots()

    levels = np.linspace(-2, 2, 9)
    # as used in Nikolic software

    shrink = 1

    im = ax.imshow(_phi, extent=extent)
    ax.contour(x_grid, y_grid, _phi, levels=levels, colors='k', alpha=0.5)
    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    ax.set_title(subtitle)
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    cb.ax.set_ylabel(bartitle)
    fig.suptitle(title)

    return fig


def plot_data_path(pathfits, save, rad):

    name = find_name_path(pathfits)[1][:-5]
    path = find_name_path(pathfits)[0]

    # Opening fits file with astropy
    hdulist = fits.open(pathfits)

    beam_data = [hdulist[i].data['fnu'] for i in range(1, 4)][::-1]
    u_data = [hdulist[i].data['DX'] for i in range(1, 4)][::-1]
    v_data = [hdulist[i].data['DY'] for i in range(1, 4)][::-1]
    d_z_m = [hdulist[i].header['DZ'] for i in range(1, 4)][::-1]

    # Permuting the position to provide same as main_functions
    beam_data.insert(1, beam_data.pop(2))
    u_data.insert(1, u_data.pop(2))
    v_data.insert(1, v_data.pop(2))
    d_z_m.insert(1, d_z_m.pop(2))

    fig_data = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        title=name + ' observed beam',
        d_z_m=d_z_m,
        rad=rad
        )

    if not os.path.exists(path + '/OOF_out'):
        os.makedirs(path + '/OOF_out')
    if not os.path.exists(path + '/OOF_out/' + name):
        os.makedirs(path + '/OOF_out/' + name)

    if save:
        fig_data.savefig(path + '/OOF_out/' + name +'/obsbeam.pdf')

    return fig_data


def plot_fit_path(pathoof, order, plim_rad, save, rad):

    n = order
    fitpar = ascii.read(pathoof + '/fitpar_n' + str(n) + '.dat')
    fitinfo = ascii.read(pathoof + '/fitinfo_n' + str(n) + '.dat')

    # Residual
    res = np.genfromtxt(pathoof + '/res_n' + str(n) + '.csv')
    u_data = np.genfromtxt(pathoof + '/u_data.csv')
    v_data = np.genfromtxt(pathoof + '/v_data.csv')
    beam_data = np.genfromtxt(pathoof + '/beam_data.csv')

    params_solution = np.array(fitpar['parfit'])

    d_z_m = np.array(
        [fitinfo['d_z-'][0], fitinfo['d_z0'][0], fitinfo['d_z+'][0]])
    name = fitinfo['name'][0]

    fig_beam = plot_beam(
        params=params_solution,
        title=name + ' fitted beam $n=' + str(n) + '$',
        d_z_m=d_z_m,
        lam=fitinfo['wavel'][0],
        illum=fitinfo['illum'][0],
        plim_rad=plim_rad,
        rad=rad
        )

    fig_phase = plot_phase(
        params=params_solution,
        d_z_m=d_z_m[2],  # only one function for the three beam maps
        title=name + ' Aperture phase distribution $n=' + str(n) + '$',
        notilt=True
        )

    fig_res = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=res,
        d_z_m=d_z_m,
        title=name + ' residual $n=' + str(n) + '$',
        rad=False,
        norm=True
        )

    fig_data = plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        d_z_m=d_z_m,
        title=name + ' observed beam',
        rad=False
        )

    if save:
        fig_beam.savefig(pathoof + '/fitbeam_n' + str(n) + '.pdf')
        fig_phase.savefig(pathoof + '/fitphase_n' + str(n) + '.pdf')
        fig_res.savefig(pathoof + '/residual_n' + str(n) + '.pdf')
        fig_data.savefig(pathoof + '/obsbeam.pdf')

    return fig_beam, fig_phase, fig_res, fig_data


def plot_fit_nikolic_path(pathoof, order, d_z_m, title, plim_rad, rad):

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
        rad=rad
        )

    fig_phase = plot_phase(
        params=params_nikolic,
        d_z_m=d_z_m[2],
        title=title + ' Nikolic fit $n=' + str(n) + '$',
        notilt=True
        )

    return fig_beam, fig_phase


if __name__ == "__main__":

    # for n in [1, 2, 3]:
    #     plot_fit_path(
    #         pathoof='../test_data/S9mm_0397_3C84/OOF_out/S9mm_0397_3C84_H1_SB',
    #         order=n,
    #         plim_rad=None,
    #         save=True,
    #         rad=False
    #         )
    #     plt.close()



    # params = fits.open('../test_data/gen_data8/o5n0/oofout/fitpars.fits')[1].data['ParValue']

    # plot_phase(params, 0.025, 'test', notilt=True)

    params = np.array([1, 1, 1, 1, 4, 5, 6])
    plot_phase(params, 0.025, 'title', notilt=True)

    plt.show()


