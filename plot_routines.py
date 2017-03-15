# Author: Tomas Cassanelli
import matplotlib.pyplot as plt
import numpy as np
from main_functions import angular_spectrum, wavevector_to_degree, \
    cart2pol, phi, antenna_shape, aperture
from scipy.constants import c as light_speed
from matplotlib.mlab import griddata
from astropy.io import fits
import ntpath
from astropy.io import ascii
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


def plot_beam(params, title, x, y, d_z_m, lam, illum, rad=False):

    I_coeff = params[:4]
    K_coeff = params[4:]

    if rad:
        angle_coeff = np.pi / 180
        uv_title = 'radians'
    else:
        angle_coeff = 1
        uv_title = 'degrees'

    d_z = np.array(d_z_m) * 2 * np.pi / lam

    u0, v0, as0 = angular_spectrum(
        x, y, K_coeff=K_coeff, d_z=d_z[0], I_coeff=I_coeff, illum=illum)
    u1, v1, as1 = angular_spectrum(
        x, y, K_coeff=K_coeff, d_z=d_z[1], I_coeff=I_coeff, illum=illum)
    u2, v2, as2 = angular_spectrum(
        x, y, K_coeff=K_coeff, d_z=d_z[2], I_coeff=I_coeff, illum=illum)

    aspectrum = np.array([as0, as1, as2])
    beam = np.abs(aspectrum) ** 2
    beam_norm = np.array([beam[i] / beam[i].max() for i in range(3)])

    u0_deg = wavevector_to_degree(u0, lam) * angle_coeff
    u1_deg = wavevector_to_degree(u1, lam) * angle_coeff
    u2_deg = wavevector_to_degree(u2, lam) * angle_coeff
    v0_deg = wavevector_to_degree(v0, lam) * angle_coeff
    v1_deg = wavevector_to_degree(v1, lam) * angle_coeff
    v2_deg = wavevector_to_degree(v2, lam) * angle_coeff

    extent0 = [u0_deg.min(), u0_deg.max(), v0_deg.min(), v0_deg.max()]
    extent1 = [u1_deg.min(), u1_deg.max(), v1_deg.min(), v1_deg.max()]
    extent2 = [u2_deg.min(), u2_deg.max(), v2_deg.min(), v2_deg.max()]

    u0_grid, v0_grid = np.meshgrid(u0_deg, v0_deg)
    u1_grid, v1_grid = np.meshgrid(u1_deg, v1_deg)
    u2_grid, v2_grid = np.meshgrid(u2_deg, v2_deg)

    fig, ax = plt.subplots(ncols=3, figsize=(14, 4))

    levels = 8  # number of colour lines
    shrink = 0.9

    im0 = ax[0].imshow(beam_norm[0], extent=extent0)
    ax[0].contour(u0_grid, v0_grid, beam_norm[0], levels)
    fig.colorbar(im0, ax=ax[0], shrink=shrink)
    ax[0].set_title(
        '$|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z_m[0], 3)) +'$ (m)')

    im1 = ax[1].imshow(beam_norm[1], extent=extent1)
    ax[1].contour(u1_grid, v1_grid, beam_norm[1], levels)
    fig.colorbar(im1, ax=ax[1], shrink=shrink)
    ax[1].set_title(
        '$|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(d_z_m[1]) + '$ (m)')

    im2 = ax[2].imshow(beam_norm[2], extent=extent2)
    ax[2].contour(u2_grid, v2_grid, beam_norm[2], levels)
    fig.colorbar(im2, ax=ax[2], shrink=shrink)
    ax[2].set_title(
        '$|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z_m[2], 3)) + '$ (m)')

    for _ax in ax:
        _ax.set_ylabel('$v$ (' + uv_title + ')')
        _ax.set_xlabel('$u$ (' + uv_title + ')')
        _ax.set_ylim(-0.15 * angle_coeff, 0.15 * angle_coeff)
        _ax.set_xlim(-0.15 * angle_coeff, 0.15 * angle_coeff)
        # _ax.set_ylim(-0.05 * angle_coeff, 0.05 * angle_coeff)  # f= 32 GHz
        # _ax.set_xlim(-0.05 * angle_coeff, 0.05 * angle_coeff)

    # fig.set_tight_layout(True)
    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.09, bottom=0.1, right=0.99,
        top=0.9, wspace=0.23, hspace=0.2
        )

    return fig


def plot_data(u_b_data, v_b_data, beam_data, title, d_z_m, rad=False):

    if rad:
        angle_coeff = 1
        uv_title = 'radians'
    else:
        angle_coeff = 180 / np.pi
        uv_title = 'degrees'

    fig, ax = plt.subplots(ncols=3, figsize=(14, 3.5))

    levels = 10
    shrink = 0.77

    xi_minus = np.linspace(
        u_b_data[0].min(), u_b_data[0].max(), 300) * angle_coeff
    yi_minus = np.linspace(
        v_b_data[0].min(), v_b_data[0].max(), 300) * angle_coeff
    zi_minus = griddata(
        u_b_data[0] * angle_coeff, v_b_data[0] * angle_coeff,
        beam_data[0], xi_minus, yi_minus, interp='linear')

    extent0 = [xi_minus.min(), xi_minus.max(), yi_minus.min(), yi_minus.max()]

    im0 = ax[0].imshow(zi_minus, extent=extent0)
    ax[0].contour(xi_minus, yi_minus, zi_minus, levels)

    cb0 = fig.colorbar(im0, ax=ax[0], shrink=shrink)
    ax[0].set_title(
        '$|P(u,v)|^2$ $d_z=' + str(round(d_z_m[0], 3)) + '$ (m)')

    xi = np.linspace(u_b_data[1].min(), u_b_data[1].max(), 300) * angle_coeff
    yi = np.linspace(v_b_data[1].min(), v_b_data[1].max(), 300) * angle_coeff
    zi = griddata(
        u_b_data[1] * angle_coeff, v_b_data[1] * angle_coeff,
        beam_data[1], xi, yi, interp='linear')

    extent1 = [xi.min(), xi.max(), yi.min(), yi.max()]

    im1 = ax[1].imshow(zi, extent=extent1)
    ax[1].contour(xi, yi, zi, levels)
    cb1 = fig.colorbar(im1, ax=ax[1], shrink=shrink)
    ax[1].set_title('$|P(u,v)|^2$ $d_z=' + str(d_z_m[1]) + '$ (m)')

    xi_plus = np.linspace(
        u_b_data[2].min(), u_b_data[2].max(), 300) * angle_coeff
    yi_plus = np.linspace(
        v_b_data[2].min(), v_b_data[2].max(), 300) * angle_coeff
    zi_plus = griddata(
        u_b_data[2] * angle_coeff, v_b_data[2] * angle_coeff,
        beam_data[2], xi_plus, yi_plus, interp='linear')

    extent2 = [xi_plus.min(), xi_plus.max(), yi_plus.min(), yi_plus.max()]

    im2 = ax[2].imshow(zi_plus, extent=extent2)
    ax[2].contour(xi_plus, yi_plus, zi_plus, levels)
    cb2 = fig.colorbar(im2, ax=ax[2], shrink=shrink)
    ax[2].set_title('$|P(u,v)|^2$ $d_z=' + str(round(d_z_m[2], 3)) + '$ (m)')

    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.05, bottom=0.15, right=1,
        top=0.91, wspace=0.13, hspace=0.2
        )

    for _ax in ax:
        _ax.set_ylabel('$v$ (' + uv_title + ')')
        _ax.set_xlabel('$u$ (' + uv_title + ')')

    for _cb in [cb0, cb1, cb2]:
            _cb.formatter.set_powerlimits((0, 0))
            _cb.update_ticks()

    return fig


def plot_data_path(pathfits, save=False, rad=False):

    name = find_name_path(pathfits)[1]
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
        u_b_data=u_data,
        v_b_data=v_data,
        beam_data=beam_data,
        title=name + ' observed beam',
        d_z_m=d_z_m,
        rad=rad)

    if not os.path.exists(path + '/OOF_out'):
        os.makedirs(path + '/OOF_out')

    if save:
        fig_data.savefig(path + '/OOF_out/obsbeam.pdf')

    return fig_data


def plot_fit_path(pathoof, order, save=False, rad=False):

    n = order
    fitpar = ascii.read(pathoof + '/fitpar_n' + str(n) + '.dat')
    fitinfo = ascii.read(pathoof + '/fitinfo_n' + str(n) + '.dat')

    params_solution = np.array(fitpar['parfit'])

    # Grid to plot the data
    box_size = 500
    x_cal = np.linspace(-box_size, box_size, 2 ** 10)
    y_cal = np.linspace(-box_size, box_size, 2 ** 10)

    d_z_m = np.array(
        [fitinfo['d_z-'][0], fitinfo['d_z0'][0], fitinfo['d_z+'][0]])
    name = fitinfo['name'][0]

    fig_beam = plot_beam(
        params=params_solution,
        title=name + ' fitted beam $n=' + str(n) + '$',
        x=x_cal,
        y=y_cal,
        d_z_m=d_z_m,
        lam=fitinfo['wavelength'][0],
        illum=fitinfo['illum'][0],
        rad=rad
        )

    fig_phase = plot_phase(
        params=params_solution,
        d_z_m=d_z_m[2],  # only one function for the three beam maps
        title=name + ' Aperture phase distribution $n=' + str(n) + '$',
        notilt=True
        )

    if save:
        fig_beam.savefig(pathoof + '/fitbeam_n' + str(n) + '.pdf')
        fig_phase.savefig(pathoof + '/fitphase_n' + str(n) + '.pdf')

    return fig_beam, fig_phase


def plot_fit_nikolic_path(pathoof, order, d_z_m, title, save=False):

    n = order
    path = pathoof + '/z' + str(n)
    params_nikolic = np.array(
        fits.open(path + '/fitpars.fits')[1].data['ParValue'])

    wavelength = fits.open(path + '/aperture-notilt.fits')[0].header['WAVE']

    # Grid to plot the data
    box_size = 500
    x_cal = np.linspace(-box_size, box_size, 2 ** 10)
    y_cal = np.linspace(-box_size, box_size, 2 ** 10)

    fig_beam = plot_beam(
        params=params_nikolic,
        title=title + ' phase distribution Nikolic fit $n=' + str(n) + '$',
        x=x_cal,
        y=y_cal,
        d_z_m=d_z_m,
        rad=False,
        lam=wavelength,
        illum='nikolic'
        )

    fig_phase = plot_phase(
        params=params_nikolic,
        d_z_m=d_z_m[2],
        title=title + ' Nikolic fit $n=' + str(n) + '$',
        notilt=True
        )

    if save:
        fig_beam.savefig(
            '/Users/tomascassanelli/Desktop/nikolic_beam_n' + str(n) + '.pdf')
        fig_phase.savefig(
            '/Users/tomascassanelli/Desktop/nikolic_phase_n' + str(n) + '.pdf')

    return fig_beam, fig_phase


def plot_phase(params, d_z_m, title, notilt=True):

    K_coeff = params[4:]

    if notilt:
        K_coeff[0] = 0
        K_coeff[1] = 0
        subtitle = (
            '$\phi_\mathrm{no\,tilt}(x, y)$ $d_z=\pm' + str(round(d_z_m, 3)) +
            '$ (m)')
        bartitle = '$\phi_\mathrm{no\,tilt}(x, y)$ amplitude (rad)'
    else:
        subtitle = '$\phi(x, y)$ $d_z=\pm' + str(round(d_z_m, 3)) + '$ (m)'
        bartitle = '$\phi(x, y)$ amplitude (rad)'

    pr = 50
    x = np.linspace(-pr, pr, 1e3)
    y = np.linspace(-pr, pr, 1e3)

    x_grid, y_grid = np.meshgrid(x, y)

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / pr

    extent = [x.min(), x.max(), y.min(), y.max()]
    _phi = phi(theta=t, rho=r_norm, K_coeff=K_coeff) * antenna_shape(x_grid, y_grid)

    fig, ax = plt.subplots()

    levels = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    # as used in N. software

    # levels = np.linspace(-2, 2, 10)  # number of colour lines
    shrink = 1

    im = ax.imshow(_phi, extent=extent)
    ax.contour(x_grid, y_grid, _phi, levels=levels)
    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    ax.set_title(subtitle)
    ax.set_ylabel('$y$ (m)')
    ax.set_xlabel('$x$ (m)')
    cb.ax.set_ylabel(bartitle)
    fig.suptitle(title)

    return fig


if __name__ == "__main__":

    # from astropy.table import Table

    # # Comparison between parameters
    # n_z_coeff = params_fitted.size - 4
    # n = int((np.sqrt(1 + 8 * n_z_coeff) - 3) / 2)

    # ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    # L = np.array(ln)[:, 0]
    # N = np.array(ln)[:, 1]

    # params_name = ['Illum amplitude', 'sigma_r', 'x_0', 'y_0']
    # for i in range(n_z_coeff):
    #     params_name.append('U(' + str(L[i]) + ',' + str(N[i]) + ')')

    # table = Table(
    #     {'Coefficient': params_name, 'Nikolic': params_nikolic,
    #     'Fit': params_fitted}, names=['Coefficient', 'Nikolic', 'Fit'])
    # print(table)
    path = '/Users/tomascassanelli/ownCloud/OOF/test_data/S28mm_9959_3C84_SB/oofout/S28mm_9959_3C84_SB2-000'

    # d_z_m = np.array([-0.077, 0, 0.077])
    # title = 'S28mm_9959_3C84_SB2'

    # for order in [5, 6, 7]:
    #     plot_fit_nikolic_path(
    #         pathoof=path,
    #         order=order,
    #         title=title,
    #         d_z_m=d_z_m,
    #         save=True)

    plot_data_path(
        pathfits='/Users/tomascassanelli/ownCloud/OOF/test_data/gen_data/gendata_o5n1.fits',
        save=False,
        rad=False)

    plt.show()
