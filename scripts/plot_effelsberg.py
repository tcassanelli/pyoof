#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
import yaml
import pyoof

# Calling plot style from pyoof package
plot_style = os.path.join(os.path.dirname(pyoof.__file__), 'pyoof.mplstyle')
plt.style.use(plot_style)


def plot_data_effelsberg(pathfits, angle):
    """
    Plot all data from an OOF Effelsberg observation given the path.
    """

    data_info, data_obs = pyoof.extract_data_effelsberg(pathfits)
    [name, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    fig_data = pyoof.plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        d_z=d_z,
        angle=angle,
        title=pyoof.str2LaTeX(name) + ' observed beam',
        res_mode=False
        )

    return fig_data


def plot_lookup_effelsberg(path_lookup):
    """
    Effelsberg look-up table plot. All degrees with exception of 32. Since it
    is almost zero value. The file must be txt or similar.
    """
    elevation = [7, 10, 20, 30, 32, 40, 50, 60, 70, 80, 90]  # degrees
    usecols = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    dtype = [('actuator', '<i8')] + [(str(i), '<f8') for i in elevation]
    # Actuator phase in displacement

    phase_mum = np.genfromtxt(path_lookup, usecols=usecols, dtype=dtype)

    # Generating the mesh from technical drawings
    theta = np.radians(np.linspace(7.5, 360 - 7.5, 24))
    correct = 0  # correction in mm for the last ring
    R = np.array([3250 - correct, 2600, 1880, 1210]) * 1e-3
    # Actuator positions
    act_x = np.outer(R, np.cos(theta)).reshape(-1)
    act_y = np.outer(R, np.sin(theta)).reshape(-1)

    # sub-reflector radius
    sr = 3.25
    # new grid
    x_ng = np.linspace(-sr, sr, 1e3)
    y_ng = x_ng

    xx, yy = np.meshgrid(x_ng, y_ng)
    circ = [(xx ** 2 + yy ** 2) >= 3.25 ** 2]

    extent = [-sr, sr] * 2

    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(10, 4.2))
    fig.subplots_adjust(
        left=0, bottom=0.01, right=1,
        top=.94, wspace=.04, hspace=0.18
        )
    ax = axes.flat

    maxi, mini = [], []
    for i in elevation:
        maxi.append(phase_mum[str(i)].max())
        mini.append(phase_mum[str(i)].min())

        print('Elev. {} RMS microns: '.format(i), pyoof.rms(phase_mum[str(i)]))

    vmax = np.max(maxi)
    vmin = np.min(mini)
    levels = np.linspace(vmin, vmax, 10)

    elevation_plot = [7] + [i for i in range(10, 100, 10)]
    for j in range(len(elevation_plot)):

        phase = interpolate.griddata(
            # coordinates of grid points to interpolate from.
            points=(act_x, act_y),
            values=phase_mum[str(elevation_plot[j])],
            # coordinates of grid points to interpolate to.
            xi=tuple(np.meshgrid(x_ng, y_ng)),
            method='cubic'
            )

        phase[circ] = 0

        ax[j].imshow(
            X=phase,
            extent=extent,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
            )
        ax[j].contour(x_ng, y_ng, phase, colors='k', alpha=0.3, levels=levels)

        circle = plt.Circle(
            (0, 0), sr, color='#24848d', fill=False, alpha=1, linewidth=1.3
            )
        ax[j].add_artist(circle)

        circle2 = plt.Circle(
            (0, 0), sr, color='k', fill=False, alpha=0.3, linewidth=0.8
            )
        ax[j].add_artist(circle2)

        ax[j].set_title('$\\alpha=' + str(elevation_plot[j]) + '$ degrees')
        ax[j].grid('off')
        ax[j].xaxis.set_major_formatter(plt.NullFormatter())
        ax[j].yaxis.set_major_formatter(plt.NullFormatter())
        ax[j].xaxis.set_ticks_position('none')
        ax[j].yaxis.set_ticks_position('none')

    return fig


def plot_phase_um(pts, phase, wavel, act, act_name, title, show_actuator):
    """
    Plot the phase in microns for the sub-reflector, in the future the correct
    convention need to be added to the phase error map.
    """

    # input interpolation function is the real beam grid
    phase_um = phase * wavel / (4 * np.pi) * 1e6
    levels = np.linspace(-2, 2, 9) * wavel / (4 * np.pi) * 1e6
    extent = [-3.25, 3.25] * 2
    shrink = 1

    fig, ax = plt.subplots()

    im = ax.imshow(phase_um, extent=extent)
    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    cb.ax.set_ylabel('$\\varphi_{\\bot\mathrm{no\,tilt}}(x, y)$ amplitude $\mu$m')
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()

    ax.contour(pts[0], pts[1], phase_um, levels=levels, colors='k', alpha=0.3)

    if show_actuator:
        ax.scatter(act[0], act[1], c='r', s=5)
        for i in range(act_name.size):
            ax.annotate(
                s=act_name[i],
                xy=(act[0][i] + 0.01, act[1][i] + 0.01),
                size=5
                )

    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    ax.grid('off')
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-3.7, 3.7)
    ax.set_title(title)

    return fig


def plot_rse(paths_pyoof, order, r_max, title):
    """
    Computes the random-surface-error efficiency using Ruze's formula for
    several observations at a time. You need to provide a list the OOF_out
    directory.
    """
    # Generating mesh
    x = np.linspace(-3.25, 3.25, 1e3)
    y = x
    xx, yy = np.meshgrid(x, y)

    values = []
    for path in paths_pyoof:
        phase = np.genfromtxt(path + '/phase_n' + str(order) + '.csv')
        phase[xx ** 2 + yy ** 2 > r_max ** 2] = 0

        # importing phase related data
        with open(path + '/pyoof_info.yaml', 'r') as inputfile:
            pyoof_info = yaml.load(inputfile)


        rad_to_um = pyoof_info['wavel'] / (4 * np.pi)
        phase_m = phase * rad_to_um

        values.append([
            pyoof_info['name'], pyoof_info['meanel'],
            pyoof.aperture.e_rse(phase_m, pyoof_info['wavel'])
            ])

    fig, ax = plt.subplots()

    for name, meanel, rse in values:
        ax.plot(meanel, rse, 'o', label=pyoof.str2LaTeX(name))

        ax.legend(loc='best')
        ax.set_xlabel('$\\alpha$ degrees')
        ax.set_title(title)
        ax.set_ylabel(
            'Random-surface-error efficiency $\\varepsilon_\mathrm{rs}$'
            )

    return fig


# take a closer look at this and make it better!
# Needs corrections in the phase and right convention on parameters!
def plot_fit_nikolic(pathoof, order, d_z, telgeo, plim_rad, angle):
    """
    Designed to plot Bojan Nikolic solutions given a path to its oof output.
    telgeo = [block, delta, pr]
    """

    def illum_nikolic(x, y, I_coeff, pr):
        i_amp, sig_dB, x0, y0 = I_coeff
        sig_r = sig_dB
        Ea = i_amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / pr ** 2 * sig_r)
        return Ea

    path = pathoof + '/z' + str(order)
    params_nikolic = np.array(
        fits.open(path + '/fitpars.fits')[1].data['ParValue']
        )
    wavel = fits.open(path + '/aperture-notilt.fits')[0].header['WAVE']

    fig_beam = pyoof.plot_beam(
        params=params_nikolic,
        d_z=d_z,
        wavel=wavel,
        illum_func=illum_nikolic,
        telgeo=telgeo,
        resolution=2 ** 8,
        title='Power pattern Nikolic fit $n=' + str(order) + '$',
        plim_rad=plim_rad,
        angle=angle
        )

    fig_phase = pyoof.plot_phase(
        K_coeff=params_nikolic[4:],
        d_z=d_z[2],
        notilt=True,
        pr=telgeo[2],
        title='Aperture phase distribution Nikolic fit $n=' + str(order) + '$'
        )

    return fig_beam, fig_phase


if __name__ == '__main__':

    plot_rse(['/Users/tomascassanelli/ownCloud/OOF/data/S9mm_bump/OOF_out/S9mm_3478_3C454.3_32deg_H6-011'], 2, 3.25, 'hi')
    plt.show()
