#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits, ascii
from scipy import interpolate
import yaml
import pyoof

# Calling plot style from pyoof package
plot_style = os.path.join(os.path.dirname(pyoof.__file__), 'pyoof.mplstyle')
plt.style.use(plot_style)


def plot_data_effelsberg(pathfits):
    """
    Plot all data from an OOF Effelsberg observation given the path.
    """

    data_info, data_obs = pyoof.extract_data_effelsberg(pathfits)
    [name, obs_object, obs_date, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    fig_data = pyoof.plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        d_z=d_z,
        angle='degrees',
        title=str(obs_object) + ' observed beam',
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
        print('max displacement: {}'.format(phase_mum[str(i)].max()))
        print('min displacement: {}\n'.format(phase_mum[str(i)].min()))

    vmax = np.max(maxi)
    vmin = np.min(mini)
    levels = np.linspace(vmin, vmax, 9)
    # levels = np.linspace(-2, 2, 9) * 4 * np.pi / 0.009 * 1e-6

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
        phase_rot90 = np.rot90(phase)

        ax[j].imshow(
            X=phase_rot90,
            extent=extent,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
            )
        ax[j].contour(
            x_ng, y_ng, phase_rot90, colors='k', alpha=0.3, levels=levels
            )

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

    fig, ax = plt.subplots()

    im = ax.imshow(phase_um, extent=extent)
    ax.contour(pts[0], pts[1], phase_um, levels=levels, colors='k', alpha=0.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(
        '$\\varphi_{\\scriptsize{\\textrm{no-tilt}}}(x,y)$ amplitude $\mu$m'
        )
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()

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

    fig.tight_layout()

    return fig


def plot_rse(paths_pyoof, order, r_max, taper, obs):
    """
    Computes the random-surface-error efficiency using Ruze's formula for
    several observations at a time. You need to provide a list the OOF_out
    directory.
    """
    # Generating mesh
    sr = 3.25  # sub-reflector radius
    x = np.linspace(-sr, sr, 1e3)
    y = x
    xx, yy = np.meshgrid(x, y)

    source_symbol = {'3C454.3': '^', '3C273': 'o', '3C84': 's'}
    obs_type = {'FEM': '#1f2db4', 'no-FEM': '#b41f2d', 'OOF': '#1fb4a6'}

    _source, _meanel, _rse, _rse_tot = [], [], [], []
    for path in paths_pyoof:

        # Importing result from pyoof package
        phase_imported = np.genfromtxt(path + '/phase_n{}.csv'.format(order))
        phase_imported[xx ** 2 + yy ** 2 > r_max ** 2] = 0

        # Extracting K_coeff error from variance-covariance
        cov = np.genfromtxt(path + '/cov_n{}.csv'.format(order))
        _error = np.sqrt(np.diag(cov[1:]))
        params_used = np.array([int(i) for i in cov[:1][0]])

        if not (params_used == 4).any():
            K_err = np.insert(_error[params_used >= 4], 0, 0)
        else:
            K_err = _error[params_used >= 4]

        # Computing the phase of the uncertainties
        phase_err = pyoof.aperture.phase(
            K_coeff=K_err,
            notilt=True,
            pr=50  # adding the original, same as computed in pyoof
            )[2]
        phase_err[xx ** 2 + yy ** 2 > r_max ** 2] = 0

        # Phase taper to reduce edges
        if taper:
            I_coeff = ascii.read(
                path + '/fitpar_n' + str(order) + '.csv')['parfit'][:4]
            Ea = pyoof.aperture.illum_pedestal(xx, yy, I_coeff, sr)
            _phase = phase_imported * Ea
            _phase_err = phase_err * Ea
        else:
            _phase = phase_imported
            _phase_err = phase_err

        # importing phase related data
        with open(path + '/pyoof_info.yaml', 'r') as inputfile:
            pyoof_info = yaml.load(inputfile)

        _source.append(pyoof_info['obs_object'])
        _meanel.append(pyoof_info['meanel'])
        _rse.append(pyoof.aperture.e_rse(_phase))
        _rse_tot.append(pyoof.aperture.e_rse(_phase + _phase_err))

    source = np.array(_source, dtype='<U8')
    meanel = np.array(_meanel, dtype='<f8')
    rse = np.array(_rse, dtype='<f8')
    rse_tot = np.array(_rse_tot, dtype='<f8')
    yerr = rse_tot - rse

    fig, ax = plt.subplots()
    for name in set(source):
        ax.errorbar(
            x=meanel[source == name],
            y=rse[source == name],
            yerr=yerr[source == name],
            fmt=source_symbol[name],
            color=obs_type[obs],
            fillstyle=None,
            label=name,
            capsize=3,
            capthick=0.3,
            markersize=4
            )
    if taper:
        title = '{} + taper $n={}$'.format(obs, order)
    else:
        title = '{}  $n={}$'.format(obs, order)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper right', numpoints=1)

    ax.set_xlabel('$\\alpha$ degrees')
    ax.set_title(title)
    ax.set_ylabel(
        'Random-surface-error efficiency ' +
        '$\\varepsilon_{\\scriptsize{\\textrm{rs}}}$')

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

    import glob

    obs_noFEM = glob.glob('../../data/S9mm_noFEM/OOF_out/*001')
    obs_nobump2 = glob.glob('../../data/S9mm_nobump2/OOF_out/*001')
    obs_FEM = glob.glob('../../data/S9mm_FEM/OOF_out/*001')

    obs_noFEM_BW = glob.glob('../../data/S9mm_noFEM/OOF_out/*BW-001*')

    a=['../../data/S9mm_noFEM/OOF_out/S9mm_3800-3807_3C84_48deg_H6_LON-001', '../../data/S9mm_noFEM/OOF_out/S9mm_3800-3819_3C84_53deg_H6_BW-001', '../../data/S9mm_noFEM/OOF_out/S9mm_3812-3819_3C84_58deg_H6_LAT-001']

    plot_rse(
        paths_pyoof=a,
        order=7,
        r_max=3.25,  # r_max = 3.25 m
        taper=True,
        obs='no-FEM',
        )

    # plot_lookup_effelsberg('../../data/S9mm_bump2/lookup_bump2')

    plt.show()
