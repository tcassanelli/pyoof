#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import interpolate
from .aux_functions import str2LaTeX

# Import plot style matplotlib, change to same directory in future
# plt.style.use('../plot_gen_thesis/master_thesis_sty.mplstyle')


__all__ = [
    'actuator_displacement', 'plot_actuator',
    ]


def actuator_displacement(pathoof, order, save):

    print('\n ##### OOF ACTUATOR DISPLACEMENT ##### \n')

    # Input data with its initial grid
    n = order
    phase = np.genfromtxt(pathoof + '/phase_n' + str(n) + '.csv')
    sr = 3.25  # sub-reflector radius m
    phase_size = phase.shape[0]
    x = np.linspace(-sr, sr, phase_size)
    y = x

    # Grid that wants to be calculated
    # Generating the mesh from technical drawings
    theta = np.radians(np.linspace(7.5, 360 - 7.5, 24))

    correct = 5  # correction in mm for the last ring
    R = np.array([1210, 1880, 2600, 3250 - correct]) * 1e-3

    # Actuator positions
    act_x = np.outer(R, np.cos(theta)).reshape(-1)
    act_y = np.outer(R, np.sin(theta)).reshape(-1)

    act_name = np.core.defchararray.add(
        'PVA', np.array(range(1, 97), dtype='<U3')
        )

    # Interpolation
    intrp = interpolate.RegularGridInterpolator(
        points=(x, y),  # points defining grid
        values=phase.T,  # data on a grid
        method='linear'  # linear or nearest
        )

    act_phase = intrp(np.array([act_x, act_y]).T)

    # transforming to distance
    wavel = ascii.read(pathoof + '/fitinfo.dat')['wavel'][0]
    name = ascii.read(pathoof + '/fitinfo.dat')['name'][0]
    name_LaTeX = str2LaTeX(name)

    rad_to_um = wavel / (4 * np.pi) * 1e6  # converted to microns

    # Test plot for interpolation
    # extent = [act_x.min(), act_x.max(), act_y.min(), act_y.max()]
    # plt.imshow(act_phase, extent=extent)

    # Storing the data
    path_actdisp = pathoof + '/actdisp_n' + str(n) + '.dat'
    act_to_save = [act_name, act_x, act_y, act_phase, act_phase * rad_to_um]

    ascii.write(
        table=act_to_save,
        output=path_actdisp,
        names=['actuator', 'act_x', 'act_y', 'disp_rad', 'disp_um']
        )

    # printing the full table
    ascii.read(path_actdisp).pprint(max_lines=-1, max_width=-1)

    fig_act = plot_actuator(
        phase=phase,  # phase in microns
        rad_to_um=rad_to_um,
        pts=[x, y],
        act=[act_x, act_y],
        act_name=act_name,
        show_label=True,
        title='Actuators ' + name_LaTeX + ' $n=' + str(n) + '$'
        )

    if save:
        fig_act.savefig(
            filename=pathoof + '/actdisp_n' + str(n) + '.pdf',
            bbox_inches='tight'
            )


def plot_actuator(phase, rad_to_um, pts, act, act_name, show_label, title):

    # input interpolation function is the real beam grid
    phase_um = phase * rad_to_um
    levels = np.linspace(-2, 2, 9) * rad_to_um
    sr = 3.25
    extent = [-sr, sr, -sr, sr]
    shrink = 1

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(phase_um, extent=extent)

    cb = fig.colorbar(im, ax=ax, shrink=shrink)
    cb.ax.set_ylabel('$\phi_\mathrm{no\,tilt}(x, y)$ amplitude $\mu$m')
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()

    ax.contour(pts[0], pts[1], phase_um, levels=levels, colors='k', alpha=0.5)
    ax.scatter(act[0], act[1], c='r', s=5)

    if show_label:
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


if __name__ == "__main__":

    # Get solution from fit_beam.py
    pathoof = '../data/S9mm/OOF_out/S9mm_0397_3C84_H1_SB'
    n = 5

    displ = actuator_displacement(
        pathoof=pathoof,
        order=n,
        save=True
        )
    plt.show()
