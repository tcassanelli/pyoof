#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli

"""
This is a developer module at the moment and only Effeslberg functions.
"""

import os
import numpy as np
from astropy.io import ascii
import yaml
from astropy import units as apu
from astropy.utils.data import get_pkg_data_filename
from scipy import interpolate
from ..math_functions import rms
from ..aperture import e_rs

__all__ = [
    'sr_actuators', 'actuator_displacement', 'plot_actuator_displacement'
    ]

act_data_csv = get_pkg_data_filename('data/act_effelsberg.csv')

# TODO: perhaps it'll be a good idea to change structure to classes for the
# telescope geometry?
# class telescope(object):
#     def __init__(self):
#         self.name = 'effelsberg'
#         self.pr = 50 * apu.m
#         self.sr = 3.25 * apu.m


def generate_actuators(edge):
    """
    This function generates the actuators coordinates for the sub-reflector at
    the Effelsberg telecope.
    """

    # mesh for the active surface at the Effelsberg telescope
    theta = np.linspace(7.5, 360 - 7.5, 24) * apu.deg - 90 * apu.deg
    R = [3250, 2600, 1880, 1210] * apu.mm

    if edge is not None:
        R[0] -= edge.to(apu.mm)

    # Actuator positions
    act_x = np.outer(R.to_value(apu.mm), np.cos(theta)).reshape(-1)
    act_y = np.outer(R.to_value(apu.mm), np.sin(theta)).reshape(-1)
    act_name = np.array(range(1, 97), dtype='<U3')

    act_data = [act_name, act_x, act_y]

    return act_data


def sr_actuators(phase, wavel):
    """
    After the `~pyoof` packages calculates the overall phase-error of the radio
    telescope, the output file `phase_n<iter>.csv` has the phase information
    in radians, with `<iter>`, the order of Zernike circle polynomials
    coefficients used. Note that this formula is a rough approximation, only
    for a Cassegrain/Gregorian configuration. It assumes that the reflected
    rays are exactly perpendicular to the sub-reflector, which is not the case.
    This function is specially useful for an active surface working in the
    telescope's sub-reflector.

    Parameters
    ----------
    phi : `~astropy.units.quantity.Quantity`
        Aperture phase distribution, :math:`\\varphi(x, y)`, for an specific
        primary dish radius, measured in radians.
    wavel : `~astropy.units.quantity.Quantity`
        Wavelength, :math:`\\lambda`, of the observation in length units.

    Returns
    -------
    phase_ad : `~astropy.units.quantity.Quantity`
        Phase-error in terms of perpendicular displacement in the sub-reflector
        of a Cassegrain/Gregorian telescope. It is measured in units of length.
        This phase can be applied to an active surface to counteract the
        (mainly primary dish) deformations.
    """

    # phase in terms of the actuators displacement
    phase_ad = phase * wavel / (4 * np.pi * apu.rad)
    # the extra 2 is given by the double path carried by the light ray.

    return phase_ad.to(apu.um)


def actuator_displacement(
    path_pyoof_out, order, edge=None, make_plots=True
        ):
    """
    Calculates displacement for the active surface at the Effelsberg telescope.
    Given the phase error (output from `~pyoof.fit_zpoly`), the phase errors
    in the sub-reflector can be approximated and specified as perpendicualar
    displacement for a set of 92 actuators in the current active surface.
    """

    print('\n ***** ACTUATOR DISPLACEMENT ***** \n')

    # reading the pyoof output phase
    phase = np.genfromtxt(
        os.path.join(path_pyoof_out, 'phase_n{}.csv'.format(order))
        )
    phase = phase * apu.rad  # stored as default radians

    pr = 50 * apu.m    # Primary refelctor
    sr = 3.25 * apu.m  # sub-reflector radius
    x = np.linspace(-sr, sr, phase.shape[0])
    y = x.copy()

    # mesh for the active surface at the Effelsberg telescope
    # theta = np.linspace(7.5, 360 - 7.5, 24) * apu.deg - 90 * apu.deg
    # R = [3250, 2600, 1880, 1210] * apu.mm

    # if edge is not None:
    #     R[0] -= edge.to(apu.mm)

    if edge is not None:
        act_data = generate_actuators(edge)
    else:
        act_data = ascii.read(act_data_csv)

    # Interpolation
    intrp = interpolate.RegularGridInterpolator(
        points=(x.to_value(apu.mm), y.to_value(apu.mm)),
        values=phase.to_value(apu.rad).T,  # data on a grid
        method='linear'                    # linear or nearest
        )
    act_phase = intrp(np.array([act_data['x'], act_data['y']]).T) * phase.unit

    # importing phase related data
    with open(
        os.path.join(path_pyoof_out, 'pyoof_info.yml'), 'r'
            ) as inputfile:
        pyoof_info = yaml.load(inputfile)

    wavel = pyoof_info['wavel'] * apu.m

    # phase in each actuator point transformed to perpendicular displacement
    act_phase_ad = sr_actuators_effelsberg(phase=act_phase, wavel=wavel)

    # Storing the data actuator displacement
    path_ad = path_pyoof_out + '/actdisp_n' + str(order) + '.csv'
    save_ad = [
        act_data['name'],              # Label actuator as in Effelsberg sr
        act_data['x'], act_data['y'],  # coordinates actuators [mm]
        act_phase.to_value(apu.rad),   # Phase-error
        act_phase_ad.to_value(apu.um)  # Phase to perpendicualr displacement
        ]

    print(
        'File name: {}'.format(pyoof_info['name']),
        'Order: {}'.format(order),
        'Telescope name: {}'.format(pyoof_info['telescope']),
        'File name: {}'.format(pyoof_info['name']),
        'Obs Wavelength: {} m'.format(pyoof_info['wavel']),
        'Mean elevation {} deg'.format(pyoof_info['meanel']),
        'd_z (out-of-focus): {} m'.format(pyoof_info['d_z']),
        'Illumination to be fitted: {}'.format(pyoof_info['illumination']),
        'Considered max radius: {} \n'.format(R[0]),
        sep='\n',
        end='\n'
        )

    mean_ad = sr_actuators_effelsberg(phase=phase, wavel=wavel).mean()

    print(
        'STATISTICS',
        'RMS: {}'.format(sr_actuators_effelsberg(
            phase=rms(phase=phase, radius=sr),
            wavel=wavel)
            ),
        'MEAN: {}'.format(mean_ad),
        'eff-random-surface-error: {} \n'.format(e_rs(phase=phase, radius=pr)),
        sep='\n',
        end='\n'
        )

    ascii.write(
        table=save_ad,
        output=path_ad,
        names=['name', 'x', 'y', 'phase', 'displacement'],
        overwrite=True
        )

    # printing the full table
    ascii.read(path_ad).pprint(max_lines=-1, max_width=-1)

    if make_plots:
        print('\n... Making plots ...')
        plot_actuator_displacement(
            path_pyoof_out=path_pyoof_out,
            order=order,
            title=(
                'Actuators displacement $n={}$ $\\alpha={}$ degrees'.format(
                    order, round(pyoof_info['meanel'], 2)
                    )
                ),
            act_data=act_data,
            actuators=False
            )

    print('\n **** COMPLETED **** \n')


def plot_actuator_displacement(
        path_pyoof_out, order, title, act_data=None, actuators=False
        ):
    """
    """

    # reading the pyoof output phase
    phase = np.genfromtxt(
        os.path.join(path_pyoof_out, 'phase_n{}.csv'.format(order))
        )

    phase = phase * apu.rad

    # importing phase related data
    with open(
        os.path.join(path_pyoof_out, 'pyoof_info.yml'), 'r'
            ) as inputfile:
        pyoof_info = yaml.load(inputfile)

    wavel = pyoof_info['wavel'] * apu.m
    meanel = np.around(pyoof_info['meanel'], 1)

    sr = 3.25 * apu.m
    extent = [-sr.to_value(apu.m), sr.to_value(apu.m)] * 2
    x = np.linspace(-sr, sr, phase.shape[0])
    y = np.linspace(-sr, sr, phase.shape[1])

    phase = sr_actuators_effelsberg(phase=phase, wavel=wavel)
    levels = sr_actuators_effelsberg(
        phase=np.linspace(-2, 2, 9) * apu.rad, wavel=wavel)

    fig, ax = plt.subplots()

    im = ax.imshow(phase.to_value(apu.um), extent=extent)
    ax.contour(
        x.to_value(apu.m), y.to_value(apu.m), phase.to_value(apu.um),
        level=levels.to_value(apu.um), colors='k', alpha=0.3
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(
        '$\\varphi^{(' + str(meanel) + '^{\\circ})}' +
        '_{\\scriptsize{\\textrm{no-tilt}}, \\bot}(x,y)$ amplitude $\\mu$m'
        )
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('left')
    cb.update_ticks()

    if actuators:
        if act_data is None:
            act_data = ascii.read(act_data_csv)
        act_data['x'] *= apu.mm
        act_data['y'] *= apu.mm

        ax.scatter(
            act_data['x'].to_value(apu.m), act_data['y'].to_value(apu.m),
            c='r', s=5
            )

        for i in range(act_data['name'].size):
            ax.annotate(
                s=act_data['name'][i],
                xy=(act_data['x'][i] + 0.01, act_data['y'][i] + 0.01),
                size=5
                )

        ax.set_xlim(-sr.to_value(apu.m) * 1.1, sr.to_value(apu.m) * 1.1)
        ax.set_ylim(-sr.to_value(apu.m) * 1.1, sr.to_value(apu.m) * 1.1)

    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')
    ax.grid(False)
    ax.set_title(title)

    fig.tight_layout()

    return fig
