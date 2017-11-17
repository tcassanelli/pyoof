#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from astropy.io import ascii
import yaml
from scipy import interpolate
import pyoof
import plot_effelsberg


# In the future make it independent of the plot function
def actuator_displacement(path_pyoof_out, order, edge_mm, make_plots):
    """
    Calculates displacement for the active surface on the Effelsberg telescope.
    Given the phase error from the primary reflector the phase error for the
    sub-reflector computed and it returns a list for the specific displacement
    for each of the 92 actuators.
    """

    print('\n ***** ACTUATOR DISPLACEMENT ***** \n')

    # Input data with its initial grid
    n = order
    phase = np.genfromtxt(path_pyoof_out + '/phase_n' + str(n) + '.csv')
    sr = 3.25  # sub-reflector radius m
    phase_size = phase.shape[0]
    x = np.linspace(-sr, sr, phase_size)
    y = x
    xx, yy = np.meshgrid(x, y)

    # Grid that wants to be calculated
    # Generating the mesh from technical drawings
    # theta = np.radians(np.linspace(7.5, 360 - 7.5, 24))
    # only changing theta you can rotate the actuators
    theta = np.radians(np.linspace(7.5, 360 - 7.5, 24) - 90)
    external_ring = 3250 - edge_mm  # real external ring of actuators mm
    R = np.array([external_ring, 2600, 1880, 1210]) * 1e-3  # meters

    # Actuator positions
    act_x = np.outer(R, np.cos(theta)).reshape(-1)
    act_y = np.outer(R, np.sin(theta)).reshape(-1)
    act_name = np.array(range(1, 97), dtype='<U3')

    # Interpolation
    intrp = interpolate.RegularGridInterpolator(
        points=(x, y),  # points defining grid
        values=phase.T,  # data on a grid
        method='linear'  # linear or nearest
        )

    act_phase = intrp(np.array([act_x, act_y]).T)

    # importing phase related data
    with open(path_pyoof_out + '/pyoof_info.yaml', 'r') as inputfile:
        pyoof_info = yaml.load(inputfile)

    rad_to_um = pyoof_info['wavel'] / (4 * np.pi) * 1e6  # converted to microns

    # Storing the data
    path_actdisp = path_pyoof_out + '/actdisp_n' + str(n) + '.csv'
    act_to_save = [act_name, act_x, act_y, act_phase, act_phase * rad_to_um]

    print('File name: ', pyoof_info['name'])
    print('Obs Wavelength : ', pyoof_info['wavel'], 'm')
    print('d_z (out-of-focus): ', pyoof_info['d_z'], 'm')
    print('Mean elevation: ', pyoof_info['meanel'], 'degrees')
    print('Considered radius: ', external_ring, 'microns')

    # RMS for the maximum radius
    _phase_microns = phase.copy() * rad_to_um
    _phase_microns[xx ** 2 + yy ** 2 > (external_ring * 1e-3) ** 2] = 0
    print('RMS: ', pyoof.rms(_phase_microns), 'microns')
    print('Mean: ', np.mean(_phase_microns), 'microns')

    # for the efficiency everything is computed in radians
    print('Random-surface-error efficiency: ', pyoof.aperture.e_rse(phase))
    print('\n')

    ascii.write(
        table=act_to_save,
        output=path_actdisp,
        names=['actuator', 'act_x', 'act_y', 'phase_rad', 'disp_um'],
        )

    # printing the full table
    ascii.read(path_actdisp).pprint(max_lines=-1, max_width=-1)

    if make_plots:
        fig_act = plot_effelsberg.plot_phase_um(
            pts=(x, y),
            phase=phase,  # phase in microns
            wavel=pyoof_info['wavel'],
            act=(act_x, act_y),
            act_name=act_name,
            show_actuator=True,
            title=(
                'Actuators amplitude $n={}$ $\\alpha={}$ degrees'.format(
                    n, round(pyoof_info['meanel'], 2)))
            )

        fig_act.savefig(
            filename=path_pyoof_out + '/plots/actdisp_n' + str(n) + '.pdf',
            bbox_inches='tight'
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Get solution from fit_beam.py
    path_pyoof_out = '../../data/S9mm_noFEM/pyoof_out/S9mm_3800-3819_3C84_53deg_H6_BW-001'

    for n in range(1, 10):
        displ = actuator_displacement(
            path_pyoof_out=path_pyoof_out,
            order=n,
            edge_mm=0,  # mm
            make_plots=True
            )

        plt.close('all')
