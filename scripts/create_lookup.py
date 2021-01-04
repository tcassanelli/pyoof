#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import glob
import yaml
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.table import Table
import pyoof
from pyoof.actuator import EffelsbergActuator

"""
The purpose of this script is to create a lookup table given a set of phase
maps from the pyoof software. The method only applies to those observations
that were made with the active surface on, i.e. with corrections from FEM.
Here we isolate the true deformations to apply directly a look-up table to the
active surface.
"""

pyoof_run = '000'
path_pyoof_out = '/Users/tomascassanelli/MPIfR/OOF/data/pyoof_out'
n = 5
resolution = 1000
pr = 50 * u.m

actuators = EffelsbergActuator(
    wavel=7 * u.mm,
    nrot=1,
    sign=-1,
    order=n,
    sr=3.25 * u.m,
    pr=pr,
    resolution=resolution,
    )

# reading data from the pyoof_out
path_pyoof_out_all = glob.glob(''.join((path_pyoof_out, f'/*{pyoof_run}')))
tab = pyoof.table_pyoof_out(path_pyoof_out=path_pyoof_out_all, order=n)

tab.sort('obs-date')
tab.add_index('name')
# tab.pprint_all()

idx_offset = np.zeros((len(tab), ), dtype=bool)
for i in range(len(tab)):
    if tab['name'][i].split('-')[-1] == 'offset':
        idx_offset[i] = True

section = (
    # (tab['phase-rms'] < 1. * u.rad)
    (tab['beam-snr'] > 200.)
    & ~idx_offset
    )
tab_section = tab[section].copy()
tab_section.add_index('name')
tab_section.pprint_all()

N_K_coeff = (n + 1) * (n + 2) // 2
K_coeff_obs = np.zeros((len(tab_section), N_K_coeff), dtype=np.float64)

for j, _name in enumerate(tab_section['name']):

    path_params = os.path.join(
        path_pyoof_out, f'{_name}-{pyoof_run}', f'fitpar_n{n}.csv'
        )
    params = Table.read(path_params, format='ascii')

    K_coeff_obs[j, :] = params['parfit'][5:]

g_coeff = actuators.fit_grav_deformation(
    K_coeff_alpha=K_coeff_obs,
    alpha=tab_section['meanel']
    )

# for every elevation angle there is a set of K
alpha_list = np.linspace(
    start=tab_section['meanel'].min(),
    stop=tab_section['meanel'].max(),
    num=resolution
    )

K_coeff_model = np.zeros((alpha_list.size, N_K_coeff))
for a, _alpha in enumerate(alpha_list):
    for i in range(N_K_coeff):
        K_coeff_model[a, i] = actuators.grav_deformation(g_coeff[i, :], _alpha)


phases_model = actuators.generate_phase_pr(
    g_coeff=g_coeff,
    alpha=actuators.alpha_lookup
    )

fig_phase = actuators.plot(phases_model)
fig_fem_oof = actuators.plot(actuators.phase_pr_lookup - phases_model)

# list of tuples with (n, l) allowed values
# nl = [(i, j) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
# phase_K_coeff_obs = np.zeros_like(K_coeff_obs) << u.rad
# rho = -0.25
# theta = -35 * u.deg
# for j, _nl in enumerate(nl[3:]):
#     for i, _alpha in enumerate(tab_section['meanel']):
#         phase_K_coeff_obs[i, j] = (
#             K_coeff_obs[i, j] * pyoof.zernike.U(*_nl, rho, theta) * 2 * np.pi
#             ) * u.rad

# removing large amplitude edge effects
sr = 3.25 * u.m
x = np.linspace(-sr, sr, resolution)
xx, yy = np.meshgrid(x, x)

# actuator rings at the active surface
R = np.array([3250, 2600, 1880, 1210]) * u.mm
phase_max = 2 * np.pi * u.rad
nl = [(i, j) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]

for k, _R in enumerate(R):
    if np.abs(phases_model[:, xx ** 2 + yy ** 2 > _R ** 2]).max() > phase_max:
        phases_model[:, xx ** 2 + yy ** 2 > _R ** 2] = 0. * u.rad

fig_fit, axes = plt.subplots(
    nrows=6, ncols=3, sharex=True, sharey=True, figsize=(14, 10)
    )
ax = axes.flatten()
for j in range(N_K_coeff - 3):
    for i, _alpha in enumerate(tab_section['meanel']):

        if tab_section['obs-date'][i] > Time('2020-01-01'):
            _color = 'b'
        else:
            _color = 'k'

        ax[j].scatter(
            _alpha.to_value(u.deg), K_coeff_obs[i, j + 3] * 2 * np.pi,
            # _alpha.to_value(u.deg), phase_K_coeff_obs[i, j].to_value(u.rad),
            marker='o',
            s=tab_section['beam-snr'][i] / 100,
            color=_color,
            )
        ax[j].plot(alpha_list, K_coeff_model[:, j + 3] * 2 * np.pi, 'r')
        # ax[j].set_ylim(-1.5, 1.5)
        ax[j].set_xlim(7, 90)

    patch = Patch(label=f"K({nl[j + 3][0]}, {nl[j + 3][1]})")
    ax[j].legend(handles=[patch], loc='upper right', handlelength=0)

fig_fit.tight_layout()

fig_phase_modified = actuators.plot(phases_model)
fig_normal = actuators.plot()
fig_fem_oof_modified = actuators.plot(actuators.phase_pr_lookup - phases_model)

plt.show()



