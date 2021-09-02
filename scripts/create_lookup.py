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
path2save = 'grav_deformation'
path_pyoof_out = '/Users/tomascassanelli/MPIfR/OOF/data/pyoof_out'
n = 5
resolution = 1000
pr = 50 * u.m

actuators = EffelsbergActuator(
    frequency=34.75 * u.GHz,
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

try:
    # AttributeError: 'Time' object has no attribute 'view'
    tab.sort('obs-date')
except AttributeError:
    tab[tab['obs-date'].argsort()]

tab.add_index('name')
# tab.pprint_all()

idx_offset = np.zeros((len(tab), ), dtype=bool)
for i in range(len(tab)):
    if tab['name'][i].split('-')[-1] == 'offset':
        idx_offset[i] = True

section = ((tab['beam-snr-in'] > 200.) & ~idx_offset)
tab_section = tab[section].copy()
tab_section.add_index('name')
tab_section.pprint_all()

print("\ntotal number of OOF holography observations:", len(tab_section))

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

# gravitational model for the Knl coefficients
K_coeff_model = np.zeros((alpha_list.size, N_K_coeff))
for a, _alpha in enumerate(alpha_list):
    for i in range(N_K_coeff):
        K_coeff_model[a, i] = actuators.grav_deformation(g_coeff[i, :], _alpha)

phases_pr_model = actuators.generate_phase_pr(
    g_coeff=g_coeff,
    alpha=actuators.alpha_lookup,
    eac=True
    )

fig_phase_pr_model = actuators.plot(
    data_r=phases_pr_model,
    title="Phase-error model"
    )

fig_phase_pr_lookup = actuators.plot(title="Phase-error look-up table")
nl = [(i, j) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
fig_gravfit, axes = plt.subplots(
    nrows=6, ncols=3, sharex=True, sharey=True, figsize=(14, 10)
    )
ax = axes.flatten()
for j in range(N_K_coeff - 3):
    for i, _alpha in enumerate(tab_section['meanel']):

        # if tab_section['obs-date'][i] > Time('2020-01-01'):
        #     _color = 'b'
        # else:
        #     _color = 'k'

        ax[j].scatter(
            _alpha.to_value(u.deg), K_coeff_obs[i, j + 3] * 2 * np.pi,
            # _alpha.to_value(u.deg), phase_K_coeff_obs[i, j].to_value(u.rad),
            marker='o',
            s=tab_section['beam-snr-in'][i] / 100,
            color='k',
            )
        ax[j].plot(alpha_list, K_coeff_model[:, j + 3] * 2 * np.pi, 'r')
        # ax[j].set_ylim(-1.5, 1.5)
        ax[j].set_xlim(7, 90)

        if j % 3 == 0:
            ax[j].set_ylabel("Phase rad")

        if j > 14:
            ax[j].set_xlabel("Elevation angle ($\\alpha$) deg")

    patch = Patch(label=f"K({nl[j + 3][0]}, {nl[j + 3][1]})")
    ax[j].legend(handles=[patch], loc='upper right', handlelength=0)
fig_gravfit.tight_layout()

if not os.path.isdir(path2save):
    os.makedirs(path2save, exist_ok=True)

# creating lookup table
fname_fem_oof_table = os.path.join(
        path2save, f"FEM_OOF_Table_{Time.now().strftime('%y%m%d')}.dat"
        )

alpha_lookup, atuator_sr_lookup = actuators.read_lookup(False)
actuator_sr_model = actuators.interp_surface2rings(
    actuator_sr=actuators.itransform(phases_pr_model)
    )

actuators.write_lookup(
    fname=fname_fem_oof_table,
    actuator_sr=atuator_sr_lookup - actuator_sr_model
    )

actuators_fem_oof = EffelsbergActuator(
    frequency=34.75 * u.GHz,
    nrot=1,
    sign=-1,
    order=n,
    sr=3.25 * u.m,
    pr=pr,
    resolution=resolution,
    path_lookup=fname_fem_oof_table
    )
fig_phases_pr_correction = actuators_fem_oof.plot(
    title="Phase-error look-up table minus phase-error model (FEM + OOF corrections)"
    )

# plt.show()
fig_phase_pr_model.savefig(os.path.join(
    path2save, f"phase_pr_model_{Time.now().strftime('%y%m%d')}.pdf"
    ))
fig_phase_pr_lookup.savefig(os.path.join(path2save, f"phase_pr_lookup.pdf"))
fig_phases_pr_correction.savefig(os.path.join(
    path2save, f"phases_pr_correction_{Time.now().strftime('%y%m%d')}.pdf"
    ))
fig_gravfit.savefig(os.path.join(
    path2save, f"gravfit_{Time.now().strftime('%y%m%d')}.pdf"
    ))

