#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import constants
from astropy import units as u
from pyoof import aperture
from pyoof.actuator import EffelsbergActuator

"""
The purpose of this script is to create a lookup table given a set of phase
maps from the pyoof software. The method only applies to those observations
that were made with the active surface on, i.e. with corrections from FEM.
Here we isolate the true deformations to apply directly a look-up table to the
active surface.
"""

# we need to call the output from the pyoof
# path_pyoof_out = '/Users/tomascassanelli/MPIfR/OOF/data2019/pyoof_out'
path_pyoof_out = '/scratch/v/vanderli/cassane/pyoof_data/pyoof_out'
path2save = os.path.join(path_pyoof_out, 'grav_deformation')

actuator = EffelsbergActuator(
    wavel=7 * u.mm,
    nrot=3,
    sign=-1,
    order=5,
    sr=3.25 * u.m,
    pr=50 * u.m,
    resolution=1000,
    )

# reading data from the pyoof_out
files = glob.glob(os.path.join(path_pyoof_out, '*'))
alpha_obs = np.zeros((len(files))) << u.deg
phase_pr_obs = np.zeros(
    (len(files), actuator.resolution, actuator.resolution)) << u.rad
for k, _f in enumerate(files):

    with open(os.path.join(_f, 'pyoof_info.yml'), 'r') as inputfile:
        pyoof_info = yaml.load(inputfile, Loader=yaml.Loader)

    alpha_obs[k] = pyoof_info['meanel'] * u.deg
    phase_pr_obs[k, :, :] = np.genfromtxt(
        os.path.join(_f, f'phase_n{actuator.n}.csv')
        ) * u.rad

# Generating the G coeff for the look-up table
try:
    path_G_coeff_lookup = os.path.join(
        path2save, f'G_coeff_lookup_{actuator.wavel.to_value(u.mm)}mm.npy'
        )
    G_coeff_lookup = np.load(path_G_coeff_lookup, allow_pickle=True)
    # G_coeff.shape = (21, 3)

except FileNotFoundError:

    if not os.path.exists(path2save):
        os.makedirs(path2save, exist_ok=True)

    G_coeff_lookup = actuator.fit_all(
        phase_pr=actuator.phase_pr_lookup,
        alpha=actuator.alpha_lookup
        )[0]

    np.save(
        file=path_G_coeff_lookup[:-4],
        arr=G_coeff_lookup
        )

# Generating the phase from the original look-up table
phase_pr_lookup = actuator.generate_phase_pr(
    G_coeff=G_coeff_lookup,
    alpha=alpha_obs
    )

# Generating the G coeff for the observations
G_coeff_obs = actuator.fit_all(
    phase_pr=phase_pr_obs,
    alpha=alpha_obs
    )[0]

phase_pr_obs = actuator.generate_phase_pr(
    G_coeff=G_coeff_obs,
    alpha=alpha_obs
    )

# Corrected phase the observed minus the original look-up table
phase_pr_real = phase_pr_obs - phase_pr_lookup

G_coeff_obs = actuator.fit_all(
    phase_pr=phase_pr_real,
    alpha=alpha_obs
    )[0]

actuator.write_lookup(
    fname=os.path.join(path2save, 'lookup_table_Dec2019.txt'),
    actuator_sr=actuator.itransform(phase_pr=phase_pr_real)
    )
