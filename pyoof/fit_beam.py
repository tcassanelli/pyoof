#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import interpolate, optimize
import os
import time
from .aperture import angular_spectrum, phase, illum_choice
from .math_functions import wavevector2radians, co_matrices
from .plot_routines import plot_fit_path
from .aux_functions import store_csv, store_ascii

__all__ = [
    'residual_true', 'residual', 'params_complete', 'fit_beam',
    ]


def residual_true(
    params, beam_data, u_data, v_data, d_z, lam, illum, telescope, inter,
    resolution
        ):

    I_coeff = params[:4]
    K_coeff = params[4:]

    beam_model = []
    for i in range(3):

        u, v, aspectrum = angular_spectrum(
            K_coeff=K_coeff,
            d_z=d_z[i],
            I_coeff=I_coeff,
            illum=illum,
            telescope=telescope,
            resolution=resolution
            )

        beam = np.abs(aspectrum) ** 2
        beam_norm = beam / beam.max()

        if inter:

            # Generated beam u and v: wavevectors -> degrees -> radians
            u_rad = wavevector2radians(u, lam)
            v_rad = wavevector2radians(v, lam)

            # The calculated beam needs to be transformed!
            # RegularGridInterpolator
            intrp = interpolate.RegularGridInterpolator(
                points=(u_rad, v_rad),  # points defining grid
                values=beam_norm.T,  # data in grid
                method='linear'  # linear or nearest
                )

            # input interpolation function is the real beam grid
            beam_model.append(intrp(np.array([u_data[i], v_data[i]]).T))

        else:
            beam_model.append(beam_norm)

    beam_model_all = np.hstack((beam_model[0], beam_model[1], beam_model[2]))
    beam_data_all = np.hstack((beam_data[0], beam_data[1], beam_data[2]))

    # Residual = data - model (or fitted)
    residual = beam_data_all - beam_model_all

    return residual


def residual(
    params, idx, N_K_coeff, beam_data, u_data, v_data, d_z, lam, resolution,
    illum, telescope, inter
        ):

    # params for the true fit
    params_res = params_complete(params, idx, N_K_coeff)

    res_true = residual_true(
        params=params_res,  # needs to be a numpy array
        beam_data=beam_data,
        u_data=u_data,
        v_data=v_data,
        d_z=d_z,
        lam=lam,
        resolution=resolution,
        illum=illum,
        telescope=telescope,
        inter=inter,
        )

    return res_true


def params_complete(params, idx, N_K_coeff):
    # N_K_coeff number of Zernike coeff
    if params.size != (4 + N_K_coeff):
        _params = params
        for i in idx:
            if i == 0:
                _params = np.insert(_params, i, 1.0)
                # assigned default value for amp
            elif i == 1:
                _params = np.insert(_params, i, -8.0)
                # assigned default value for c_dB
            else:
                _params = np.insert(_params, i, 0.0)
                # for x0, y0 and K(l, n) coefficients
    else:
        _params = params

    return _params


# Insert path for the fits file with pre-calibration
def fit_beam(data, order, illum, telescope, fit_previous, resolution, angle):

    start_time = time.time()

    print('\n ####### OOF FIT POWER PATTERN ####### \n')
    print('... Reading data ... \n')

    # All observed data needed to fit the beam
    data_info, data_obs = data
    [name, pthto, freq, wavel, d_z_m, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    print('File name: ', name)
    print('Observed frequency: ', freq, 'Hz')
    print('Wavelenght : ', wavel, 'm')
    print('d_z (out-of-focus): ', d_z_m, 'm')
    print('Order n to be fitted: ', order)
    print('Illumination to be fitted: ', illum)

    # Setting limits for plotting fitted beam
    plim_u = [np.min(u_data[0]), np.max(u_data[0])]  # input data in radians
    plim_v = [np.min(v_data[0]), np.max(v_data[0])]
    plim_rad = np.array(plim_u + plim_v)

    # d_z is given in units of wavelength (m/m)
    d_z = np.array(d_z_m) * 2 * np.pi / wavel  # convert to radians

    # Beam normalisation
    beam_data_norm = [beam_data[i] / beam_data[i].max() for i in range(3)]

    n = order  # order polynomial to fit
    N_K_coeff = (n + 1) * (n + 2) // 2  # number of Zernike coeff to fit

    # Storing files in OOF_out directory
    name_dir = pthto + '/OOF_out/' + name
    # pthto: path or directory where the fits file is located

    if not os.path.exists(pthto + '/OOF_out'):
        os.makedirs(pthto + '/OOF_out')
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

    # Looking for result parameters lower order
    if fit_previous and n != 1:
        N_K_coeff_previous = n * (n + 1) // 2
        path_params_previous = name_dir + '/fitpar_n' + str(n - 1) + '.dat'
        params_to_add = N_K_coeff - N_K_coeff_previous

        if os.path.exists(path_params_previous):
            params_init = np.hstack(
                (ascii.read(path_params_previous)['parfit'],
                    np.ones(params_to_add) * 0.1)
                )
            print('Using initial params from n=' + str(n - 1) + ' fit')
        else:
            print(
                '\n ERROR: There is no previous parameters file fitpar_n' +
                str(n - 1) + '.dat in directory \n'
                )
    else:
        params_init = np.array([0.1, -15, 0, 0, 0] + [0.1] * (N_K_coeff - 1))
        print('Using standard initial params')
        # amp, sigma_r, x0, y0, K(l,m)
        # Giving an initial value of 0.1 for each coeff

    bounds_min = np.array([0, -25, -1e-2, -1e-2] + [-5] * N_K_coeff)
    bounds_max = np.array([np.inf, -8, 1e-2, 1e-2] + [5] * N_K_coeff)

    idx = [0, 1, 2, 3, 4]  # exclude params from fit
    # [0, 1, 2, 3, 4] = [amp, c_dB, x0, y0, K(0, 0)] or 'None' to include all

    params_init_true = np.delete(params_init, idx)
    bounds_min_true = np.delete(bounds_min, idx)
    bounds_max_true = np.delete(bounds_max, idx)

    print(
        '\n... Starting fit for ' + str(len(params_init_true)) +
        ' parameters ... \n'
        )

    # Running non-linear least-squared optimization
    res_lsq = optimize.least_squares(
        fun=residual,
        x0=params_init_true,
        # conserve the same order of the arguments as the residual func
        args=(
            idx,
            N_K_coeff,
            beam_data_norm,
            u_data,
            v_data,
            d_z,
            wavel,
            resolution,
            illum,
            telescope,
            True  # Grid interpolation
            ),
        bounds=tuple([bounds_min_true, bounds_max_true]),
        method='trf',
        verbose=2,
        # max_nfev=100
        )

    print('\n')

    # Solutions from least squared optimisation
    params_solution = params_complete(res_lsq.x, idx, N_K_coeff)
    params_init = params_init
    res_optim = res_lsq.fun.reshape(3, -1)  # Optimum residual from fitting
    jac_optim = res_lsq.jac
    grad_optim = res_lsq.grad

    cov, corr = co_matrices(
        res=res_lsq.fun,
        jac=res_lsq.jac,
        n_pars=params_init_true.size  # num of parameters fitted
        )
    cov_ptrue = np.vstack((np.delete(np.arange(N_K_coeff + 4), idx), cov))
    corr_ptrue = np.vstack((np.delete(np.arange(N_K_coeff + 4), idx), corr))

    # Final phase from fit in the telescope's primary reflector
    _phase = phase(
        K_coeff=params_solution[4:],
        notilt=True,
        telescope=telescope
        )

    # Making nice table :)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    # string with the illumination taper name
    illum_taper = illum_choice(illum)[1]

    params_names = ['illum_amp', illum_taper, 'x_0', 'y_0']
    for i in range(N_K_coeff):
        params_names.append('K(' + str(L[i]) + ',' + str(N[i]) + ')')

    params_to_save = [params_names, params_solution, params_init]
    info_to_save = [
        [name], [d_z_m[0]], [d_z_m[1]], [d_z_m[2]], [wavel],
        [freq], [illum], [meanel], [resolution]
        ]

    # Storing files in directory
    print('... Saving data ... \n')

    # To store fit information and found parameters in ascii file
    store_ascii(name, n, name_dir, params_to_save, info_to_save)

    # To store large files in csv format
    save_to_csv = [
        beam_data, u_data, v_data, res_optim, jac_optim, grad_optim, _phase,
        cov_ptrue, corr_ptrue
        ]
    store_csv(name, n, name_dir, save_to_csv)

    # Printing the results from saved ascii file
    print(ascii.read(name_dir + '/fitpar_n' + str(n) + '.dat'))
    print('\n')

    # Making all relevant plots
    print('... Making plots ... \n')

    plot_fit_path(
        pathoof=name_dir,
        order=n,
        telescope=telescope,
        plim_rad=plim_rad,
        save=True,
        angle=angle,
        resolution=resolution
        )

    plt.close('all')

    print(' ###### %s mins ######' % str((time.time() - start_time) / 60))
    print('\n')
