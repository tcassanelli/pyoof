#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import interpolate, optimize
import os
import time
import yaml
from .aperture import radiation_pattern, phase
from .math_functions import wavevector2radians, co_matrices
from .plot_routines import plot_fit_path
from .aux_functions import store_data_csv, store_info_csv

__all__ = [
    'residual_true', 'residual', 'params_complete', 'fit_beam',
    ]

# Calling configuration file
config_params_dir = os.path.dirname(__file__)
config_params_pth = os.path.join(config_params_dir, 'config_params.yaml')
with open(config_params_pth, 'r') as stream:
    config_params = yaml.load(stream)


def residual_true(
    params, beam_data, u_data, v_data, d_z, wavel, illum_func, telgeo,
    inter, resolution
        ):
    """
    Computes the true residual ready to use for the fit_beam function.

    Parameters
    ----------
    params : ndarray
        Contains the parameters that will be fitted in the least squares
        minimization. The parameters must be in the following sequence
        params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)].
    beam_data : list
        The beam_data is a list with the three observed maps, minus, zero and
        plus out-of-focus.
    """

    I_coeff = params[:4]
    K_coeff = params[4:]

    beam_model = []
    for i in range(3):

        u, v, F = radiation_pattern(
            K_coeff=K_coeff,
            d_z=d_z[i],
            wavel=wavel,
            I_coeff=I_coeff,
            illum_func=illum_func,
            telgeo=telgeo,
            resolution=resolution
            )

        power_pattern = np.abs(F) ** 2
        power_norm = power_pattern / power_pattern.max()

        if inter:

            # Generated beam u and v: wavevectors -> degrees -> radians
            u_rad = wavevector2radians(u, wavel)
            v_rad = wavevector2radians(v, wavel)

            # The calculated beam needs to be transformed!
            # RegularGridInterpolator
            intrp = interpolate.RegularGridInterpolator(
                points=(u_rad, v_rad),  # points defining grid
                values=power_norm.T,  # data in grid
                method='linear'  # linear or nearest
                )

            # input interpolation function is the real beam grid
            beam_model.append(intrp(np.array([u_data[i], v_data[i]]).T))

        else:
            beam_model.append(power_norm)

    beam_model_all = np.hstack((beam_model[0], beam_model[1], beam_model[2]))
    beam_data_all = np.hstack((beam_data[0], beam_data[1], beam_data[2]))

    # Residual = data - model (or fitted)
    residual = beam_data_all - beam_model_all

    return residual


def residual(
    params, idx, N_K_coeff, beam_data, u_data, v_data, d_z, wavel, resolution,
    illum_func, telgeo, inter
        ):

    # params for the true fit
    params_res = params_complete(params, idx, N_K_coeff)

    res_true = residual_true(
        params=params_res,  # needs to be a numpy array
        beam_data=beam_data,
        u_data=u_data,
        v_data=v_data,
        d_z=d_z,
        wavel=wavel,
        resolution=resolution,
        illum_func=illum_func,
        telgeo=telgeo,
        inter=inter,
        )

    return res_true


def params_complete(params, idx, N_K_coeff):
    # Fixed values for parameters, in case they're excluded, see idx
    [i_amp_f, taper_dB_f, x0_f, y0_f, K_f] = config_params['params_fixed']

    # N_K_coeff number of Zernike coeff
    if params.size != (4 + N_K_coeff):
        _params = params
        for i in idx:
            if i == 0:
                _params = np.insert(_params, i, i_amp_f)
                # assigned value for i_amp
            elif i == 1:
                _params = np.insert(_params, i, taper_dB_f)
                # assigned value for c_dB
            elif i == 2:
                _params = np.insert(_params, i, x0_f)
                # assigned value for x0
            elif i == 3:
                _params = np.insert(_params, i, y0_f)
                # assigned value for y0
            else:
                _params = np.insert(_params, i, K_f)
                # assigned value for any other
    else:
        _params = params

    return _params


# Insert path for the fits file with pre-calibration
def fit_beam(
    data, order_max, illumination, telescope, fit_previous, resolution, angle,
    make_plots
        ):
    """
    Computes the Zernike circle polynomials coefficients using the least
    squares minimization, stores and plot data from the analysis.
    Obsevational data is required. The most important function in the pyoof
    package, please provide data as stated here or in the repository examples.

    Parameters
    ----------
    data : list
        Input data necessary to perform the lest squares minimization.
        data = [
            [beam_data, u_data, v_data],
            [name, pthto, freq, wavel, d_z, meanel]
            ]
        The beam_data is a list with the three observed maps, minus, zero and
        plus out-of-focus. u_data and v_data are the x- y-axis in radians for
        the three maps minus, zero and plus out-of-focus both are lists.
        name is a string characterising the observation, pthto the path to the
        file. freq the frequency in Hz, wavel the wavelength in m. d_z is a
        list which contains the radial offsets from minus, zero and plus
        defocus. meanel is the mean elevation for the three beam maps, not
        necessary for the OOF holography but importat to keep track of.
    order_max : int
        Maximum order to be fitted in the least squares minimization, e.g.
        order 3 will calculate order 1, 2 and 3.
    illumination : list
        List which contains illumination function, and two strings.
        illumination = [illum_func, illum_name, taper_name].
    telescope : list
        List which contains blockage function, delta function, radius primary
        dish and the telescope name (string).
        telescope = [blockage, delta, pr, tel_name].
    fit_previous : bool
        If set to True will fit the coefficients from the previous minimization
        this feature is strongly suggested.
    resolution : int
        Fast Fourier Transform resolution for a rectancular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    angle : str
        Angle unit, it can be 'degrees' or 'radians'.
    """

    start_time = time.time()

    print('\n ******* OOF FIT POWER PATTERN ******* \n')
    print('... Reading data ... \n')

    # All observed data needed to fit the beam
    data_info, data_obs = data
    [name, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    [illum_func, illum_name, taper_name] = illumination
    [blockage, delta, pr, tel_name] = telescope
    telgeo = telescope[:3]

    # Storing files in OOF_out directory
    # pthto: path or directory where the fits file is located
    if not os.path.exists(pthto + '/OOF_out'):
        os.makedirs(pthto + '/OOF_out')

    for j in ["%03d" % i for i in range(101)]:
        name_dir = pthto + '/OOF_out/' + name + '-' + str(j)
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)
            break

    print('Maximum order to be fitted: ', order_max)
    print('Telescope name: ', tel_name)
    print('File name: ', name)
    print('Observed frequency: ', freq, 'Hz')
    print('Wavelenght : ', wavel, 'm')
    print('d_z (out-of-focus): ', d_z, 'm')
    print('Illumination to be fitted: ', illum_name)

    for order in range(1, order_max + 1):

        print('\n... Fit order ' + str(order) + ' ... \n')

        # Setting limits for plotting fitted beam
        # input data in radians
        plim_u = [np.min(u_data[0]), np.max(u_data[0])]
        plim_v = [np.min(v_data[0]), np.max(v_data[0])]
        plim_rad = np.array(plim_u + plim_v)

        # d_z is given in units of wavelength (m/m)
        # d_z = np.array(d_z)  # needs to be a numpy array

        # Beam normalisation
        beam_data_norm = [beam_data[i] / beam_data[i].max() for i in range(3)]

        n = order  # order polynomial to fit
        N_K_coeff = (n + 1) * (n + 2) // 2  # number of Zernike coeff to fit

        # Looking for result parameters lower order
        if fit_previous and n != 1:
            N_K_coeff_previous = n * (n + 1) // 2
            path_params_previous = name_dir + '/fitpar_n' + str(n - 1) + '.csv'
            params_to_add = N_K_coeff - N_K_coeff_previous

            if os.path.exists(path_params_previous):
                params_init = np.hstack(
                    (ascii.read(path_params_previous)['parfit'],
                        np.ones(params_to_add) * 0.1)
                    )
                print('Initial params: n=' + str(n - 1) + ' fit')
            else:
                print(
                    '\n ERROR: There is no previous parameters file fitpar_n' +
                    str(n - 1) + '.csv in directory \n'
                    )
        else:
            params_init = np.array(
                config_params['params_init'] + [0.1] * (N_K_coeff - 1)
                )
            print('Initial params: default')
            # i_amp, sigma_r, x0, y0, K(l,m)
            # Giving an initial value of 0.1 for each coeff

        bounds_min = np.array(
            config_params['params_bounds_min'] + [-5] * (N_K_coeff - 1)
            )
        bounds_max = np.array(
            config_params['params_bounds_max'] + [5] * (N_K_coeff - 1)
            )

        idx = config_params['params_excluded']  # exclude params from fit
        # [0, 1, 2, 3, 4] = [i_amp, c_dB, x0, y0, K(0, 0)]
        # or 'None' to include all

        params_init_true = np.delete(params_init, idx)
        bounds_min_true = np.delete(bounds_min, idx)
        bounds_max_true = np.delete(bounds_max, idx)

        print('Parameters to fit: ' + str(len(params_init_true)) + '\n')

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
                illum_func,
                telgeo,
                True  # Grid interpolation
                ),
            bounds=tuple([bounds_min_true, bounds_max_true]),
            method='trf',
            verbose=2,
            max_nfev=None
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
        corr_ptrue = np.vstack(
            (np.delete(np.arange(N_K_coeff + 4), idx), corr)
            )

        # Final phase from fit in the telescope's primary reflector
        _phase = phase(
            K_coeff=params_solution[4:],
            notilt=True,
            pr=pr
            )[2]

        # Making nice table :)
        ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
        L = np.array(ln)[:, 0]
        N = np.array(ln)[:, 1]

        params_names = ['i_amp', taper_name, 'x_0', 'y_0']
        for i in range(N_K_coeff):
            params_names.append('K(' + str(L[i]) + ',' + str(N[i]) + ')')

        params_to_save = [params_names, params_solution, params_init]

        # Storing files in directory
        print('... Saving data ... \n')

        # To store fit information and found parameters in ascii file
        ascii.write(
            table=params_to_save,
            output=name_dir + '/fitpar_n' + str(n) + '.csv',
            names=['parname', 'parfit', 'parinit'],
            comment='Fitted parameters ' + name
            )

        if n == 1:
            # Experiment
            info_dict = {
                'telescope': tel_name,
                'name': name,
                'd_z-': d_z[0],
                'd_z0': d_z[1],
                'd_z+': d_z[2],
                'wavel': wavel,
                'freq': freq,
                'illumination': illum_name,
                'meanel': meanel,
                'fft_resolution': resolution
                }

            store_info_csv(info_dict=info_dict, name_dir=name_dir)

        # To store large files in csv format
        save_to_csv = [
            beam_data, u_data, v_data, res_optim, jac_optim, grad_optim,
            _phase, cov_ptrue, corr_ptrue
            ]

        store_data_csv(
            name=name,
            order=n,
            name_dir=name_dir,
            save_to_csv=save_to_csv
            )

        # Printing the results from saved ascii file
        print(ascii.read(name_dir + '/fitpar_n' + str(n) + '.csv'))

        if make_plots:
            # Making all relevant plots
            print('\n... Making plots ...')

            plot_fit_path(
                pathoof=name_dir,
                order=n,
                telgeo=telgeo,
                illum_func=illum_func,
                plim_rad=plim_rad,
                save=True,
                angle=angle,
                resolution=resolution
                )

            plt.close('all')

    final_time = np.round((time.time() - start_time) / 60, 2)
    print('\n **** OOF FIT COMPLETED AT ' + str(final_time) + ' mins **** \n')
