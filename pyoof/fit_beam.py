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
from .aux_functions import store_data_csv

__all__ = [
    'residual_true', 'residual', 'params_complete', 'fit_beam',
    ]


def residual_true(
    params, beam_data_norm, u_data, v_data, d_z, wavel, illum_func, telgeo,
    resolution, interp
        ):
    """
    Computes the true residual ready to use for the fit_beam function. True
    means that some of the parameters used will not be fitted, these parameter
    are selected by their position as
    params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)]
    and to exclude or include them the config_params.yaml file needs to be
    edited, in the pyoof package directory.

    Parameters
    ----------
    params : ndarray
        Contains the parameters that will be fitted in the least squares
        minimization. The parameters must be in the following sequence
        params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)].
    beam_data_norm : list
        The beam_data_norm is a list with the three observed maps, minus, zero
        and plus out-of-focus. The data has to be initially normalized by its
        maximum.
    u_data : list
        It is the x-axis for the observed beam maps (3 OOF holography
        observations), using the same indexing than the beam_data_norm. The
        dimension must be in radians.
    v_data : list
        It is the y-axis for the observed beam maps (3 OOF holography
        observations), using the same indexing than the beam_data_norm. The
        dimension must be in radians.
    d_z : list
        Distance between the secondary and primary reflector measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end. It has to contain,
        same as the beam_data_norm, the radial offsets for the minus, zero and
        plus, such as d_z = [d_z-, 0., d_z+]
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with parameters (x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage function, optical path difference
        (delta function), and the primary radius (float).
        telego = [blockage, delta, pr].
    resolution : int
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    interp : bool
        If True will process the correspondent interpolation between the
        observed mesh and the computed mesh for the FFT2 aperture distribution
        model.

    Returns
    -------
    residual : ndarray
        One dimensional array for the residual between the observed data and
        the FFT2 aperture distribution model. It has been concatenated as
        minus, zero and plus radial offset. It is required to have the
        residual in one dimension in order to use a least squares
        minimization optimize.least_squares minipack.
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

        if interp:

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
    beam_data_all = np.hstack(
        (beam_data_norm[0], beam_data_norm[1], beam_data_norm[2])
        )

    # Residual = data - model (or fitted)
    residual = beam_data_all - beam_model_all

    return residual


def residual(
    params, idx, N_K_coeff, beam_data_norm, u_data, v_data, d_z, wavel,
    illum_func, telgeo, resolution, interp, config_params
        ):
    """
    Wrapper for the residual_true function. The objective of this function is
    to fool the optimize.least_squares minipack by changing the number of
    parameters that will be used. The parameters must be in the following
    order,
    params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)]
    and to exclude or include them the config_params.yaml file needs to be
    edited, in the pyoof package directory.

    Parameters
    ----------
    params : ndarray
        Contains the false array of parameters, the params array will be
        updated here for the correct number of parameters to then be used in
        the residual_ture function. The params array should be of the form,
        params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)].
    idx : list
        List of the positions of the parameters that are desired to left out \
        from the optimization. e.g. 0 corresponds to i_amp, 1 to c_dB, and so
        on.
    N_K_coeff : int
        Number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        N_K_coeff = (n + 1) * (n + 2) // 2.
    beam_data_norm : list
        The beam_data_norm is a list with the three observed maps, minus, zero
        and plus out-of-focus. The data has to be initially normalized by its
        maximum.
    u_data : list
        It is the x-axis for the observed beam maps (3 OOF holography
        observations), using the same indexing than the beam_data_norm. The
        dimension must be in radians.
    v_data : list
        It is the y-axis for the observed beam maps (3 OOF holography
        observations), using the same indexing than the beam_data_norm. The
        dimension must be in radians.
    d_z : list
        Distance between the secondary and primary reflector measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end. It has to contain,
        same as the beam_data_norm, the radial offsets for the minus, zero and
        plus, such as d_z = [d_z-, 0., d_z+]
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with parameters (x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage function, optical path difference
        (delta function), and the primary radius (float).
        telego = [blockage, delta, pr].
    resolution : int
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    interp : bool
        If True will process the correspondent interpolation between the
        observed mesh and the computed mesh for the FFT2 aperture distribution
        model.
    config_params : dict
        Contains the values for the fixed parameters, commonly the first four
        parameters are fixed. See the config_params file.
    Returns
    -------
    residual : ndarray
        One dimensional array for the residual between the observed data and
        the FFT2 aperture distribution model. See residual_true function.
    """

    # parameters for the true fit
    params_res = params_complete(params, idx, N_K_coeff, config_params)

    res_true = residual_true(
        params=params_res,  # needs to be a numpy array
        beam_data_norm=beam_data_norm,
        u_data=u_data,
        v_data=v_data,
        d_z=d_z,
        wavel=wavel,
        resolution=resolution,
        illum_func=illum_func,
        telgeo=telgeo,
        interp=interp,
        )

    return res_true


def params_complete(params, idx, N_K_coeff, config_params):
    """
    Fills the missing parameters not used in the lease squares optimization,
    They are required to compute the correct aperture distribution. Generally
    the following parameters are excluded [i_amp, taper_dB, x0, y0, K(0, 0)],
    which correspond to the first 4 in the params array. These parameters can
    be excluded or included in the config_params.yaml configuration file,
    located in the pyoof package directory.

    Parameters
    ----------
    params : ndarray
        Contains the false array of parameters, the params array will be
        updated here for the correct number of parameters to then be used in
        the residual_ture function. The params array should be of the form,
        params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)].
    idx : list
        List of the positions of the parameters that are desired to left out \
        from the optimization. e.g. 0 corresponds to i_amp, 1 to c_dB, and so
        on.
    N_K_coeff : int
        Number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        N_K_coeff = (n + 1) * (n + 2) // 2.
    config_params : dict
        Contains the values for the fixed parameters, commonly the first four
        parameters are fixed. See the config_params file.

    Returns
    -------
    _params : ndarray
        Complete set of parameters to calculate the FFT2 aperture distribution.
    """

    # Fixed values for parameters, in case they're excluded, see idx
    [i_amp_f, taper_dB_f, x0_f, y0_f, K_f] = config_params['params_fixed']

    # N_K_coeff number of Zernike circle polynomials coefficients
    if params.size != (4 + N_K_coeff):
        _params = params.copy()
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
    data_info, data_obs, order_max, illumination, telescope, fit_previous,
    config_params_file, resolution, make_plots
        ):
    """
    Computes the Zernike circle polynomials coefficients using the least
    squares minimization, stores and plot data from the analysis. These data
    correspond to the best fitted power pattern, and its correspondent phase
    error, as well as information of the optimization.
    Observational data is required. The most important function in the pyoof
    package, please provide data as stated here or as explained in the
    notebooks examples.

    Parameters
    ----------
    data_info : list
        It contains useful information for the least squares optimization.
        data_info = [name, pthto, freq, wavel, d_z, meanel].
    data_obs : list
        It contains beam maps and x-, y-axis data for the least squares
        optimization.
        data_obs = [beam_data, u_data, v_data].
    order_max : int
        Maximum order to be fitted in the least squares minimization, e.g.
        order 3 will calculate order 1, 2 and 3.
    illumination : list
        Contains illumination function, and two strings, the name
        and the taper name.
        illumination = [illum_func, illum_name, taper_name].
    telescope : list
        Contains blockage function, delta function, radius primary dish and
        the telescope name (str).
        telescope = [blockage, delta, pr, tel_name].
    fit_previous : bool
        If set to True will fit the coefficients from the previous optimization
        this feature is strongly suggested.
    config_params_file : str
        Path for the configuration file, this includes, the maximum and
        minimum bounds, excluded, fixed and initial parameters for the
        optimization. See config_params.yaml in the pyoof package directory.
    config_params_path : str
        Path for the configuration file, required for the excluded or included
        parameters and their bounds for the optimization. If None the default
        setting will be used.
    resolution : int
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    make_plots : bool
        If True will generate a sub-directory with all the important plots for
        the OOF holography, including phase and beam fit.
    """

    start_time = time.time()

    print('\n ******* OOF FIT POWER PATTERN ******* \n')
    print('... Reading data ... \n')

    # All observed data needed to fit the beam
    [name, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    [illum_func, illum_name, taper_name] = illumination
    [blockage, delta, pr, tel_name] = telescope
    telgeo = telescope[:3]

    # Calling default configuration file from the pyoof package
    if config_params_file is None:
        config_params_dir = os.path.dirname(__file__)
        config_params_pyoof = os.path.join(
            config_params_dir, 'config_params.yaml'
            )
        with open(config_params_pyoof, 'r') as yaml_config:
            config_params = yaml.load(yaml_config)
    else:
        with open(config_params_file, 'r') as yaml_config:
            config_params = yaml.load(yaml_config)

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
    print('Obs frequency: ', freq, 'Hz')
    print('Obs Wavelength : ', wavel, 'm')
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

        # Beam normalization
        beam_data_norm = [beam_data[i] / beam_data[i].max() for i in range(3)]

        n = order  # order polynomial to fit
        N_K_coeff = (n + 1) * (n + 2) // 2  # number of K(n, l) to fit

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
            # i_amp, sigma_r, x0, y0, K(n, l)
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
                illum_func,
                telgeo,
                resolution,
                True,  # Grid interpolation
                config_params
                ),
            bounds=tuple([bounds_min_true, bounds_max_true]),
            method='trf',
            verbose=2,
            max_nfev=None
            )

        print('\n')

        # Solutions from least squared optimization
        params_solution = params_complete(
            params=res_lsq.x,
            idx=idx,
            N_K_coeff=N_K_coeff,
            config_params=config_params
            )
        params_init = params_init
        res_optim = res_lsq.fun.reshape(3, -1)  # Optimum residual from fitting
        jac_optim = res_lsq.jac
        grad_optim = res_lsq.grad

        cov, corr = co_matrices(
            res=res_lsq.fun,
            jac=res_lsq.jac,
            n_pars=params_init_true.size  # number of parameters fitted
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
            params_names.append('K({}, {})'.format(N[i], L[i]))

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
            pyoof_info = dict(
                telescope=tel_name,
                name=name,
                d_z=d_z,
                wavel=wavel,
                frequency=freq,
                illumination=illum_name,
                meanel=meanel,
                fft_resolution=resolution
                )

            with open(name_dir + '/pyoof_info.yaml', 'w') as outfile:
                yaml.dump(pyoof_info, outfile, default_flow_style=False)

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
                path_pyoof=name_dir,
                order=n,
                telgeo=telgeo,
                illum_func=illum_func,
                plim_rad=plim_rad,
                save=True,
                angle='degrees',
                resolution=resolution
                )

            plt.close('all')

    final_time = np.round((time.time() - start_time) / 60, 2)
    print('\n **** OOF FIT COMPLETED AT {} mins **** \n'.format(final_time))
