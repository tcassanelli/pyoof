#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename
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
    resolution, box_factor, interp
        ):
    """
    Computes the true residual ready to use for the fit_beam function. True
    means that some of the parameters used will not be fitted, these parameter
    are selected by their position as
    params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)]
    and to exclude or include them the config_params.yaml file needs to be
    edited, in the pyoof package directory or by adding a new configuration
    file.

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
        Radial offset added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It has to contain, same as the
        beam_data_norm, the radial offsets for the minus, zero and plus, such
        as d_z = [d_z-, 0., d_z+] all of them in meters.
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with coefficients illum_func(x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (float) in meters.
        telego = [block_dist, opd_func, pr].
    resolution : int
        FFT resolution for a rectangular grid. The input value has to be
        greater or equal to the telescope resolution and a power of 2 for FFT
        faster processing.
    box_factor : int
        Related to the FFT resolution, defines the image pixel size level,
        depending on the data a good value has to be chosen, the standard is
        5, then the box_size = 5 * pr.
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
        minimization scipy.optimize.least_squares package.
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
            resolution=resolution,
            box_factor=box_factor
            )

        power_pattern = np.abs(F) ** 2

        # Normalized power pattern model (Beam model)
        power_norm = power_pattern / power_pattern.max()

        if interp:

            # Generated beam u and v: wave-vectors -> degrees -> radians
            u_rad = wavevector2radians(u, wavel)
            v_rad = wavevector2radians(v, wavel)

            # The calculated beam needs to be transformed!
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

    # Residual = data - model
    residual = beam_data_all - beam_model_all

    return residual


def residual(
    params, idx, N_K_coeff, beam_data_norm, u_data, v_data, d_z, wavel,
    illum_func, telgeo, resolution, box_factor, interp, config_params
        ):
    """
    Wrapper function for the residual_true function. The objective of this
    function is to fool the scipy.optimize.least_squares package by changing
    the number of parameters that will be used. The parameters must be in the
    following order,
    params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)]
    and to exclude or include them the config_params.yaml file needs to be
    edited, in the pyoof package directory or by adding a new configuration
    file.

    Parameters
    ----------
    params : ndarray
        Contains the false array of parameters, the params array will be
        updated here for the correct number of parameters and then be used in
        the residual_ture function. The params array should be of the form,
        params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)].
    idx : list
        List of the positions of the parameters that are desired to left out
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
        Radial offset added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It has to contain, same as the
        beam_data_norm, the radial offsets for the minus, zero and plus, such
        as d_z = [d_z-, 0., d_z+] all of them in meters.
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with coefficients illum_func(x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (float) in meters.
        telego = [block_dist, opd_func, pr].
    resolution : int
        FFT resolution for a rectangular grid. The input value has to be
        greater or equal to the telescope resolution and a power of 2 for FFT
        faster processing.
    box_factor : int
        Related to the FFT resolution, defines the image pixel size level,
        depending on the data a good value has to be chosen, the standard is
        5, then the box_size = 5 * pr.
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

    # Parameters list for the true fit
    params_res = params_complete(params, idx, N_K_coeff, config_params)

    res_true = residual_true(
        params=params_res,
        beam_data_norm=beam_data_norm,
        u_data=u_data,
        v_data=v_data,
        d_z=d_z,
        wavel=wavel,
        resolution=resolution,
        box_factor=box_factor,
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
        updated here for the correct number of parameters to be used in the
        residual_ture function. The params array should be of the form,
        params = [i_amp, c_dB, x0, y0, K(0, 0), ... , K(l, n)].
    idx : list
        List of the positions of the parameters that are desired to left out
        from the optimization. e.g. 0 corresponds to i_amp, 1 to c_dB, and so
        on.
    N_K_coeff : int
        Number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        N_K_coeff = (n + 1) * (n + 2) // 2.
    config_params : dict
        Contains the values for the fixed parameters, commonly the first four
        parameters are fixed. See the config_params.yaml file.

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


def fit_beam(
    data_info, data_obs, method, order_max, illumination, telescope,
    fit_previous, config_params_file, resolution, box_factor, make_plots,
    verbose=2
        ):
    """
    Computes the Zernike circle polynomials coefficients using the least
    squares minimization, stores and plot data (optional) from the analysis.
    These data correspond to the best fitted power pattern (beam maps), and its
    correspondent phase error, as well as information of the optimization.
    Observational data is required. It is the core function of the pyoof
    package, please provide data as stated here or as explained in the
    notebooks examples.

    Parameters
    ----------
    data_info : list
        It contains useful information for the least squares optimization.
        data_info = [name, pthto, obs_object, obs_date, freq, wavel, d_z,
        meanel]. File name, directory which contains fits file, observed
        object, date of observation, frequency Hz, wavelength m, three d_z
        values m and mean elevation degrees. See aux_functions for more
        information.
    data_obs : list
        It contains beam maps and x-, y-axis data for the least squares
        optimization. data_obs = [beam_data, u_data, v_data]. u_data, v_data
        must be in radians.
    method : str
        Least squares minimization algorithm, it can be 'trf', 'lm' or
        'dogbox'. 'lm' does not handle bounds, please refer to documentation
        scipy.optimize.least_squares.
    order_max : int
        Maximum order to be fitted in the least squares minimization, e.g.
        order 3 will calculate order 1, 2 and 3, consecutively.
    illumination : list
        Contains illumination function, and two strings, the name and the
        taper name. illumination = [illum_func, illum_name, taper_name].
    telescope : list
        Contains blockage distribution, OPD function, radius primary
        dish in meters and the telescope name (str).
        telescope = [block_dist, opd_func, pr, tel_name].
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
        FFT2 resolution for a rectangular grid. The input value has to be
        greater or equal to the telescope resolution and a power of 2 for FFT
        faster processing.
    box_factor : int
        Related to the FFT resolution, defines the image pixel size level,
        depending on the data a good value has to be chosen, the standard is
        5, then the box_size = 5 * pr.
    make_plots : bool
        If True will generate a sub-directory with all the important plots for
        the OOF holography, including phase and fitted beam.
    verbose : int
        {0, 1, 2} Level of algorithm verbosity. 0 work silent, 1 display
        termination report, 2, display progress during iteration (default).
    """

    start_time = time.time()

    print('\n ******* PYOOF FIT POWER PATTERN ******* \n')
    print('... Reading data ... \n')

    # All observed data needed to fit the beam
    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    try:

        [illum_func, illum_name, taper_name] = illumination
        telgeo, tel_name = telescope[:3], telescope[3]

        # Calling default configuration file from the pyoof package
        if config_params_file is None:
            config_params_pyoof = get_pkg_data_filename(
                'data/config_params.yml'
                )

            with open(config_params_pyoof, 'r') as yaml_config:
                config_params = yaml.load(yaml_config)
        else:
            with open(config_params_file, 'r') as yaml_config:
                config_params = yaml.load(yaml_config)

        # Generating specific exceptions
        if not (
            callable(illum_func) and isinstance(illum_name, str) and
            isinstance(taper_name, str)
                ):
            raise ValueError('illumination has to be a list [func, str, str]')

        if not (
            callable(telescope[0]) and callable(telescope[1]) and
            isinstance(telescope[2], float) and isinstance(telescope[3], str)
                ):
            raise ValueError(
                'telescope has to be a list [func, func, float, str]'
                )

    except ValueError as error:
        print(error.args)
    except NameError:
        print(
            'Configuration file .yaml does not exist in path: ' +
            config_params_file
            )

    else:
        pass

    # Storing files in pyoof_out directory
    # pthto: path or directory where the fits file is located
    if not os.path.exists(pthto + '/pyoof_out'):
        os.makedirs(pthto + '/pyoof_out')

    for j in ["%03d" % i for i in range(101)]:
        name_dir = pthto + '/pyoof_out/' + name + '-' + str(j)
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)
            break

    print('Maximum order to be fitted: ', order_max)
    print('Telescope name: ', tel_name)
    print('File name: ', name)
    print('Obs frequency: ', freq, 'Hz')
    print('Obs Wavelength : ', np.round(wavel, 4), 'm')
    print('d_z (out-of-focus): ', np.round(d_z, 4), 'm')
    print('Illumination to be fitted: ', illum_name)

    for order in range(1, order_max + 1):

        if not verbose == 0:
            print('\n... Fit order ' + str(order) + ' ... \n')

        # Setting limits for plotting fitted beam
        # input data in radians
        plim_u = [np.min(u_data[0]), np.max(u_data[0])]
        plim_v = [np.min(v_data[0]), np.max(v_data[0])]
        plim_rad = np.array(plim_u + plim_v)

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
                if not verbose == 0:
                    print('Initial params: n={} fit'.format(n - 1))
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

        idx = config_params['params_excluded']  # exclude params from fit
        # [0, 1, 2, 3, 4] = [i_amp, c_dB, x0, y0, K(0, 0)]
        # or 'None' to include all
        params_init_true = np.delete(params_init, idx)

        if method == 'lm':
            bounds = tuple([
                -np.ones(params_init_true.shape) * np.inf,
                np.ones(params_init_true.shape) * np.inf
                ])

        else:
            bounds_min = np.array(
                config_params['params_bounds_min'] + [-5] * (N_K_coeff - 1)
                )
            bounds_max = np.array(
                config_params['params_bounds_max'] + [5] * (N_K_coeff - 1)
                )

            bounds_min_true = np.delete(bounds_min, idx)
            bounds_max_true = np.delete(bounds_max, idx)

            bounds = tuple([bounds_min_true, bounds_max_true])

        if not verbose == 0:
            print('Parameters to fit: {}\n'.format(len(params_init_true)))

        # Running non-linear least-squared optimization
        res_lsq = optimize.least_squares(
            fun=residual,
            x0=params_init_true,
            # Conserve the same order of the arguments as the residual func
            args=(
                idx,  # Index of parameters to be excluded (params)
                N_K_coeff,  # Total number of Zernike circle polynomial coeff
                beam_data_norm,  # Normalized beam maps
                u_data,  # x coordinate beam map
                v_data,  # y coordinate beam map
                d_z,  # Radial offset
                wavel,  # Wavelength of observation
                illum_func,  # Illumination function
                telgeo,  # [block_dist, opd_func, pr]
                resolution,  # FFT2 resolution for a rectangular grid
                box_factor,  # Image pixel size level
                True,  # Grid interpolation
                config_params  # Coeff configuration for minimization (dict)
                ),
            bounds=bounds,
            method=method,
            verbose=verbose,
            max_nfev=None
            )

        # Solutions from least squared optimization
        params_solution = params_complete(
            params=res_lsq.x,
            idx=idx,
            N_K_coeff=N_K_coeff,
            config_params=config_params
            )
        params_init = params_init  # Initial parameters used
        res_optim = res_lsq.fun.reshape(3, -1)  # Optimum residual solution
        jac_optim = res_lsq.jac  # Last evaluation Jacobian matrix
        grad_optim = res_lsq.grad  # Last evaluation Gradient

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
            pr=telgeo[2]
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
        if not verbose == 0:
            print('... Saving data ... \n')

        # To store fit information and found parameters in ascii file
        ascii.write(
            table=params_to_save,
            output=name_dir + '/fitpar_n' + str(n) + '.csv',
            names=['parname', 'parfit', 'parinit'],
            comment='Fitted parameters ' + name
            )

        # Printing the results from saved ascii file
        if not verbose == 0:
            print(ascii.read(name_dir + '/fitpar_n' + str(n) + '.csv'))

        if n == 1:
            pyoof_info = dict(
                telescope=tel_name,
                name=name,
                obs_object=obs_object,
                obs_date=obs_date,
                d_z=d_z,
                wavel=wavel,
                frequency=freq,
                illumination=illum_name,
                meanel=meanel,
                fft_resolution=resolution,
                box_factor=box_factor,
                opt_method=method
                )

            with open(name_dir + '/pyoof_info.yml', 'w') as outfile:
                outfile.write('# pyoof relevant information\n')
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

        if make_plots:
            if not verbose == 0:
                print('\n... Making plots ...')

            plot_fit_path(  # Making all relevant plots
                path_pyoof=name_dir,
                order=n,
                telgeo=telgeo,
                illum_func=illum_func,
                plim_rad=plim_rad,
                save=True,
                angle='degrees',
                resolution=resolution,
                box_factor=box_factor
                )

            plt.close('all')

    final_time = np.round((time.time() - start_time) / 60, 2)
    print('\n **** PYOOF FIT COMPLETED AT {} mins **** \n'.format(final_time))
