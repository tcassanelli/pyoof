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
from .aux_functions import store_data_csv, illum_strings, store_data_ascii

__all__ = [
    'residual_true', 'residual', 'params_complete', 'fit_beam',
    ]


def residual_true(
    params, beam_data_norm, u_data, v_data, d_z, wavel, illum_func, telgeo,
    resolution, box_factor, interp
        ):
    """
    Computes the true residual ready to use for the `~pyoof.fit_beam`
    function. True means that some of the parameters used will not be fitted.
    Their selection is done by default or by adding a ``config_params.yml``
    file to the `~pyoof.fit_beam` function.

    Parameters
    ----------
    params : `~numpy.ndarray`
        Two stacked arrays, the illumination and Zernike circle polynomials
        coefficients. ``params = np.hstack([I_coeff, K_coeff])``.
    beam_data_norm : `list`
        The ``beam_data_norm`` is a list with the three observed beam maps,
        :math:`P^\\mathrm{obs}_\\mathrm{norm}(u, v)`, minus, zero and plus
        out-of-focus. The data has to be initially normalized by its maximum.
    u_data : `~numpy.ndarray`
        :math:`x` axis value for the 3 beam maps in radians. The values have
        to be flatten, in one dimension, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    v_data : `~numpy.ndarray`
        :math:`y` axis value for the 3 beam maps in radians. The values have
        to be flatten, one dimensional, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    d_z : `list`
        Radial offset :math:`d_z`, added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. The radial offset list must be as follows,
        ``d_z = [d_z-, 0., d_z+]`` all of them in meters.
    wavel : `float`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telgeo : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (`float`) in meters. The list
        must have the following order, ``telego = [block_dist, opd_func, pr]``.
    resolution : `int`
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and with
        power of 2 for faster FFT processing. It is recommended a value higher
        than ``resolution = 2 ** 8``.
    box_factor : `int`
        Related to the FFT resolution (**resolution** key), defines the image
        pixel size level. It depends on the primary radius, ``pr``, of the
        telescope, e.g. a ``box_factor = 5`` returns ``x = np.linspace(-5 *
        pr, 5 * pr, resolution)``, an array to be used in the FFT2
        (`~numpy.fft.fft2`).
    interp : `bool`
        If `True`, it will process the correspondent interpolation between
        the observed grid (:math:`P^\\mathrm{obs}_\\mathrm{norm}(u, v)`) and
        the computed grid (:math:`P_\\mathrm{norm}(u, v)`) for the FFT2
        aperture distribution model (:math:`\\underline{E_\\mathrm{a}}(x,
        y)`).

    Returns
    -------
    _residual_true : `~numpy.ndarray`
        One dimensional array of the residual between the observed data and
        the FFT aperture distribution model. It has been concatenated as
        minus, zero and plus radial offset (to do a multiple fit). It is
        required to have the residual in one dimension in order to use a least
        squares minimization `~scipy.optimize.least_squares` package.
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
                values=power_norm.T,    # data in grid
                method='linear'         # linear or nearest
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
    _residual_true = beam_data_all - beam_model_all

    return _residual_true


def residual(
    params, idx, N_K_coeff, beam_data_norm, u_data, v_data, d_z, wavel,
    illum_func, telgeo, resolution, box_factor, interp, config_params
        ):
    """
    Wrapper for the `~pyoof.residual_true` function. The objective of
    this function is to fool the `~scipy.optimize.least_squares` package by
    changing the number of parameters that will be used in the fit. The
    parameter array must be organized as follows, ``params = np.hstack([
    I_coeff, K_coeff])``. The parameter selection is done by default or by
    adding a ``config_params.yml`` file to the `~pyoof.fit_beam` function.

    Parameters
    ----------
    params : `~numpy.ndarray`
        Two stacked arrays, the illumination and Zernike circle polynomials
        coefficients. ``params = np.hstack([I_coeff, K_coeff])``.
    idx : `list`
        List of the positions for the removed parameters for the least squares minimization in the ``params`` array.
        on.
    N_K_coeff : `int`
        Total number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        ``N_K_coeff = (n + 1) * (n + 2) // 2.``.
    beam_data_norm : `list`
        The ``beam_data_norm`` is a list with the three observed beam maps,
        :math:`P^\\mathrm{obs}_\\mathrm{norm}(u, v)`, minus, zero and plus
        out-of-focus. The data has to be initially normalized by its maximum.
    u_data : `~numpy.ndarray`
        :math:`x` axis value for the 3 beam maps in radians. The values have
        to be flatten, in one dimension, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    v_data : `~numpy.ndarray`
        :math:`y` axis value for the 3 beam maps in radians. The values have
        to be flatten, one dimensional, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    d_z : `list`
        Radial offset :math:`d_z`, added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. The radial offset list must be as follows,
        ``d_z = [d_z-, 0., d_z+]`` all of them in meters.
    wavel : `float`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telgeo : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (`float`) in meters. The list
        must have the following order, ``telego = [block_dist, opd_func, pr]``.
    resolution : `int`
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and with
        power of 2 for faster FFT processing. It is recommended a value higher
        than ``resolution = 2 ** 8``.
    box_factor : `int`
        Related to the FFT resolution (**resolution** key), defines the image
        pixel size level. It depends on the primary radius, ``pr``, of the
        telescope, e.g. a ``box_factor = 5`` returns ``x = np.linspace(-5 *
        pr, 5 * pr, resolution)``, an array to be used in the FFT2
        (`~numpy.fft.fft2`).
    interp : `bool`
        If `True`, it will process the correspondent interpolation between
        the observed grid (:math:`P^\\mathrm{obs}_\\mathrm{norm}(u, v)`) and
        the computed grid (:math:`P_\\mathrm{norm}(u, v)`) for the FFT2
        aperture distribution model (:math:`\\underline{E_\\mathrm{a}}(x,
        y)`).
    config_params : `dict`
        Contains the values for the fixed parameters (excluded from the least
        squares minimization), by default four parameters are kept fixed,
        ``i_amp``, ``x0``, ``y0`` and ``K(0, 0)``. See the
        ``config_params.yml`` file.

    Returns
    -------
    _residual_true : `~numpy.ndarray`
        Same output from `~pyoof.residual_true`.
        One dimensional array of the residual between the observed data and
        the FFT aperture distribution model. It has been concatenated as
        minus, zero and plus radial offset (to do a multiple fit). It is
        required to have the residual in one dimension in order to use a least
        squares minimization `~scipy.optimize.least_squares` package.

    Notes
    -----
    The **idx** key needs an indices list of the parameters to be removed.
    The structure of the parameters always follows, ``params = np.hstack([
    I_coeff, K_coeff])``, a list with ``idx = [0, 1, 2, 4]`` will remove from
    the least squares minimization, ``[i_amp, taper_dB, x0, y0, K(0, 0)]``.
    """

    # Parameters list for the true fit
    params_res = params_complete(params, idx, N_K_coeff, config_params)

    _residual_true = residual_true(
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

    return _residual_true


def params_complete(params, idx, N_K_coeff, config_params):
    """
    This function fills the missing parameters not used in the lease squares
    minimization, they are required to compute the correct aperture
    distribution, :math:`\\underline{E_\\mathrm{a}}(x, y)`. By default the
    following parameters are excluded ``i_amp``, ``x0``, ``y0``, ``K(0, 0)``. The
    parameter selection is done by default or by adding a
    ``config_params.yml`` file to the `~pyoof.fit_beam` function.

    Parameters
    ----------
    params : `~numpy.ndarray`
        Contains the incomplete array of parameters, the ``params`` array will
        be updated for the correct number of parameters to be used in the
        `~pyoof.residual_true` function. The array should be of the shape,
        ``params = np.hstack([I_coeff, K_coeff])``, missing or not some of the
        ``idx = [0, 1, 2, 3, 4]`` parameters.
    idx : `list`
        List of the positions for the removed parameters for the least squares
        minimization in the ``params`` array.
    N_K_coeff : `int`
        Total number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        ``N_K_coeff = (n + 1) * (n + 2) // 2.``.
    config_params : `dict`
        Contains the values for the fixed parameters (excluded from the least
        squares minimization), by default four parameters are kept fixed,
        ``i_amp``, ``x0``, ``y0`` and ``K(0, 0)``. See the
        ``config_params.yml`` file.

    Returns
    -------
    params_updated : `~numpy.ndarray`
        Complete set of parameters to be used in the `~pyoof.residual_true`
        function.
    """

    # Fixed values for parameters, in case they're excluded, see idx
    [i_amp_f, taper_dB_f, x0_f, y0_f, K_f] = config_params['params_fixed']

    # N_K_coeff number of Zernike circle polynomials coefficients
    if params.size != (4 + N_K_coeff):
        params_updated = params.copy()
        for i in idx:
            if i == 0:
                params_updated = np.insert(params_updated, i, i_amp_f)
                # assigned value for i_amp
            elif i == 1:
                params_updated = np.insert(params_updated, i, taper_dB_f)
                # assigned value for c_dB
            elif i == 2:
                params_updated = np.insert(params_updated, i, x0_f)
                # assigned value for x0
            elif i == 3:
                params_updated = np.insert(params_updated, i, y0_f)
                # assigned value for y0
            else:
                params_updated = np.insert(params_updated, i, K_f)
                # assigned value for any other
    else:
        params_updated = params

    return params_updated


def fit_beam(
    data_info, data_obs, method, order_max, illum_func, telescope, resolution,
    box_factor, fit_previous=True, config_params_file=None, make_plots=True,
    verbose=2
        ):
    """
    Computes the Zernike circle polynomial coefficients, ``K_coeff``, and the
    illumination function coefficients, ``I_coeff``, stores and plots data (
    optional) by using a least squares minimization. The stored data belongs
    to the best fitted power pattern (or beam map). `~pyoof.fit_beam` is the
    core function from the `~pyoof` package.

    Parameters
    ----------
    data_info : `list`
        It contains all extra data besides the beam map. The output
        corresponds to a list,
        ``[name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]``.
        These are, name of the fits file, paht of the fits file, observed
        object, observation date, frequency, wavelength, radial offset and
        mean elevation, respectively.
    data_obs : `list`
        It contains beam maps and :math:`x`-, and :math:`y`-axis
        (:math:`uv`-plane in Fourier space) data for the least squares
        minimization (see `~pyoof.fit_beam`). The list has the following order
        ``[beam_data, u_data, v_data]``. ``beam_data`` is the three beam
        observations, minus, zero and plus out-of-focus, in a flat array.
        ``u_data`` and ``v_data`` are the beam axes in a flat array.
    method : `str`
        Least squares minimization algorithm, it can be ``'trf'``, ``'lm'`` or
        ``'dogbox'``. ``'lm'`` does not handle bounds, see documentation
        `~scipy.optimize.least_squares`.
    order_max : `int`
        Maximum order used for the Zernike circle polynomials, :math:`n`, least
        squares minimization. If ``order_max = 3``, it will do the
        optimization for orders 1, 2 and 3.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telescope : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, primary radius (`float`) in meters, and telescope name
        (`str`). The list must have the following order, ``telescope =
        [block_dist, opd_func, pr, tel_name]``.
    resolution : `int`
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and with
        power of 2 for faster FFT processing. It is recommended a value higher
        than ``resolution = 2 ** 8``.
    box_factor : `int`
        Related to the FFT resolution (**resolution** key), defines the image
        pixel size level. It depends on the primary radius, ``pr``, of the
        telescope, e.g. a ``box_factor = 5`` returns ``x = np.linspace(-5 *
        pr, 5 * pr, resolution)``, an array to be used in the FFT2
        (`~numpy.fft.fft2`).
    fit_previous : `bool`
        If set to `True`, it will fit the coefficients from the previous
        optimization this feature is strongly suggested. If `False`, it will
        find the new coefficients by using the standard initial coefficients.
    config_params_file : `str`
        Path for the configuration file, this includes, the maximum and
        minimum bounds, excluded, fixed and initial parameters for the
        optimization. See ``config_params.yml`` in the pyoof package directory.
    make_plots : `bool`
        If `True` will generate a sub-directory with all the important plots
        for the OOF holography, including the phase error, :math:`\\varphi(x,
        y)` and fitted beam, :math:`P_\\mathrm{norm}(u, v)`.
    verbose : `int`
        {0, 1, 2} Level of algorithm verbosity. 0 work silent, 1 display
        termination report, 2, display progress during iteration (default).
    """

    start_time = time.time()

    print('\n ******* PYOOF FIT POWER PATTERN ******* \n')
    print('... Reading data ... \n')

    # All observed data needed to fit the beam
    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    illum_name, taper_name = illum_strings(illum_func)

    try:
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
        if not callable(illum_func):
            raise ValueError('illum_func must be a function')

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
            'Configuration file .yml does not exist in path: ' +
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
        plim_u = [np.min(u_data[0]), np.max(u_data[0])]  # radians
        plim_v = [np.min(v_data[0]), np.max(v_data[0])]  # radians
        plim_rad = np.array(plim_u + plim_v)

        # Beam normalization
        beam_data_norm = [beam_data[i] / beam_data[i].max() for i in range(3)]

        n = order                           # order polynomial to fit
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

        if method == 'lm':  # see scipy.optimize.least_squares
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

        # Running nonlinear least squares minimization
        res_lsq = optimize.least_squares(
            fun=residual,
            x0=params_init_true,
            args=(               # Conserve this order in arguments!
                idx,             # Index of parameters to be excluded (params)
                N_K_coeff,       # Total Zernike circle polynomial coeff
                beam_data_norm,  # Normalized beam maps
                u_data,          # x coordinate beam map
                v_data,          # y coordinate beam map
                d_z,             # Radial offset
                wavel,           # Wavelength of observation
                illum_func,      # Illumination function
                telgeo,          # telgeo = [block_dist, opd_func, pr]
                resolution,      # FFT2 resolution for a rectangular grid
                box_factor,      # Image pixel size level
                True,            # Grid interpolation
                config_params    # Coeff configuration for minimization (dict)
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
        params_init = params_init               # Initial parameters used
        res_optim = res_lsq.fun.reshape(3, -1)  # Optimum residual solution
        jac_optim = res_lsq.jac                 # Last Jacobian matrix
        grad_optim = res_lsq.grad               # Last Gradient

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

        # Storing files in directory
        if not verbose == 0:
            print('... Saving data ... \n')

        store_data_ascii(
            name=name,
            name_dir=name_dir,
            taper_name=taper_name,
            order=n,
            params_solution=params_solution,
            params_init=params_init,
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
            name_dir=name_dir,
            order=n,
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
