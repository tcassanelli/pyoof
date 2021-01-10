#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as apu
from astropy.utils.data import get_pkg_data_filename
from scipy import interpolate, optimize
import os
import time
import yaml
from .aperture import radiation_pattern, phase
from .math_functions import co_matrices, norm, snr
from .plot_routines import plot_fit_path
from .aux_functions import store_data_csv, store_data_ascii

__all__ = [
    'residual_true', 'residual', 'params_complete', 'fit_zpoly',
    ]


def residual_true(
    params, beam_data, u_data, v_data, d_z, wavel, illum_func, telgeo,
    resolution, box_factor, interp
        ):
    """
    Computes the true residual ready to use for the `~pyoof.fit_zpoly`
    function. True means that some of the parameters used will **not** be
    fitted. Their selection is done by default or by adding
    ``config_params.yml`` file to the `~pyoof.fit_zpoly` function.

    Parameters
    ----------
    params : `~numpy.ndarray`
        Two stacked arrays, the illumination and Zernike circle polynomials
        coefficients. ``params = np.hstack([I_coeff, K_coeff])``.
    beam_data : `~numpy.ndarray`
        The ``beam_data`` is an array with the three observed beam maps,
        :math:`P^\\mathrm{obs}(u, v)`, minus, zero and plus out-of-focus.
    u_data : `~astropy.units.quantity.Quantity`
        :math:`x` axis value for the 3 beam maps in radians. The values have
        to be flatten, in one dimension, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    v_data : `~astropy.units.quantity.Quantity`
        :math:`y` axis value for the 3 beam maps in radians. The values have
        to be flatten, one dimensional, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    d_z : `~astropy.units.quantity.Quantity`
        Radial offset :math:`d_z`, added to the sub-reflector in length units.
        This characteristic measurement adds the classical interference
        pattern to the beam maps, normalized squared (field) radiation
        pattern, which is an out-of-focus property. The radial offset list
        must be as follows, ``d_z = [d_z-, 0., d_z+]`` all of them in length
        units.
    wavel : `~astropy.units.quantity.Quantity`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key ``I_coeff``. The illumination functions available are
        `~pyoof.aperture.illum_parabolic` and `~pyoof.aperture.illum_gauss`.
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
        the observed grid (:math:`P^\\mathrm{obs}(u, v)`) and the computed
        grid (:math:`P(u, v)`) for the FFT2 aperture distribution model
        (:math:`\\underline{E_\\mathrm{a}}(x, y)`).

    Returns
    -------
    _residual_true : `~numpy.ndarray`
        One dimensional array of the residual between the observed data and
        the FFT aperture distribution model. It has been concatenated as
        minus, zero and plus radial offset (to do a multiple fit). It is
        required to have the residual in one dimension in order to use a least
        squares minimization `~scipy.optimize.least_squares` package.
    """

    I_coeff, K_coeff = params[:5], params[5:]
    beam_model = np.zeros_like(beam_data)
    for i in range(3):
        u, v, F = radiation_pattern(
            I_coeff=I_coeff,
            K_coeff=K_coeff,
            d_z=d_z[i],
            wavel=wavel,
            illum_func=illum_func,
            telgeo=telgeo,
            resolution=resolution,
            box_factor=box_factor
            )

        power_pattern = np.abs(F) ** 2

        if interp:

            # The calculated beam needs to be transformed!
            intrp = interpolate.RegularGridInterpolator(
                points=(u.to_value(apu.rad), v.to_value(apu.rad)),
                values=power_pattern.T,     # data in grid
                method='linear'             # linear or nearest
                )

            # input interpolation function is the real beam grid
            beam_model[i, ...] = (
                intrp(np.array([
                    u_data[i, ...].to_value(apu.rad),
                    v_data[i, ...].to_value(apu.rad)
                    ]).T)
                )
        else:
            beam_model[i, ...] = power_pattern

    _residual_true = norm(beam_data, axis=1) - norm(beam_model, axis=1)

    return _residual_true.flatten()


def residual(
    params, N_K_coeff, beam_data, u_data, v_data, d_z, wavel,
    illum_func, telgeo, resolution, box_factor, interp, config_params
        ):
    """
    Wrapper for the `~pyoof.residual_true` function. The objective of
    this function is to fool the `~scipy.optimize.least_squares` package by
    changing the number of parameters that will be used in the fit. The
    parameter array must be organized as follows, ``params = np.hstack([
    I_coeff, K_coeff])``. The parameter selection is done by default or by
    adding a ``config_params.yml`` file to the `~pyoof.fit_zpoly` function.

    Parameters
    ----------
    params : `~numpy.ndarray`
        Two stacked arrays, the illumination and Zernike circle polynomials
        coefficients. ``params = np.hstack([I_coeff, K_coeff])``.
    N_K_coeff : `int`
        Total number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        ``N_K_coeff = (n + 1) * (n + 2) // 2.``.
    beam_data : `~numpy.ndarray`
        The ``beam_data`` is an array with the three observed beam maps,
        :math:`P^\\mathrm{obs}(u, v)`, minus, zero and plus out-of-focus.
    u_data : `~astropy.units.quantity.Quantity`
        :math:`x` axis value for the 3 beam maps in radians. The values have
        to be flatten, in one dimension, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    v_data : `~astropy.units.quantity.Quantity`
        :math:`y` axis value for the 3 beam maps in radians. The values have
        to be flatten, one dimensional, and stacked in the same order as the
        ``d_z = [d_z-, 0., d_z+]`` values from each beam map.
    d_z : `~astropy.units.quantity.Quantity`
        Radial offset :math:`d_z`, added to the sub-reflector in length units.
        This characteristic measurement adds the classical interference
        pattern to the beam maps, normalized squared (field) radiation
        pattern, which is an out-of-focus property. The radial offset list
        must be as follows, ``d_z = [d_z-, 0., d_z+]`` all of them in length
        units.
    wavel : `~astropy.units.quantity.Quantity`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key ``I_coeff``. The illumination functions available are
        `~pyoof.aperture.illum_parabolic` and `~pyoof.aperture.illum_gauss`.
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
        the observed grid (:math:`P^\\mathrm{obs}(u, v)`) and the computed
        grid (:math:`P(u, v)`) for the FFT2 aperture distribution model
        (:math:`\\underline{E_\\mathrm{a}}(x, y)`).
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
    The **idx_exclude** key (``config_params['excluded']``) needs an
    indices list of the parameters to be removed. The structure of the
    parameters always follows, ``params = np.hstack([I_coeff, K_coeff])``, a
    list with ``idx_exclude = [0, 1, 2, 4, 5, 6, 7]`` will remove from
    the least squares minimization, ``[i_amp, c_dB, q, x0, y0, K(0, 0),
    K(1, 1), K(1, -1)]``.
    """

    # Parameters list for the true fit
    params_res = params_complete(params, N_K_coeff, config_params)

    _residual_true = residual_true(
        params=params_res,
        beam_data=beam_data,
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


def params_complete(params, N_K_coeff, config_params):
    """
    This function fills the missing parameters not used in the lease squares
    minimization, they are required to compute the correct aperture
    distribution, :math:`\\underline{E_\\mathrm{a}}(x, y)`.
    The parameter selection is done by default or by adding a
    ``config_params.yml`` file to the `~pyoof.fit_zpoly` function.

    Parameters
    ----------
    params : `~numpy.ndarray`
        Contains the incomplete array of parameters, the ``params`` array will
        be updated for the correct number of parameters to be used in the
        `~pyoof.residual_true` function. The array should be of the shape,
        ``params = np.hstack([I_coeff, K_coeff])``, missing or not some of the
        ``idx_exclude = [0, 1, 2, 3, 4, 5, 6, 7]`` parameters.
    N_K_coeff : `int`
        Total number of Zernike circle polynomials coefficients to fit. It is
        obtained from the order to be fitted with the formula
        ``N_K_coeff = (n + 1) * (n + 2) // 2.``.
    config_params : `dict`
        Contains the values for the fixed parameters (excluded from the least
        squares minimization), for default parameters, see the
        ``config_params.yml`` file.

    Returns
    -------
    params_updated : `~numpy.ndarray`
        Complete set of parameters to be used in the `~pyoof.residual_true`
        function.
    """
    [
        i_amp_f, c_dB_f, q_f, x0_f, y0_f, Knl0_f, Knl1_f, Knl2_f
        ] = config_params['fixed']

    # N_K_coeff number of Zernike circle polynomials coefficients
    if params.size != (5 + N_K_coeff):
        params_updated = params.copy()
        for i in config_params['excluded']:
            if i == 0:
                params_updated = np.insert(params_updated, i, i_amp_f)
            elif i == 1:
                params_updated = np.insert(params_updated, i, c_dB_f)
            elif i == 2:
                params_updated = np.insert(params_updated, i, q_f)
            elif i == 3:
                params_updated = np.insert(params_updated, i, x0_f)
            elif i == 4:
                params_updated = np.insert(params_updated, i, y0_f)
            elif i == 5:
                params_updated = np.insert(params_updated, i, Knl0_f)
            elif i == 6:
                params_updated = np.insert(params_updated, i, Knl1_f)
            elif i == 7:
                params_updated = np.insert(params_updated, i, Knl2_f)
    else:
        params_updated = params

    return params_updated


def fit_zpoly(
    data_info, data_obs, order_max, illum_func, telescope, resolution,
    box_factor, fit_previous=True, config_params_file=None, make_plots=False,
    verbose=2, work_dir=None
        ):
    """
    Computes the Zernike circle polynomial coefficients, ``K_coeff``, and the
    illumination function coefficients, ``I_coeff``, stores and plots data (
    optional) by using a least squares minimization. The stored data belongs
    to the best fitted power pattern (or beam map). `~pyoof.fit_zpoly` is the
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
        minimization (see `~pyoof.fit_zpoly`). The list has the following order
        ``[beam_data, u_data, v_data]``. ``beam_data`` is the three beam
        observations, minus, zero and plus out-of-focus, in a flat array.
        ``u_data`` and ``v_data`` are the beam axes in a flat array.
    order_max : `int`
        Maximum order used for the Zernike circle polynomials, :math:`n`, least
        squares minimization. If ``order_max = 3``, it will do the
        optimization for orders 1, 2 and 3.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key ``I_coeff``. The illumination functions available are
        `~pyoof.aperture.illum_parabolic` and `~pyoof.aperture.illum_gauss`.
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
        for the OOF holography, including the phase-error, :math:`\\varphi(x,
        y)` and fitted beam, :math:`P_\\mathrm{norm}(u, v)`.
    verbose : `int`
        {0, 1, 2} Level of algorithm verbosity. 0 work silent, 1 display
        termination report, 2, display progress during iteration (default).
    work_dir : `str`
        Default is `None`, it will store the ``pyoof_out/`` folder in the fits
        file current directory, for other provide the desired path.
    """

    start_time = time.time()

    print('\n ***** PYOOF FIT POLYNOMIALS ***** \n')
    print(' ... Reading data ...\n')

    # All observed data needed to fit the beam
    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    if work_dir is None:
        work_dir = pthto

    telgeo, tel_name = telescope[:3], telescope[3]

    # Calling default configuration file from the pyoof package
    if config_params_file is None:
        config_params_pyoof = get_pkg_data_filename('data/config_params.yml')
        with open(config_params_pyoof, 'r') as yaml_config:
            config_params = yaml.load(yaml_config, Loader=yaml.Loader)
    else:
        with open(config_params_file, 'r') as yaml_config:
            config_params = yaml.load(yaml_config, Loader=yaml.Loader)

    # Storing files in pyoof_out directory
    if not os.path.exists(os.path.join(work_dir, 'pyoof_out')):
        os.makedirs(os.path.join(work_dir, 'pyoof_out'), exist_ok=True)

    for j in ["%03d" % i for i in range(101)]:
        name_dir = os.path.join(work_dir, 'pyoof_out', name + '-' + j)
        if not os.path.exists(name_dir):
            os.makedirs(name_dir, exist_ok=True)
            break

    _snr = []
    for i in range(3):
        _snr.append(
            np.round(snr(beam_data[i, ...], u_data[i, ...], v_data[i, ...]), 2)
            )

    print(
        f'Maximum order to be fitted: {order_max}',
        f'Telescope name: {tel_name}',
        f'File name: {name}',
        f'Obs frequency: {freq.to(apu.GHz)}',
        f'Obs Wavelength: {wavel.to(apu.cm)}',
        f'Mean elevation {meanel.to(apu.deg)}',
        f'd_z (out-of-focus): {d_z.to(apu.cm)}',
        f'Illumination to be fitted: {illum_func.__qualname__}',
        f'SNR out-, in-, and out-focus beam: {_snr}',
        f'Beam data shape: {beam_data.shape}',
        sep='\n',
        end='\n'
        )

    for order in range(1, order_max + 1):

        if not verbose == 0:
            print('\n ... Fit order {} ... \n'.format(order))

        # Setting limits for plotting fitted beam
        plim = np.array([
            u_data[0, ...].min().to_value(apu.rad),
            u_data[0, ...].max().to_value(apu.rad),
            v_data[0, ...].min().to_value(apu.rad),
            v_data[0, ...].max().to_value(apu.rad)
            ]) * u_data.unit

        n = order                           # order polynomial to fit
        N_K_coeff = (n + 1) * (n + 2) // 2  # number of K(n, l) to fit

        # Looking for result parameters lower order
        if fit_previous and n != 1:
            N_K_coeff_previous = n * (n + 1) // 2

            path_params_previous = os.path.join(
                name_dir, f'fitpar_n{n - 1}.csv'
                )

            params_to_add = np.ones(N_K_coeff - N_K_coeff_previous) * 0.1
            params_previous = Table.read(path_params_previous, format='ascii')
            params_init = np.hstack((params_previous['parfit'], params_to_add))

            if not verbose == 0:
                print('Initial params: n={} fit'.format(n - 1))
        else:
            params_init = config_params['init'] + [0.1] * (N_K_coeff - 3)
            print('Initial parameters: default')
            # i_amp, c_dB, q, x0, y0, K(n, l)
            # Giving an initial value of 0.1 for each K_coeff

        idx_exclude = config_params['excluded']  # exclude params from fit
        # [0, 1, 2, 3, 4, 5, 6, 7] =
        # [i_amp, c_dB, q, x0, y0, K(0, 0), K(1, 1), K(1, -1)]
        # or 'None' to include all

        params_init_true = np.delete(params_init, idx_exclude)

        bound_min = config_params['bound_min'] + [-5] * (N_K_coeff - 3)
        bound_max = config_params['bound_max'] + [5] * (N_K_coeff - 3)

        bound_min_true = np.delete(bound_min, idx_exclude)
        bound_max_true = np.delete(bound_max, idx_exclude)

        if not verbose == 0:
            print('Parameters to fit: {}\n'.format(len(params_init_true)))

        # Running nonlinear least squares minimization
        res_lsq = optimize.least_squares(
            fun=residual,
            x0=params_init_true,
            args=(               # Conserve this order in arguments!
                N_K_coeff,       # Total Zernike circle polynomial coeff
                beam_data,       # Power pattern maps
                u_data,          # x coordinate beam map
                v_data,          # y coordinate beam map
                d_z,             # Radial offset
                wavel,           # Wavelength of observation
                illum_func,      # Illumination function
                telgeo,          # telgeo = [block_dist, opd_func, pr]
                resolution,      # FFT2 resolution for a rectangular grid
                box_factor,      # Image pixel size level
                True,            # Grid interpolation
                config_params    # Params configuration for minimization (dict)
                ),
            bounds=tuple([bound_min_true, bound_max_true]),
            method='trf',
            tr_solver='exact',
            verbose=verbose,
            max_nfev=None
            )

        # Solutions from least squared optimization
        params_solution = params_complete(
            params=res_lsq.x,
            N_K_coeff=N_K_coeff,
            config_params=config_params
            )
        res_optim = res_lsq.fun.reshape(3, -1)  # Optimum residual solution
        jac_optim = res_lsq.jac                 # Last Jacobian matrix
        grad_optim = res_lsq.grad               # Last Gradient

        # covariance and correlation
        cov, cor = co_matrices(
            res=res_lsq.fun,
            jac=res_lsq.jac,
            n_pars=params_init_true.size        # number of parameters fitted
            )
        cov_ptrue = np.vstack(
            (np.delete(np.arange(N_K_coeff + 5), idx_exclude), cov))
        cor_ptrue = np.vstack(
            (np.delete(np.arange(N_K_coeff + 5), idx_exclude), cor))

        # Final phase from fit in the telescope's primary reflector
        _phase = phase(
            K_coeff=params_solution[5:],
            pr=telgeo[2],
            piston=False,
            tilt=False
            )[2].to_value(apu.rad)

        # Storing files in directory
        if not verbose == 0:
            print('\n ... Saving data ...\n')

        store_data_ascii(
            name=name,
            name_dir=name_dir,
            order=n,
            params_solution=params_solution,
            params_init=params_init,
            )

        # Printing the results from saved ascii file
        if not verbose == 0:
            Table.read(
                os.path.join(name_dir, f'fitpar_n{n}.csv'),
                format='ascii'
                ).pprint_all()

        if n == 1:
            # TODO: yaml doesn't like astropy :(
            pyoof_info = dict(
                tel_name=tel_name,
                # tel_blockage=telgeo[0].__qualname__,
                tel_opd=telgeo[1].__qualname__,
                pr=float(telgeo[2].to_value(apu.m)),
                name=name,
                obs_object=obs_object,
                obs_date=obs_date,
                d_z=d_z.to_value(apu.m).tolist(),
                wavel=float(wavel.to_value(apu.m)),
                frequency=float(freq.to_value(apu.Hz)),
                illumination=illum_func.__qualname__,
                meanel=float(meanel.to_value(apu.deg)),
                fft_resolution=resolution,
                box_factor=box_factor,
                snr=list(float(_snr[i]) for i in range(3))
                )

            with open(os.path.join(name_dir, 'pyoof_info.yml'), 'w') as outf:
                outf.write('# pyoof relevant information\n')
                yaml.dump(
                    pyoof_info, outf,
                    default_flow_style=False,
                    Dumper=yaml.Dumper
                    )

        # To store large files in csv format
        save_to_csv = [
            beam_data, u_data.to_value(apu.rad), v_data.to_value(apu.rad),
            res_optim, jac_optim, grad_optim, _phase, cov_ptrue, cor_ptrue
            ]

        store_data_csv(
            name=name,
            name_dir=name_dir,
            order=n,
            save_to_csv=save_to_csv
            )

        if make_plots:
            if not verbose == 0:
                print('\n ... Making plots ...')

            # Making all relevant plots
            plot_fit_path(
                path_pyoof_out=name_dir,
                order=n,
                telgeo=telgeo,
                illum_func=illum_func,
                plim=plim,
                save=True,
                angle=apu.deg
                )

            plt.close('all')

    final_time = np.round((time.time() - start_time) / 60, 2)
    print(f'\n ***** PYOOF FIT COMPLETED AT {final_time} mins *****\n')
