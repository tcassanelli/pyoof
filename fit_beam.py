# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy import interpolate, optimize
from plot_routines import plot_fit_path
import os
import time
from main_functions import (
    angular_spectrum, wavevector_to_degree, par_variance, sr_phase
    )
from aux_functions import extract_data_fits


def residual_true(params, beam_data, u_data, v_data, d_z, lam, illum, inter):

    I_coeff = params[:4]
    K_coeff = params[4:]

    beam_model = []
    for i in range(3):

        u, v, aspectrum = angular_spectrum(
            K_coeff=K_coeff,
            d_z=d_z[i],
            I_coeff=I_coeff,
            illum=illum
            )

        beam = np.abs(aspectrum) ** 2
        beam_norm = beam / beam.max()

        if inter:

            # Generated beam u and v: wavevectors -> degrees -> radians
            u_rad = wavevector_to_degree(u, lam) * np.pi / 180
            v_rad = wavevector_to_degree(v, lam) * np.pi / 180

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
    params, idx, N_K_coeff, beam_data, u_data, v_data, d_z, lam, illum, inter
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
        illum=illum,
        inter=inter
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
            if i == 1:
                _params = np.insert(_params, i, -8.0)
                # assigned default value for c_dB
            if i >= 2:
                _params = np.insert(_params, i, 0.0)
                # for x0, y0 and K(l, n) coefficients
    else:
        _params = params

    return _params


# Insert path for the fits file with pre-calibration
def fit_beam(pathfits, order, illum, fit_previous):

    start_time = time.time()

    print('\n ####### OOF FIT POWER PATTERN ####### \n')
    print('... Reading data ... \n')

    name, freq, wavel, d_z_m, meanel, pthto, data = extract_data_fits(pathfits)
    [beam_data, u_data, v_data] = data

    print('File name: ', name)
    print('Observed frequency: ', freq, 'Hz')
    print('Wavelenght : ', wavel, 'm')
    print('d_z (out-of-focus): ', d_z_m, 'm')
    print('Order n to be fitted: ', order)
    print('Illumination to be fitted: ', illum)

    # Setting limits for plotting fitted beam
    plim_u = [np.min(u_data[0]), np.max(u_data[0])]
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

    # Looking for result parameters from lower order to use them
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
        params_init = np.array([0.1, -8, 0, 0, 0] + [0.1] * (N_K_coeff - 1))
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
        args=(
            idx,
            N_K_coeff,
            beam_data_norm,
            u_data,
            v_data,
            d_z,
            wavel,
            illum,
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

    cov, corr = par_variance(
        res=res_lsq.fun,
        jac=res_lsq.jac,
        n_pars=params_init_true.size  # num of parameters fitted
        )
    cov_ptrue = np.vstack((np.delete(np.arange(N_K_coeff + 4), idx), cov))
    corr_ptrue = np.vstack((np.delete(np.arange(N_K_coeff + 4), idx), corr))

    # Making nice table :)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    params_names = ['illum_amp', 'c_dB', 'x_0', 'y_0']
    for i in range(N_K_coeff):
        params_names.append('K(' + str(L[i]) + ',' + str(N[i]) + ')')

    params_to_save = [params_names, params_solution, params_init]
    info_to_save = [
        [name], [d_z_m[0]], [d_z_m[1]], [d_z_m[2]], [wavel],
        [freq], [illum], [meanel]
        ]

    # Storing files in directory
    path_params = name_dir + '/fitpar_n' + str(n) + '.dat'
    ascii.write(
        table=params_to_save,
        output=path_params,
        names=['parname', 'parfit', 'parinit']
        )

    ascii.write(
        table=info_to_save,
        output=name_dir + '/fitinfo.dat',
        names=[
            'name', 'd_z-', 'd_z0', 'd_z+', 'wavel', 'freq', 'illum', 'meanel'
            ],
        fast_writer=False
        )

    # Printing the results from saved ascii file
    print(ascii.read(path_params))
    print('\n')

    np.savetxt(name_dir + '/beam_data.csv', beam_data)
    np.savetxt(name_dir + '/u_data.csv', u_data)
    np.savetxt(name_dir + '/v_data.csv', v_data)
    np.savetxt(name_dir + '/res_n' + str(n) + '.csv', res_optim)
    np.savetxt(name_dir + '/jac_n' + str(n) + '.csv', jac_optim)
    np.savetxt(name_dir + '/grad_n' + str(n) + '.csv', grad_optim)

    # Storing phase for subreflector analysis
    np.savetxt(
        fname=name_dir + '/phase_n' + str(n) + '.csv',
        X=sr_phase(params=params_solution, notilt=True)
        )

    np.savetxt(
        fname=name_dir + '/cov_n' + str(n) + '.csv',
        X=cov_ptrue,
        header='Cov matrix (first row included elements)',
        )

    np.savetxt(
        fname=name_dir + '/corr_n' + str(n) + '.csv',
        X=corr_ptrue,
        header='Corr matrix (first row included elements)',
        )

    # Making all relevant plots
    print('... Making plots ... \n')

    plot_fit_path(
        pathoof=name_dir,
        order=n,
        plim_rad=plim_rad,
        save=True,
        rad=False
        )

    print(' ###### %s mins ######' % str((time.time() - start_time) / 60))
    print('\n')

    # plt.show()

    plt.close('all')


if __name__ == "__main__":

    import glob
    observations = glob.glob('../data/S9mm/*.fits')  # len = 8

    for n in [6]:
        fit_beam(
            pathfits=observations[7],
            order=n,
            illum='pedestal',
            fit_previous=True
            )
