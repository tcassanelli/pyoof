# Author: Tomas Cassanelli
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as light_speed
from main_functions import angular_spectrum, wavevector_to_degree
from scipy.optimize import least_squares
from astropy.io import fits, ascii
from scipy.interpolate import RegularGridInterpolator
from plot_routines import plot_fit_path
import os
import time
import ntpath


def find_name_path(path):
    head, tail = ntpath.split(path)
    return head, tail


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
            intrp = RegularGridInterpolator((u_rad, v_rad), beam_norm.T)

            # input interpolation function is the real beam grid
            beam_model.append(intrp(np.array([u_data[i], v_data[i]]).T))
        else:
            beam_model.append(beam_norm)

    beam_model_all = np.hstack((beam_model[0], beam_model[1], beam_model[2]))
    beam_data_all = np.hstack((beam_data[0], beam_data[1], beam_data[2]))

    # Residual = data - model (or fitted)
    residual = beam_data_all - beam_model_all

    return residual


def residual(params, idx, N_K_coeff, beam_data, u_data, v_data, d_z, lam, illum, inter):

    # params for the true fit
    params_res = np.array(params_residual(params, idx, N_K_coeff))

    print('params.shape: ', params.shape)

    print('params: ', params)

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


def params_true_fit(params, idx):
    # extracts the params given idx
    if idx is None:
        params_ture = params
    else:
        params_ture = [i for j, i in enumerate(params) if j not in idx]

    return params_ture


def params_residual(params, idx, N_K_coeff):

    params = list(params)

    if len(params) != (4 + N_K_coeff):
        for i in idx:
            if i == 1:
                params.insert(i, -8.0)  # assigned default value for c_dB
            else:
                params.insert(i, 0.0)

    return params


def extract_data_fits(pathfits):
    # Opening fits file with astropy
    hdulist = fits.open(pathfits)

    # Observation frequency
    freq = hdulist[0].header['FREQ']  # Hz
    wavel = light_speed / freq

    # name of the fit file to fit
    name = find_name_path(pathfits)[1][:-5]

    beam_data = [hdulist[i].data['fnu'] for i in range(1, 4)][::-1]
    u_data = [hdulist[i].data['DX'] for i in range(1, 4)][::-1]
    v_data = [hdulist[i].data['DY'] for i in range(1, 4)][::-1]
    d_z_m = [hdulist[i].header['DZ'] for i in range(1, 4)][::-1]

    # Permuting the position to provide same as main_functions
    beam_data.insert(1, beam_data.pop(2))
    u_data.insert(1, u_data.pop(2))
    v_data.insert(1, v_data.pop(2))
    d_z_m.insert(1, d_z_m.pop(2))

    return name, freq, wavel, d_z_m, [beam_data, u_data, v_data]


# Insert path for the fits file with pre-calibration
def fit_beam(pathfits, order, illum):

    start_time = time.time()

    print('\n ####### OOF FIT BEAM PATTERN ####### \n')
    print('... Reading data ... \n')

    name, freq, wavel, d_z_m, data = extract_data_fits(pathfits)
    [beam_data, u_data, v_data] = data

    print('File name: ', name)
    print('Observed frequency: ', freq, 'Hz')
    print('Wavelenght : ', wavel, 'm')
    print('d_z (out-of-focus): ', d_z_m, 'm')
    print('Order n to be fitted: ', order)
    print('Illumination to be fitted: ', illum)
    print('\n')

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

    params_init = [0.01, -10, 0, 0, 0] + [0.1] * (N_K_coeff - 1)
    # amp, sigma_r, x0, y0, K(l,m)
    # Giving an initial value of 0.1 for each coeff

    params_bounds = [[0, 1], [-25, -8]] + [[-1e-4, 1e-4]] * 2 + [[-2.2, 2.2]] * N_K_coeff

    idx = [1, 2, 3, 4]  # exclude params from fit
    # [1, 2, 3, 4] = [c_dB, x0, y0, K(0,0)] or 'None' to include them all
    params_init_true = params_true_fit(params_init, idx)
    params_bounds_true = np.array(params_true_fit(params_bounds, idx))

    print(
        '... Starting fit for ' + str(len(params_init_true)) +
        ' parameters ... \n'
        )

    # Running non-linear least-squared optimization
    res_lsq = least_squares(
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
        bounds=tuple([params_bounds_true[:, 0], params_bounds_true[:, 1]]),
        method='trf',
        verbose=2,
        max_nfev=2
        )

    print('\n')

    # Solutions from least squared optimisation
    params_solution = params_residual(res_lsq.x.tolist(), idx, N_K_coeff)
    params_init = params_init
    res_optim = res_lsq.fun.reshape(3, -1)  # Optimum residual from fitting
    jac_optim = res_lsq.jac
    grad_optim = res_lsq.grad

    # Making nice table :)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    params_names = ['illum_amp', 'c_dB', 'x_0', 'y_0']
    for i in range(N_K_coeff):
        params_names.append('K(' + str(L[i]) + ',' + str(N[i]) + ')')

    # Storing files in OOF_out directory
    name_dir = find_name_path(pathfits)[0] + '/OOF_out/' + name

    if not os.path.exists(find_name_path(pathfits)[0] + '/OOF_out'):
        os.makedirs(find_name_path(pathfits)[0] + '/OOF_out')
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

    params_to_save = [params_names, params_solution, params_init]
    info_to_save = [
        [name], [d_z_m[0]], [d_z_m[1]], [d_z_m[2]], [wavel],
        [freq], [n], [illum]
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
        output=name_dir + '/fitinfo_n' + str(n) + '.dat',
        names=['name', 'd_z-', 'd_z0', 'd_z+', 'wavel', 'freq', 'n', 'illum'],
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

    # Making all relevant plots
    print('... Making plots ... \n')

    plot_fit_path(
        pathoof=name_dir + '/',
        order=n,
        plim_rad=plim_rad,
        save=True,
        rad=False
        )

    print(' ###### %s mins ######' % str((time.time() - start_time) / 60))
    print('\n')

    # plt.show()

    plt.close()


if __name__ == "__main__":

    for n in [4]:
        # Testing script
        fit_beam(
            # pathfits='../test_data/gen_data7/gendata7_o3n0.fits',
            pathfits='../test_data/S9mm_0397_3C84/S9mm_0397_3C84_H1_SB.fits',
            order=n,
            illum='pedestal'
            )
