import numpy as np
from scipy.constants import c as light_speed
from main_functions import aperture, angular_spectrum, wavevector_to_degree
from scipy.optimize import least_squares
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from astropy.io import ascii
from astropy.table import Table
import os
import time
import ntpath


def find_name_path(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def residual(params, beam_data, u_data, v_data, x, y, d_z, lam, illum,
    fit_illum_params):

    if fit_illum_params:
        i_coeff = params[:4]
        U_coeff = np.insert(params[4:], 0, 0)
    else:
        i_coeff = np.array([params[0], 1.0, 0.0, 0.0])
        U_coeff = np.insert(params[1:], 0, 0)

    u0, v0, aspectrum0 = angular_spectrum(x, y, U_coeff=U_coeff, d_z=d_z[0], i_coeff=i_coeff, illum=illum)
    u1, v1, aspectrum1 = angular_spectrum(x, y, U_coeff=U_coeff, d_z=d_z[1], i_coeff=i_coeff, illum=illum)
    u2, v2, aspectrum2 = angular_spectrum(x, y, U_coeff=U_coeff, d_z=d_z[2], i_coeff=i_coeff, illum=illum)

    aspectrum = np.array([aspectrum0, aspectrum1, aspectrum2])

    beam = np.abs(aspectrum) ** 2
    beam_calculated = np.array([beam[i] / beam[i].max() for i in range(3)])

    # Generated beam u and v: wavevectors -> degrees -> radians
    u0_rad = wavevector_to_degree(u0, lam) * np.pi / 180
    u1_rad = wavevector_to_degree(u1, lam) * np.pi / 180
    u2_rad = wavevector_to_degree(u2, lam) * np.pi / 180
    v0_rad = wavevector_to_degree(v0, lam) * np.pi / 180
    v1_rad = wavevector_to_degree(v1, lam) * np.pi / 180
    v2_rad = wavevector_to_degree(v2, lam) * np.pi / 180

    # The calculated beam needs to be transformed!
    # RegularGridInterpolator
    intrp0 = RegularGridInterpolator((u0_rad, v0_rad), beam_calculated[0].T)
    intrp1 = RegularGridInterpolator((u1_rad, v1_rad), beam_calculated[1].T)
    intrp2 = RegularGridInterpolator((u2_rad, v2_rad), beam_calculated[2].T)

    # input interpolation function is the real beam
    beam_data_intrp0 = intrp0(np.array([u_data[0], v_data[0]]).T)
    beam_data_intrp1 = intrp1(np.array([u_data[1], v_data[1]]).T)
    beam_data_intrp2 = intrp2(np.array([u_data[2], v_data[2]]).T)

    beam_data_intrp = np.hstack((
        beam_data_intrp0, beam_data_intrp1, beam_data_intrp2))

    beam_data_all = np.hstack((beam_data[0], beam_data[1], beam_data[2]))

    residual = beam_data_intrp - beam_data_all

    return residual


# Insert path for the fits file with pre-calibration
def fit_beam(path_fits, order, fit_illum_params):

    start_time = time.time()

    print('####### Fit Beam pattern #######')
    print('\n')

    print('... Reading data ...')
    print('\n')
    # Opening fits file with astropy
    hdulist = fits.open(path_fits)

    # Observation frequency
    frequency = hdulist[0].header['FREQ']  # Hz
    wavelength = light_speed / frequency

    beam_data = [hdulist[i].data['fnu'] for i in range(1, 4)][::-1]
    u_data = [hdulist[i].data['DX'] for i in range(1, 4)][::-1]
    v_data = [hdulist[i].data['DY'] for i in range(1, 4)][::-1]
    d_z = [hdulist[i].header['DZ'] for i in range(1, 4)][::-1]

    # Permuting the position to provide same as main_functions
    beam_data.insert(1, beam_data.pop(2))
    u_data.insert(1, u_data.pop(2))
    v_data.insert(1, v_data.pop(2))
    d_z.insert(1, d_z.pop(2))

    # d_z is given in units of wavelength (m/m)
    d_z = np.array(d_z)
    d_z *= 2 * np.pi / wavelength  # convert to radians

    # Beam normalisation
    beam_data_norm = [beam_data[i] / beam_data[i].max() for i in range(3)]

    # Grid parameters to adjust data interpolation
    box_size = 500
    x_cal = np.linspace(-box_size, box_size, 2 ** 10)
    y_cal = np.linspace(-box_size, box_size, 2 ** 10)

    n = order  # order polynomials.

    z_coeff_to_fit = (n + 1) * (n + 2) // 2 - 1   # to give knl = 0

    if fit_illum_params:
        params_init = np.array([0.01, .9, 0, 0] + [0.1] * z_coeff_to_fit)
        # amp, sigma_r, x0, y0, K(l,m)
        # Giving an initial value of 0.1 for each coeff

        params_bounds = np.array(
            [[0, 1], [0.9, 1], [-1e-8, 1e-8], [-1e-8, 1e-8], [-2, 2]] +
            [[-1, 1]] * (z_coeff_to_fit - 1))

    else:
        params_init = np.array([0.01] + [0.1] * z_coeff_to_fit)
        # amp, K(l,m)
        # Giving an initial value of 0.1 for each coeff

        params_bounds = np.array(
            [[0, 1], [-2, 2]] + [[-1, 1]] * (z_coeff_to_fit - 1))

    print('... Starting fit ...')
    print('\n')

    res_lsq = least_squares(
        fun=residual,
        x0=params_init,
        args=(
            beam_data_norm,
            u_data,
            v_data,
            x_cal,
            y_cal,
            d_z,
            wavelength,
            'gauss',  # illumination
            fit_illum_params),  # True or False
        bounds=tuple(
            [params_bounds[:, 0].tolist(), params_bounds[:, 1].tolist()]),
        method='trf',
        verbose=2,
        # max_nfev=50
        )

    print('\n')
    print(res_lsq.message)
    print('\n')

    if fit_illum_params:
        params_solution = np.insert(res_lsq.x, 4, 0.0)
        params_init = np.insert(params_init, 4, 0)
    else:
        params_solution = np.insert(res_lsq.x, 1, [1.0, 0.0, 0.0, 0.0])
        params_init = np.insert(params_init, 1, [1.0, 0.0, 0.0, 0.0])

    # Making nice table :)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)][1:]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    params_names = ['Illum_amp', 'sigma_r', 'x_0', 'y_0', 'K(0,0)']
    for i in range(z_coeff_to_fit):
        params_names.append('K(' + str(L[i]) + ',' + str(N[i]) + ')')

    table = Table(
        {'Parameter': params_names, 'Fit': params_solution,
        'Initial guess': params_init},
        names=['Parameter', 'Fit', 'Initial guess'])
    print(table)
    print('\n')

    name_file = find_name_path(path_fits)[:-5]

    if not os.path.exists(find_name_path(name_file)):
        os.makedirs(find_name_path(name_file))

    name_to_save = name_file + '/' + name_file + '_parfit.dat'
    data_to_save = [params_names, params_solution, params_init]

    # Storing files in directory
    ascii.write(data_to_save, name_to_save, names=['parname', 'parfit', 'parinit'])

    print('####### %s mins #######' % str((time.time() - start_time) / 60))


if __name__ == "__main__":

    fit_beam('../test_data/3260_3C84_32deg_SB-002/3260_3C84_32deg_SB.fits', 5, False)
