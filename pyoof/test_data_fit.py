import numpy as np
from scipy.constants import c as light_speed
from main_functions import aperture, angular_spectrum, wavevector_to_degree
from scipy.optimize import least_squares
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from astropy.io import ascii
from astropy.table import Table

import time
start_time = time.time()

print('Starting Fit')

##### Real data #######
# Calling data from .fits file
path = '../test_data/3260_3C84_32deg_SB-002/3260_3C84_32deg_SB.fits'

hdulist = fits.open(path)

focus = hdulist[1].data['fnu']
x_f = hdulist[1].data['DX']
y_f = hdulist[1].data['DY']

focus_plus = hdulist[2].data['fnu']
x_fplus = hdulist[2].data['DX']
y_fplus = hdulist[2].data['DY']

focus_minus = hdulist[3].data['fnu']
x_fminus = hdulist[3].data['DX']
y_fminus = hdulist[3].data['DY']

beam_data = [focus_minus / focus_minus.max(), focus / focus.max(), focus_plus / focus_plus.max()]
u_b_data = [x_fminus, x_f, x_fplus]
v_b_data = [y_fminus, y_f, y_fplus]

# print('beam_data[0].shape: ', beam_data[0].shape)
# print('u_b_data[0].shape: ', u_b_data[0].shape)
# print('v_b_data[0].shape: ', v_b_data[0].shape)
############################

# d_z has to be given in units of wavelength
d_z = np.array([-2.5, 0., 2.5])  # Unit m/m (m/lamb)
d_z *= 2 * np.pi  # convert to radians

box_size = 500
x_cal = np.linspace(-box_size, box_size, 2 ** 10)
y_cal = np.linspace(-box_size, box_size, 2 ** 10)

frequency = 32e9  # Hz
wavelength = light_speed / frequency

n = 5  # order polynomials.

# ######### Data generation #########
# params_ = np.array([0, -2.298295e-1, 4.93943e-1, -1.379757e-1, -1.819459e-1, -9.78374e-2, 6.137946e-1, -1.684147e-1, 1.348733e-1, -2.600830e-1, 3.05227e-2, -1.045454e-1, 2.149645e-2, 6.240493e-2, -9.050347e-2, -5.502480e-1, 2.346242e-1, -3.50973e-1, -1.287273e-2, 3.709351e-1, -2.244371e-1])

# x_data = np.linspace(-box_size, box_size, 2 ** 10)
# y_data = np.linspace(-box_size, box_size, 2 ** 10)


# def gen_data(x, y, params=params_, c_db=-20, amp=1e7, noise=0, d_z=d_z, n=5, lam=wavelength):

#     # x and y must be linear
#     u0, v0, aspectrum0 = angular_spectrum(x, y, params, d_z[0], n, c_db)
#     u1, v1, aspectrum1 = angular_spectrum(x, y, params, d_z[1], n, c_db)
#     u2, v2, aspectrum2 = angular_spectrum(x, y, params, d_z[2], n, c_db)

#     u0_rad = wavevector_to_degree(u0, lam) * np.pi / 180
#     u1_rad = wavevector_to_degree(u1, lam) * np.pi / 180
#     u2_rad = wavevector_to_degree(u2, lam) * np.pi / 180
#     v0_rad = wavevector_to_degree(v0, lam) * np.pi / 180
#     v1_rad = wavevector_to_degree(v1, lam) * np.pi / 180
#     v2_rad = wavevector_to_degree(v2, lam) * np.pi / 180

#     a = 438  # to generate rectangle and take important data
#     b = 587
#     div = 3  # number of divisons to take out of data

#     u0_grid, v0_grid = np.meshgrid(u0_rad[a:b:div], v0_rad[a:b:div])
#     u1_grid, v1_grid = np.meshgrid(u1_rad[a:b:div], v1_rad[a:b:div])
#     u2_grid, v2_grid = np.meshgrid(u2_rad[a:b:div], v2_rad[a:b:div])

#     u0_flat, v0_flat = u0_grid.flatten(), v0_grid.flatten()
#     u1_flat, v1_flat = u1_grid.flatten(), v1_grid.flatten()
#     u2_flat, v2_flat = u2_grid.flatten(), v2_grid.flatten()

#     beam = amp * np.array([np.abs(aspectrum0) ** 2, np.abs(aspectrum1) ** 2, np.abs(aspectrum2) ** 2])

#     beam_noise = beam + np.random.normal(0., noise, beam.shape)
#     beam_noise_flat = [beam_noise[0][a:b:div, a:b:div].flatten(), beam_noise[1][a:b:div, a:b:div].flatten(), beam_noise[2][a:b:div, a:b:div].flatten()]

#     return [u0_flat, u1_flat, u2_flat], [v0_flat, v1_flat, v2_flat], beam_noise_flat

# u_gen, v_gen, data_gen = gen_data(x=x_data, y=y_data, noise=2)

# np.save('tests_solutions/u_gen', u_gen)
# np.save('tests_solutions/v_gen', v_gen)
# np.save('tests_solutions/data_gen', data_gen)

# print('data_gen[0].shape: ', data_gen[0].shape)
# print('u_gen[0].shape: ', u_gen[0].shape)
# print('v_gen[0].shape: ', v_gen[0].shape)
# #####################################


# Function to compute the residuals and initial estimate of parameters
def residual(coeff, beam_data, u_b_data, v_b_data, x=x_cal, y=y_cal, d_z=d_z, n=n, lam=wavelength):
    # The coefficients are [c_db, params]

    # Computation angular spectrum then beam
    # x_cal and y_cal are to calculate the angular spectrum
    # u_b_data and v_b_data wavevector in radiands for data

    # c_db = coeff[1]
    i_coeff = coeff[:4]

    U_coeff = np.array([0] + coeff[4:].tolist())  # aisolate zernike coefficients

    u0, v0, aspectrum0 = angular_spectrum(x, y, U_coeff=U_coeff, d_z=d_z[0], n=n, i_coeff=i_coeff, illum='pedestal')
    u1, v1, aspectrum1 = angular_spectrum(x, y, U_coeff=U_coeff, d_z=d_z[1], n=n, i_coeff=i_coeff, illum='pedestal')
    u2, v2, aspectrum2 = angular_spectrum(x, y, U_coeff=U_coeff, d_z=d_z[2], n=n, i_coeff=i_coeff, illum='pedestal')

    aspectrum = np.array([aspectrum0, aspectrum1, aspectrum2])

    beam_calculated = np.abs(aspectrum) ** 2

    # Generated beam wavevectors (degree)
    u0_rad = wavevector_to_degree(u0, lam) * np.pi / 180
    u1_rad = wavevector_to_degree(u1, lam) * np.pi / 180
    u2_rad = wavevector_to_degree(u2, lam) * np.pi / 180
    v0_rad = wavevector_to_degree(v0, lam) * np.pi / 180
    v1_rad = wavevector_to_degree(v1, lam) * np.pi / 180
    v2_rad = wavevector_to_degree(v2, lam) * np.pi / 180

    # print('x_cal.shape: ', x_cal.shape)
    # print('y_cal.shape: ', y_cal.shape)
    # print('u0.shape: ', u0.shape)
    # print('v0.shape: ', v0.shape)
    # print('aspectrum0.shape: ', aspectrum0.shape)
    # print('beam_calculated[0].shape: ', beam_calculated[0].shape)
    # print('u0_rad.shape: ', u0_rad.shape)
    # print('v0_rad.shape: ', v0_rad.shape)
    # print('u_b_data[0].shape: ', u_b_data[0].shape)
    # print('u_b_data[0].shape: ', u_b_data[0].shape)

    # The calculated beam needs to be transformed!
    # RegularGridInterpolator
    intrp0 = RegularGridInterpolator((u0_rad, v0_rad), beam_calculated[0].T)
    intrp1 = RegularGridInterpolator((u1_rad, v1_rad), beam_calculated[1].T)
    intrp2 = RegularGridInterpolator((u2_rad, v2_rad), beam_calculated[2].T)


    # input interpolation function is the real beam
    beam_data_intrp0 = intrp0(np.array([u_b_data[0], v_b_data[0]]).T)
    beam_data_intrp1 = intrp1(np.array([u_b_data[1], v_b_data[1]]).T)
    beam_data_intrp2 = intrp2(np.array([u_b_data[2], v_b_data[2]]).T)

    beam_data_intrp = np.hstack((beam_data_intrp0, beam_data_intrp1, beam_data_intrp2))

    # print('beam_data_intrp0: ', beam_data_intrp0)
    # print('beam_data_intrp0.shape: ', beam_data_intrp0.shape)

    beam_data_all = np.hstack((beam_data[0], beam_data[1], beam_data[2]))

    residual = beam_data_intrp - beam_data_all

    # print('beam_data_all.shape: ', beam_data_all.shape)
    # print('residual.shape: ', residual.shape)
    # print('res: ', residual)

    return residual


z_coeff_to_fit = (n + 1) * (n + 2) // 2 - 1   # to give knl = 0

# Using the Gauss Illuminatin function
i_coeff_init = [42, .5, 0, 0]  # amp, sigma_r, x0, y0
initial_coeff = np.array(i_coeff_init + [0.1] * z_coeff_to_fit)
# Giving an initial value of 0.1 for each coeff

i_coeff_bound_inf = [0, 0, -1e-3, -1e-3] + [-2] + [-1] * (z_coeff_to_fit - 1)
i_coeff_bound_sup = [np.inf, 1, 1e-3, 1e-3] + [2] + [1] * (z_coeff_to_fit - 1)

# Using the Pedestal Illumination
# amp, sigma_r, x0, y0, U(-1,1), ...
# initial_coeff = np.array([40, -5, 0, 0] + [0.1] * z_coeff_to_fit)
# i_coeff_bound_inf = [0, -20, -1e-5, -1e-5] + [-2] + [-1] * (z_coeff_to_fit - 1)
# i_coeff_bound_sup = [np.inf, 0, 1e-5, 1e-5] + [2] + [1] * (z_coeff_to_fit - 1)


res_lsq = least_squares(
    fun=residual,
    x0=initial_coeff,
    args=(beam_data, u_b_data, v_b_data),
    bounds=tuple([i_coeff_bound_inf, i_coeff_bound_sup]),
    method='trf',
    verbose=2,
    max_nfev=20
    )

print(res_lsq.message)
# print('Fitted coefficients: ', res_lsq.x)

# Making nice table :)
ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)][1:]
L = np.array(ln)[:, 0]
N = np.array(ln)[:, 1]

params_name = ['Illum amp', 'sigma_r', 'x_0', 'y_0', 'U(0,0)']
for i in range(z_coeff_to_fit):
    params_name.append('U(' + str(L[i]) + ',' + str(N[i]) + ')')

solu_coeff = np.insert(res_lsq.x, 4, 0)
intial_coeff_print = np.insert(initial_coeff, 4, 0)

table = Table({'Coefficient': params_name,
               'Fit': solu_coeff,
               'Initial guess' : intial_coeff_print},
                names=['Coefficient', 'Fit', 'Initial guess'])
print(table)

# Saving the array in current directory generated data
np.save('tests_solutions/sol_norm', solu_coeff)
np.save('tests_solutions/cost_norm', res_lsq.cost)
np.save('tests_solutions/residual_norm', res_lsq.fun)

print("--- %s mins ---" % str((time.time() - start_time) / 60))
