import matplotlib.pyplot as plt
import numpy as np
from main_functions import *
from scipy.constants import c as light_speed
import matplotlib
from matplotlib.mlab import griddata
from astropy.io import fits, ascii
from astropy.table import Table

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['lines.linewidth'] = 0.7
matplotlib.rcParams['image.cmap'] = 'viridis'
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['figure.titlesize'] = 15


def plot_beam(params_list, title, x, y, d_z, n, rad, lam, illum):

    if illum == 'gauss' or illum == 'nikolic':
        i_coeff = params_list[:4]
        U_coeff = params_list[4:]

    if illum == 'pedestal': # needs correction!
        amp = params_list[0]
        i_coeff = params_list[1]
        U_coeff = params_list[2:]

    if rad:
        angle_coeff = np.pi / 180
        uv_title = 'radians'
    else:
        angle_coeff = 1
        uv_title = 'degrees'

    u0, v0, as0 = angular_spectrum(
        x, y, U_coeff=U_coeff, d_z=d_z[0], n=n, i_coeff=i_coeff, illum=illum)
    u1, v1, as1 = angular_spectrum(
        x, y, U_coeff=U_coeff, d_z=d_z[1], n=n, i_coeff=i_coeff, illum=illum)
    u2, v2, as2 = angular_spectrum(
        x, y, U_coeff=U_coeff, d_z=d_z[2], n=n, i_coeff=i_coeff, illum=illum)

    aspectrum = np.array([as0, as1, as2])
    beam = np.abs(aspectrum) ** 2
    beam_norm = np.array([beam[i] / beam[i].max() for i in range(3)])

    u0_deg = wavevector_to_degree(u0, lam) * angle_coeff
    u1_deg = wavevector_to_degree(u1, lam) * angle_coeff
    u2_deg = wavevector_to_degree(u2, lam) * angle_coeff
    v0_deg = wavevector_to_degree(v0, lam) * angle_coeff
    v1_deg = wavevector_to_degree(v1, lam) * angle_coeff
    v2_deg = wavevector_to_degree(v2, lam) * angle_coeff

    extent0 = [u0_deg.min(), u0_deg.max(), v0_deg.min(), v0_deg.max()]
    extent1 = [u1_deg.min(), u1_deg.max(), v1_deg.min(), v1_deg.max()]
    extent2 = [u2_deg.min(), u2_deg.max(), v2_deg.min(), v2_deg.max()]

    u0_grid, v0_grid = np.meshgrid(u0_deg, v0_deg)
    u1_grid, v1_grid = np.meshgrid(u1_deg, v1_deg)
    u2_grid, v2_grid = np.meshgrid(u2_deg, v2_deg)

    fig, ax = plt.subplots(ncols=3, figsize=(14, 4))

    levels = 8  # number of black lines
    shrink = 0.9

    im0 = ax[0].imshow(beam_norm[0], extent=extent0)
    contour0 = ax[0].contour(u0_grid, v0_grid, beam_norm[0], levels)
    cb0 = fig.colorbar(im0, ax=ax[0], shrink=shrink)
    ax[0].set_title(
        'Beam $|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z[0], 2)) + '$')

    im1 = ax[1].imshow(beam_norm[1], extent=extent1)
    contour1 = ax[1].contour(u1_grid, v1_grid, beam_norm[1], levels)
    cb1 = fig.colorbar(im1, ax=ax[1], shrink=shrink)
    ax[1].set_title(
        'Beam $|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(d_z[1]) + '$')

    im2 = ax[2].imshow(beam_norm[2], extent=extent2)
    contour2 = ax[2].contour(u2_grid, v2_grid, beam_norm[2], levels)
    cb2 = fig.colorbar(im2, ax=ax[2], shrink=shrink)
    ax[2].set_title(
        'Beam $|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z[2], 2)) + '$')

    for _ax in ax:
        _ax.set_ylabel('$v$ (' + uv_title + ')')
        _ax.set_xlabel('$u$ (' + uv_title + ')')
        _ax.set_ylim(-0.05 * angle_coeff, 0.05 * angle_coeff)
        _ax.set_xlim(-0.05 * angle_coeff, 0.05 * angle_coeff)

    #fig.set_tight_layout(True)
    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.07, bottom=0.1, right=0.99,
        top=0.9, wspace=0.23, hspace=0.2
        )

    # for _cb in [cb0, cb1, cb2]:
    #     _cb.formatter.set_powerlimits((0, 0))
    #     _cb.update_ticks()

    return fig


# Test to plot the data
def plot_data(u_b_data, v_b_data, beam_data, title, d_z, rad):

    if rad:
        angle_coeff = 1
        uv_title = 'radians'
    else:
        angle_coeff = 180 / np.pi
        uv_title = 'degrees'

    fig, ax = plt.subplots(ncols=3, figsize=(14, 3.5))

    levels = 10
    shrink = 0.77

    xi_minus = np.linspace(u_b_data[0].min(), u_b_data[0].max(), 300) * angle_coeff
    yi_minus = np.linspace(v_b_data[0].min(), v_b_data[0].max(), 300) * angle_coeff
    zi_minus = griddata(u_b_data[0] * angle_coeff, v_b_data[0] * angle_coeff, beam_data[0], xi_minus, yi_minus, interp='linear')

    extent0 = [xi_minus.min(), xi_minus.max(), yi_minus.min(), yi_minus.max()]

    im0 = ax[0].imshow(zi_minus, extent=extent0)
    contours0 = ax[0].contour(xi_minus, yi_minus, zi_minus, levels)

    cb0 = fig.colorbar(im0, ax=ax[0], shrink=shrink)
    ax[0].set_title(
        'Beam $|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z[0], 2)) + '$')

    xi = np.linspace(u_b_data[1].min(), u_b_data[1].max(), 300) * angle_coeff
    yi = np.linspace(v_b_data[1].min(), v_b_data[1].max(), 300) * angle_coeff
    zi = griddata(u_b_data[1] * angle_coeff, v_b_data[1] * angle_coeff, beam_data[1], xi, yi, interp='linear')

    extent1 = [xi.min(), xi.max(), yi.min(), yi.max()]

    im1 = ax[1].imshow(zi, extent=extent1)
    contours1 = ax[1].contour(xi, yi, zi, levels)
    cb1 = fig.colorbar(im1, ax=ax[1], shrink=shrink)
    ax[1].set_title(
        'Beam $|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(d_z[1]) + '$')

    xi_plus = np.linspace(u_b_data[2].min(), u_b_data[2].max(), 300) * angle_coeff
    yi_plus = np.linspace(v_b_data[2].min(), v_b_data[2].max(), 300) * angle_coeff
    zi_plus = griddata(u_b_data[2] * angle_coeff, v_b_data[2] * angle_coeff, beam_data[2], xi_plus, yi_plus, interp='linear')

    extent2 = [xi_plus.min(), xi_plus.max(), yi_plus.min(), yi_plus.max()]

    im2 = ax[2].imshow(zi_plus, extent=extent2)
    contours2 = ax[2].contour(xi_plus, yi_plus, zi_plus, levels)
    cb2 = fig.colorbar(im2, ax=ax[2], shrink=shrink)
    ax[2].set_title(
        'Beam $|P(u,v)|^2_\mathrm{norm}$ $d_z=' + str(round(d_z[2], 2)) + '$')

    fig.suptitle(title)
    fig.subplots_adjust(
        left=0.06, bottom=0.15, right=1,
        top=0.91, wspace=0.13, hspace=0.2
        )

    for _ax in ax:
        _ax.set_ylabel('$v$ (' + uv_title + ')')
        _ax.set_xlabel('$u$ (' + uv_title + ')')

    # for _cb in [cb0, cb1, cb2]:
    #         _cb.formatter.set_powerlimits((0, 0))
    #         _cb.update_ticks()

    return fig


params_fitted = np.load('tests_solutions/sol_norm.npy')

# Calling generated data
data_gen = np.load('tests_solutions/data_gen.npy')
u_gen = np.load('tests_solutions/u_gen.npy')
v_gen = np.load('tests_solutions/v_gen.npy')

# Generating mesh and points for plot routines
box_size = 500
y_data = np.linspace(-box_size, box_size, 2 ** 10)
x_data = np.linspace(-box_size, box_size, 2 ** 10)

frequency = 32e9  # Hz
wavelength = light_speed / frequency

# d_z has to be given in units of wavelength
d_z = np.array([-2.5, 0., 2.5])  # Unit m/m (m/lamb)
d_z *= 2 * np.pi  # convert to radians

# Comparison between parameters
n_z_coeff = params_fitted.size - 4
n = int((np.sqrt(1 + 8 * n_z_coeff) - 3) / 2)

# Calling data from .fits file
path = '../test_data/3260_3C84_32deg_SB-002/'
# Extracting the Nikolic coefficients to plot beam
# Parameters found by Nikolic software example: 3260_3C84_32deg_SB-002
params_nikolic = fits.open(path + 'z' + str(n) +
    '/fitpars.fits')[1].data['ParValue']

ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
L = np.array(ln)[:, 0]
N = np.array(ln)[:, 1]

params_name = ['Illum amplitude', 'sigma_r', 'x_0', 'y_0']
for i in range(n_z_coeff):
    params_name.append('U(' + str(L[i]) + ',' + str(N[i]) + ')')

table = Table(
    {'Coefficient': params_name,
    'Nikolic': params_nikolic,
    'Fit': params_fitted},
    names=['Coefficient', 'Nikolic', 'Fit'])
print(table)

# Extracting real data
hdulist = fits.open(path + '3260_3C84_32deg_SB.fits')

focus = hdulist[1].data['fnu']
x_f = hdulist[1].data['DX']
y_f = hdulist[1].data['DY']

focus_plus = hdulist[2].data['fnu']
x_fplus = hdulist[2].data['DX']
y_fplus = hdulist[2].data['DY']

focus_minus = hdulist[3].data['fnu']
x_fminus = hdulist[3].data['DX']
y_fminus = hdulist[3].data['DY']

# Check this normalisation
beam_data = [focus_minus / focus_minus.max(), focus / focus.max(), focus_plus / focus_plus.max()]

u_b_data = [x_fminus, x_f, x_fplus]
v_b_data = [y_fminus, y_f, y_fplus]

fig0 = plot_data(
    u_b_data=u_b_data,
    v_b_data=v_b_data,
    beam_data=beam_data,
    title='Observed data',
    d_z=d_z,
    rad=False
    )

fig1 = plot_beam(
    params_list=params_nikolic,
    title='Nikolic fit', x=x_data,
    y=y_data,
    d_z=d_z,
    n=n,
    rad=False,
    lam=wavelength,
    illum='nikolic'
    )

fig2 = plot_beam(
    params_list=params_fitted,
    title='Data fit',
    x=x_data,
    y=y_data,
    d_z=d_z,
    n=n,
    rad=False,
    lam=wavelength,
    illum='gauss'
    )

plt.show()
