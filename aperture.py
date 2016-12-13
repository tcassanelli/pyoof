import numpy as np
import matplotlib.pyplot as plt
from math import factorial as f

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Aperture function for the Effelsber telescope


def illum(x, y, c_db=-20):
    """
    Illumination function, sometimes called amplitude. Represents the
    distribution of light in the primary reflector.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.

    Returns
    -------
    illumination : ndarray
    """
    pr = 50  # Primary reflector radius
    #c_db = -20  # [dB] Constant for quadratic model illumination
    n = 2  # Order quadratic model illumination

    c = 10 ** (c_db / 20.)
    r = np.sqrt(x ** 2 + y ** 2)
    illumination = c + (1 - c) * (1. - (r / pr) ** 2) ** n
    return illumination


def antenna_shape(x, y):
    """
    Truncation in the aperture function, given by the hole generated for the
    secondary reflector and the supporting structure.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.

    Returns
    -------
    ant_model : ndarray
    """

    pr = 50  # Primary reflector radius
    sr = 3.25  # secondary reflector radius
    L = 20  # length support structure
    a = 2  # thickness support structure

    ant_model = np.zeros(x.shape)  # or y.shape same
    ant_model[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
    ant_model[(-L < x) & (x < L) & (-a < y) & (y < a)] = 0
    ant_model[(-L < y) & (y < L) & (-a < x) & (x < a)] = 0
    return ant_model


def delta(x, y, d_z):
    """
    Delta or phase change due to defocus function. Given by geometry of
    thetelescope and defocus parameter. This function is specific for each
    telescope (Check this function in the future!).

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.
    d_z : float
        Distance between the secondary and primary refelctor.
    Returns
    -------
    delta : ndarray
    """

    # Gregorian (Effelsberg)
    f1 = 30  # m
    # f2 = 385  # m
    # s = - 32  # Distance between primary and secondary reflector (check!)
    # (negative for convention)

    # F = f1 * f2 / (f1 - f2 - s)  # Total focus Gregorian telescope
    F = 387.66  # m
    r = np.sqrt(x ** 2 + y ** 2)  # polar coord. radius
    a = r / (2 * f1)
    b = r / (2 * F)
    delta = d_z * ((1 - a ** 2) / (1 + a ** 2) + (1 - b ** 2) / (1 + b ** 2))
    return delta


def U(l, n, theta, rho):
    """
    Zernike polynomial generator. l, n are intergers, n >= 0 and n - |l| even.
    Expansion of a complete set of orthonormal polynomials in a unitary circle,
    for the aberration function.
    The n value determines the total amount of polynomials 0.5(n+1)(n+2).

    Parameters
    ----------
    l : int
        Can be positive or negative, relative to angle component.
    n : int
        It is n >= 0. Relative to radial component.
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).

    Returns
    -------
    U : ndarray
        Zernile polynomail already evaluated.
    """

    m = abs(l)
    a = int((n + m) / 2)
    b = int((n - m) / 2)

    R = sum(
    (-1) ** s * f(n - s) * rho ** (n - 2 * s) / (f(s) * f(a - s) * f(b - s))
    for s in range(0, b + 1)
    )

    if l < 0:
        U = R * np.sin(m * theta)
    else:
        U = R * np.cos(m * theta)

    return U


def phi(theta, rho, params, n=6):
    """
    Generates a series of Zernike polynomials, the aberration function.

    Parameters
    ----------
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).
    params : ndarray
        Constants organized by the ln list, which gives the possible values.
    n : int
        It is n >= 0. Determines the size of the polynomial, see ln.
    Returns
    -------
    polyU : ndarray
        Zernile polynomail already evaluated and multiplied by its parameter
        or constant.
    """
    # List which contains the allowed values for the U function.
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    phi = sum(
        params[i] * U(L[i], N[i], theta, rho)
        for i in range(params.size)
        )

    return phi


def angular_spectrum(x, y, apert, params, d_z):
    """
    Angular spectrum function or aperture function FFT.

    Parameters
    ----------
    x : ndarray
        Values for variable x in 1-dimension.
    y : ndarray
        Values for variable x in 1-dimension.
    apert : callable
        Aperture function to be transformed such as apert(x_grid, y_grid), not
        more.
        inputs that this.
    Returns
    -------
    x_grid : ndarray
        Grid value for the x variable, same as the contour plot.
    y_grid : ndarray
        Grid value for the x variable, same as the contour plot.
    u_grid : ndarray
        Discrete Fourier Transform sample frequencies respect to the x input.
    v_grid : ndarray
        Discrete Fourier Transform sample frequencies respect to the y input.
    F_shift : ndarray
        Compute the two-dimensional discrete Fourier Transform and then shift
        the zero-frequency component to the center of the spectrum.
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Grid values for mesh plot
    x_grid, y_grid = np.meshgrid(x, y)

    # Normalization
    Nx = len(x)
    Ny = len(y)

    aperture = apert(x_grid, y_grid, params, d_z)

    # FFT plus normalization
    F = np.fft.fft2(aperture, norm='ortho') * 4 / np.sqrt(Nx * Ny)
    F_shift = np.fft.fftshift(F)

    u, v = np.meshgrid(np.fft.fftfreq(x.size, dx), np.fft.fftfreq(y.size, dy))
    u_grid, v_grid = np.fft.fftshift(u), np.fft.fftshift(v)

    return u_grid, v_grid, F_shift


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)  # radius normalization
    theta = np.arctan2(y, x)
    return (rho, theta)


def aperture(x, y, params, d_z, n=5):
    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / 50

    _phi = phi(t, r_norm, params, n)
    _delta = delta(x_grid, y_grid, d_z)
    _sum = _phi + _delta
    _shape = antenna_shape(x_grid, y_grid)
    _illum = illum(x_grid, y_grid, c_db=-20)

    N = x.shape[0]

    A = _shape * _illum * np.exp((_phi + _delta) * 1j)  # complex correction

    return A





# Test Zernike polynomials function
if False:
    # Generating data to plot
    box_size = 50
    y_data = np.linspace(-box_size, box_size, 2 ** 10)
    x_data = np.linspace(-box_size, box_size, 2 ** 10)
    x_grid, y_grid = np.meshgrid(x_data, y_data)

    # Zernike Pyramid
    def circle(x, y, R=box_size):
        circ = np.zeros(x.shape)  # or y.shape
        circ[x ** 2 + y ** 2 <= R ** 2] = 1
        return circ

    circ = circle(x_grid, y_grid)
    r, t = cart2pol(x_grid, y_grid)  # Chaning values for U and phi
    # if not change weird things happen to the plots (inversion at half)

    r_norm = r / 50  # Normalization R(m, n)(r=1) = 1

    n_max = 6  # Max number n for the Zernike polynomials
    ln = [(j, i) for i in range(0, n_max + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]
    n_plots = len(L)

    fig, axes = plt.subplots(figsize=(14, 7), nrows=4, ncols=7)
    ax = axes.flat
    lev = np.linspace(0, 10, 0.01)

    for i in range(n_plots):
        _U = U(L[i], N[i], t, r_norm) * circ
        im = ax[i].imshow(_U, cmap='viridis', vmin=-1, vmax=1, origin='lower')
        contours = ax[i].contour(_U, 3, colors='black')
        ax[i].set_title('$U^{' + str(L[i]) + '}_{' + str(N[i]) + '}$')
        ax[i].set_aspect('equal')
        ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        ax[i].yaxis.set_major_formatter(plt.NullFormatter())
        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')

    plt.suptitle('Zernike Polynomials $U^m_n$', fontsize=20)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cb.ax.set_ylabel('Normalized amplitude $R^m_n(1)=1$', fontsize=13)

    plt.show()
    plt.close()


# Test Illumination function
if False:

    # Generating data to plot
    box_size = 50
    y_data = np.linspace(-box_size, box_size, 2 ** 10)
    x_data = np.linspace(-box_size, box_size, 2 ** 10)

    x_grid, y_grid = np.meshgrid(x_data, y_data)

    _I = illum(x_grid, y_grid) * antenna_shape(x_grid, y_grid)

    extent = [-box_size, box_size, -box_size, box_size]
    fig, ax = plt.subplots()
    cs = ax.imshow(_I, extent=extent, vmin=0, vmax=1, cmap='viridis', origin='lower')
    contours = ax.contour(x_grid, y_grid, _I, 3)
    cb = fig.colorbar(cs, ax=ax)
    ax.set_ylabel('Position y')
    ax.set_xlabel('Position x')
    ax.set_title('Ilumination function $I(x,y)$')
    cb.ax.set_ylabel('$I(x,y)$ Amplitude', fontsize=13)
    plt.show()

# Test phi and delta functions
if True:

    # Generating data to plot
    box_size = 50
    y_data = np.linspace(-box_size, box_size, 2 ** 10)
    x_data = np.linspace(-box_size, box_size, 2 ** 10)

    x_grid, y_grid = np.meshgrid(x_data, y_data)

    def circle(x, y, R=box_size):
        circ = np.zeros(x.shape)  # or y.shape
        circ[x ** 2 + y ** 2 <= R ** 2] = 1
        return circ

    params_ = np.random.randn(10)  # rand number from 0-1 for 28 coeff.
    params = np.array([0, -2.298295e-1, 4.93943e-1, -1.379757e-1, -1.819459e-1, -9.78374e-2, 6.137946e-1, -1.684147e-1, 1.348733e-1, -2.600830e-1, 3.05227e-2, -1.045454e-1, 2.149645e-2, 6.240493e-2, -9.050347e-2, -5.502480e-1, 2.346242e-1, -3.50973e-1, -1.287273e-2, 3.709351e-1, -2.244371e-1])

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / 50
    d_z = 25e-3

    _phi = phi(t, r_norm, params=params, n=5) * antenna_shape(x_grid, y_grid)

    _delta = delta(x_grid, y_grid, d_z=d_z) * antenna_shape(x_grid, y_grid)

    _sum = _phi + _delta

    print('# Coefficients: ', len(params))
    print('Coefficients: ', params)
    print('Max _phi: ', _phi.max())
    print('Max _delta: ', _delta.max())
    print('Max _sum: ', _sum.max())

    extent = [-box_size, box_size, -box_size, box_size]
    fig, ax = plt.subplots(ncols=3, figsize=(14, 4))

    im0 = ax[0].imshow(_phi, extent=extent, cmap='viridis', origin='lower')
    contours0 = ax[0].contour(x_grid, y_grid, _phi, 10, colors='black')
    cb0 = fig.colorbar(im0, ax=ax[0], shrink=0.8)
    ax[0].set_ylabel('Position y')
    ax[0].set_xlabel('Position x')
    ax[0].set_title('$\phi(x,y)$')
    cb0.ax.set_ylabel('$\phi(x,y)$ Amplitude', fontsize=13)

    im1 = ax[1].imshow(_delta, extent=extent, cmap='viridis', origin='lower')
    contours1 = ax[1].contour(x_grid, y_grid, _delta, 10, colors='black')
    cb1 = fig.colorbar(im1, ax=ax[1], shrink=0.8)
    ax[1].set_ylabel('Position y')
    ax[1].set_xlabel('Position x')
    ax[1].set_title('$\delta(x,y,d_z =' + str(d_z) + ')$')
    cb1.ax.set_ylabel('$\delta(x,y,d_z =' + str(d_z) + ')$ Amplitude', fontsize=10)

    im2 = ax[2].imshow(_sum, extent=extent, cmap='viridis', origin='lower')
    contours2 = ax[2].contour(x_grid, y_grid, _sum, 10, colors='black')
    cb2 = fig.colorbar(im2, ax=ax[2], shrink=0.8)
    ax[2].set_ylabel('Position y')
    ax[2].set_xlabel('Position x')
    ax[2].set_title('$\phi(x,y)+\delta(x,y,d_z =' + str(d_z) + ')$')
    cb2.ax.set_ylabel('$\phi(x,y)+\delta(x,y,d_z =' + str(d_z) + ')$ Amplitude', fontsize=10)

    fig.set_tight_layout(True)

    # fig.savefig('test_aperture.pdf')
    # os.system('open test_aperture.pdf')


if True:
    # Generating data to plot
    box_size = 50
    y_data = np.linspace(-box_size, box_size, 2 ** 10)
    x_data = np.linspace(-box_size, box_size, 2 ** 10)

    x_grid, y_grid = np.meshgrid(x_data, y_data)

    # Set of example parameters
    params1 = np.zeros(21)
    params1[1] = 1

    params2 = np.random.randn(21)  # rand number from 0-1 for 28 coeff.

    params3 = np.array([0, -2.298295e-1, 4.93943e-1, -1.379757e-1, -1.819459e-1, -9.78374e-2, 6.137946e-1, -1.684147e-1, 1.348733e-1, -2.600830e-1, 3.05227e-2, -1.045454e-1, 2.149645e-2, 6.240493e-2, -9.050347e-2, -5.502480e-1, 2.346242e-1, -3.50973e-1, -1.287273e-2, 3.709351e-1, -2.244371e-1])

    d_z = [-25e-3, 0, 25e-3]

    apert = [aperture(x_grid, y_grid, params3, d_z=d_z[0]), aperture(x_grid, y_grid, params3, d_z=d_z[1]), aperture(x_grid, y_grid, params3, d_z=d_z[2])]

    extent = [-box_size, box_size, -box_size, box_size]

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(14, 8))

    im0 = ax[0, 0].imshow(apert[0].real, extent=extent, cmap='viridis', origin='lower')
    contours0 = ax[0, 0].contour(x_grid, y_grid, apert[0].real, 10, colors='black')
    cb0 = fig.colorbar(im0, ax=ax[0, 0], shrink=0.8)
    ax[0, 0].set_ylabel('Position y')
    ax[0, 0].set_xlabel('Position x')
    ax[0, 0].set_title('Aperture $\operatorname{Re}(A(x,y))$ $d_z=' + str(d_z[0]) + '$')
    cb0.ax.set_ylabel('$\operatorname{Re}(A(x,y))$ Amplitude', fontsize=13)

    im1 = ax[0, 1].imshow(apert[1].real, extent=extent, cmap='viridis', origin='lower')
    contours1 = ax[0, 1].contour(x_grid, y_grid, apert[1].real, 10, colors='black')
    cb1 = fig.colorbar(im1, ax=ax[0, 1], shrink=0.8)
    ax[0, 1].set_ylabel('Position y')
    ax[0, 1].set_xlabel('Position x')
    ax[0, 1].set_title('Aperture $\operatorname{Re}(A(x,y))$ $d_z=' + str(d_z[1]) + '$')
    cb1.ax.set_ylabel('$\operatorname{Re}(A(x,y))$ Amplitude', fontsize=13)

    im2 = ax[0, 2].imshow(apert[2].real, extent=extent, cmap='viridis', origin='lower')
    contours2 = ax[0, 2].contour(x_grid, y_grid, apert[2].real, 10, colors='black')
    cb2 = fig.colorbar(im2, ax=ax[0, 2], shrink=0.8)
    ax[0, 2].set_ylabel('Position y')
    ax[0, 2].set_xlabel('Position x')
    ax[0, 2].set_title('Aperture $\operatorname{Re}(A(x,y))$ $d_z=' + str(d_z[2]) + '$')
    cb2.ax.set_ylabel('$\operatorname{Re}(A(x,y))$ Amplitude', fontsize=13)

    im3 = ax[1, 0].imshow(apert[0].imag, extent=extent, cmap='viridis', origin='lower')
    contours3 = ax[1, 0].contour(x_grid, y_grid, apert[0].imag, 10, colors='black')
    cb3 = fig.colorbar(im0, ax=ax[1, 0], shrink=0.8)
    ax[1, 0].set_ylabel('Position y')
    ax[1, 0].set_xlabel('Position x')
    ax[1, 0].set_title('Aperture $\operatorname{Im}(A(x,y))$ $d_z=' + str(d_z[0]) + '$')
    cb3.ax.set_ylabel('$\operatorname{Im}(A(x,y))$ Amplitude', fontsize=13)

    im4 = ax[1, 1].imshow(apert[1].imag, extent=extent, cmap='viridis', origin='lower')
    contours4 = ax[1, 1].contour(x_grid, y_grid, apert[1].imag, 10, colors='black')
    cb4 = fig.colorbar(im4, ax=ax[1, 1], shrink=0.8)
    ax[1, 1].set_ylabel('Position y')
    ax[1, 1].set_xlabel('Position x')
    ax[1, 1].set_title('Aperture $\operatorname{Im}(A(x,y))$ $d_z=' + str(d_z[1]) + '$')
    cb4.ax.set_ylabel('$\operatorname{Im}(A(x,y))$ Amplitude', fontsize=13)

    im5 = ax[1, 2].imshow(apert[2].imag, extent=extent, cmap='viridis', origin='lower')
    contours5 = ax[1, 2].contour(x_grid, y_grid, apert[2].imag, 10, colors='black')
    cb5 = fig.colorbar(im5, ax=ax[1, 2], shrink=0.8)
    ax[1, 2].set_ylabel('Position y')
    ax[1, 2].set_xlabel('Position x')
    ax[1, 2].set_title('Aperture $\operatorname{Im}(A(x,y))$ $d_z=' + str(d_z[2]) + '$')
    cb5.ax.set_ylabel('$\operatorname{Im}(A(x,y))$ Amplitude', fontsize=13)

    fig.set_tight_layout(True)


if True:
        # Generating data to plot
    box_size = 50
    y_data = np.linspace(-box_size, box_size, 2 ** 10)
    x_data = np.linspace(-box_size, box_size, 2 ** 10)

    x_grid, y_grid = np.meshgrid(x_data, y_data)

    # Set of example parameters
    params1 = np.zeros(21)
    params1[1] = 1

    params2 = np.random.randn(21)  # rand number from 0-1 for 28 coeff.

    params3 = np.array([0, -2.298295e-1, 4.93943e-1, -1.379757e-1, -1.819459e-1, -9.78374e-2, 6.137946e-1, -1.684147e-1, 1.348733e-1, -2.600830e-1, 3.05227e-2, -1.045454e-1, 2.149645e-2, 6.240493e-2, -9.050347e-2, -5.502480e-1, 2.346242e-1, -3.50973e-1, -1.287273e-2, 3.709351e-1, -2.244371e-1])

    d_z = [-25e-3, 0, 25e-3]

    apert = [aperture(x_grid, y_grid, params3, d_z=d_z[0]), aperture(x_grid, y_grid, params3, d_z=d_z[1]), aperture(x_grid, y_grid, params3, d_z=d_z[2])]

    extent = [-box_size, box_size, -box_size, box_size]

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(14, 4))


    plt.show()
