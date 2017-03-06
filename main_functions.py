import numpy as np
from math import factorial as f

# All mathematical function have been adapted for the Effelsber telescope


def illumination_pedestal(x, y, i_coeff):
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

    amp = i_coeff[0]
    c_db = i_coeff[1]  # number has to be negative [-8, -20]
    # Centre illuminationprimary reflector
    x0 = i_coeff[2]
    y0 = i_coeff[3]


    pr = 50  # Primary reflector radius
    # c_db = -20  # [dB] Constant for quadratic model illumination

    # Parabolic taper on a pedestal
    n = 2  # Order quadratic model illumination (Parabolic squared)

    c = 10 ** (c_db / 20.)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    illumination = amp * (c + (1. - c) * (1. - (r / pr) ** 2) ** n)
    return illumination


def illumination_gauss(x, y, i_coeff):

    amp = i_coeff[0]
    sigma_r = i_coeff[1]  # illumination taper
    # Centre illuminationprimary reflector
    x0 = i_coeff[2]
    y0 = i_coeff[3]

    pr = 50  # Primary reflector radius

    illumination = amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma_r * pr) ** 2))
    return illumination


def illumination_nikolic(x, y, i_coeff):

    amp = i_coeff[0]
    sigma_r = i_coeff[1]  # illumination taper
    # Centre illuminationprimary reflector
    x0 = i_coeff[2]
    y0 = i_coeff[3]

    pr = 50  # Primary reflector radius

    illumination = amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) * sigma_r / pr ** 2)
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
    a = 1  # half thickness support structure

    ant_model = np.zeros(x.shape)  # or y.shape same
    ant_model[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
    ant_model[(-L < x) & (x < L) & (-a < y) & (y < a)] = 0
    ant_model[(-L < y) & (y < L) & (-a < x) & (x < a)] = 0
    return ant_model


def delta(x, y, d_z):
    """
    Delta or phase change due to defocus function. Given by geometry of
    the telescope and defocus parameter. This function is specific for each
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

    # Gregorian (focused) telescope
    f1 = 30  # Focus primary reflector [m]
    F = 387.66  # Total focus Gregorian telescope [m]
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

    if n < 0:
        print('U(l, n) can only have positive values for n')

    m = abs(l)
    a = (n + m) // 2
    b = (n - m) // 2

    # if m == 0:
    #     delta_m0 = 1
    # else:
    #     delta_m0 = 0

    # norm = np.sqrt(2 * (n + 1) / (1 + delta_m0))

    R = sum(
    (-1) ** s * f(n - s) * rho ** (n - 2 * s) / (f(s) * f(a - s) * f(b - s))
    for s in range(0, b + 1)
    )

    if l < 0:
        U = R * np.sin(m * theta)  # * norm
    else:
        U = R * np.cos(m * theta)  # * norm

    return U


def phi(theta, rho, U_coeff, n):
    """
    Generates a series of Zernike polynomials, the aberration function.

    Parameters
    ----------
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).
    U_coeff : ndarray
        Constants organized by the ln list, which gives the possible values.
    n : int
        It is n >= 0. Determines the size of the polynomial, see ln.
    Returns
    -------
    phi : ndarray
        Zernile polynomail already evaluated and multiplied by its parameter
        or constant.
    """

    # List which contains the allowed values for the U function.
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    phi = sum(
        U_coeff[i] * U(L[i], N[i], theta, rho)
        for i in range(U_coeff.size)
        )

    return phi


def cart2pol(x, y):
    """
    Transformation for the cartesian coord. to polars. It is needed for the aperture function.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.

    Returns
    -------
    rho : ndarray
        Grid value for the radial variable, same as the contour plot.
    theta : ndarray
        Grid value for the angular variable, same as the contour plot.
    """

    rho = np.sqrt(x ** 2 + y ** 2)  # radius normalization
    theta = np.arctan2(y, x)
    return rho, theta


def aperture(x, y, U_coeff, d_z, n, i_coeff, illum):
    """
    Aperture function as expresses in the paper. A multiplication from other functions.
    """
    r, t = cart2pol(x, y)

    pr = 50  # Primary reflector radius

    # It needs to be normalized to be orthogonal undet the Zernike polynomials
    r_norm = r / pr

    _phi = phi(theta=t, rho=r_norm, U_coeff=U_coeff, n=n)
    _delta = delta(x, y, d_z=d_z)
    _shape = antenna_shape(x, y)

    if illum == 'gauss':
        _illum = illumination_gauss(x, y, i_coeff=i_coeff)
    if illum == 'pedestal':
        _illum = illumination_pedestal(x, y, i_coeff=i_coeff)
    if illum == 'nikolic':
        _illum = illumination_nikolic(x, y, i_coeff=i_coeff)

    E = _shape * _illum * np.exp((_phi + _delta) * 1j)
    # Aperture: E(x/lam, y/lam)

    return E


def wavevector_to_degree(x, lam):
    """
    Converst wave vector [1/m] to degrees.
    """
    return np.degrees(x * lam)



def angular_spectrum(x, y, U_coeff, d_z, n, i_coeff, illum):

    # the input (x,y) must be a linear array, not mesh!
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Normalization
    Nx, Ny = x.size, y.size

    # fft2 doesn't work without a grid
    x_grid, y_grid = np.meshgrid(x, y)

    _aperture = aperture(x_grid, y_grid, U_coeff=U_coeff, d_z=d_z, n=n, i_coeff=i_coeff, illum=illum)

    # FFT plus normalization
    F = np.fft.fft2(_aperture, norm='ortho') * 4 / np.sqrt(Nx * Ny)
    F_shift = np.fft.fftshift(F)

    u, v = np.fft.fftfreq(x.size, dx), np.fft.fftfreq(y.size, dy)
    u_shift, v_shift = np.fft.fftshift(u), np.fft.fftshift(v)

    return u_shift, v_shift, F_shift
