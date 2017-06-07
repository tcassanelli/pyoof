# Author: Tomas Cassanelli
import numpy as np
from math import factorial as f
from aux_functions import line_func

# All mathematical function have been adapted for the Effelsber telescope


def illumination_pedestal(x, y, I_coeff):
    """
    Illumination function parabolic taper on a pedestal, sometimes called
    amplitude. Represents the distribution of light in the primary reflector.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
    Returns
    -------
    illumination : ndarray
    """

    amp = I_coeff[0]
    c_dB = I_coeff[1]  # [dB] Illumination taper it is defined by the feedhorn
    # Number has to be negative, bounds given [-8, -25], see fit
    x0 = I_coeff[2]  # Centre illumination primary reflector
    y0 = I_coeff[3]

    pr = 50  # Primary reflector radius

    # Parabolic taper on a pedestal
    n = 2  # Order quadratic model illumination (Parabolic squared)

    c = 10 ** (c_dB / 20.)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    illumination = amp * (c + (1. - c) * (1. - (r / pr) ** 2) ** n)

    return illumination


def illumination_gauss(x, y, I_coeff):
    """
    Illumination function gaussian.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
    Returns
    -------
    illumination : ndarray
    """

    amp = I_coeff[0]
    sigma_r = I_coeff[1]  # illumination taper

    # Centre illuminationprimary reflector
    x0 = I_coeff[2]
    y0 = I_coeff[3]

    pr = 50  # Primary reflector radius

    illumination = (
        amp *
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma_r * pr) ** 2))
        )

    return illumination


def illumination_nikolic(x, y, I_coeff):
    """
    Illumination function used by Bojan Nikolic in his OOF software.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
    Returns
    -------
    illumination : ndarray
    """

    pr = 50  # Primary reflector radius
    amp = I_coeff[0]

    # illumination taper, different value from gauss illumination
    sigma_r = I_coeff[1]

    # Centre illuminationprimary reflector
    x0 = I_coeff[2]
    y0 = I_coeff[3]

    illumination = (
        amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) * sigma_r / pr ** 2)
        )

    return illumination


def ant_blockage(x, y):
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
    block : ndarray
    """

    pr = 50  # Primary reflector radius
    sr = 3.25  # secondary reflector radius
    L = 20   # length support structure (from the edge of the sr)
    a = 1  # half thickness support structure

    block = np.zeros(x.shape)  # or y.shape same
    block[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
    block[(-(sr + L) < x) & (x < (sr + L)) & (-a < y) & (y < a)] = 0
    block[(-(sr + L) < y) & (y < (sr + L)) & (-a < x) & (x < a)] = 0

    # testing new block
    # make diagram in thesis for these distances
    alpha = np.radians(10)  # angle of the triangle
    csc2 = np.sin(alpha) ** (-2)

    # base of the triangle
    d = (-a + np.sqrt(a ** 2 - (a ** 2 - pr ** 2) * csc2)) / csc2

    # points for the triangle coordinates
    A = sr + L
    B = a
    C = d / np.tan(alpha)
    D = a + d

    y1 = line_func((A, B), (C, D), x)
    y2 = line_func((A, -B), (C, -D), x)
    x3 = line_func((-A, B), (-C, D), y)
    x4 = line_func((-A, -B), (-C, -D), y)
    y5 = line_func((-A, -B), (-C, -D), x)
    y6 = line_func((-A, B), (-C, D), x)
    x7 = line_func((A, -B), (C, -D), y)
    x8 = line_func((A, B), (C, D), y)


    def circ(s):

        return np.sqrt(np.abs(pr ** 2 - s ** 2))


    block[(A < x) & (C > x) & (y1 > y) & (y2 < y)] = 0
    block[(pr > x) & (C < x) & (circ(x) > y) & (-circ(x) < y)] = 0

    block[(-A > y) & (-C < y) & (x4 < x) & (x3 > x)] = 0
    block[(-pr < y) & (-C > y) & (circ(y) > x) & (-circ(y) < x)] = 0

    block[(-A > x) & (-C < x) & (y5 < y) & (y6 > y)] = 0
    block[(-pr < x) & (-C > x) & (circ(x) > y) & (-circ(x) < y)] = 0

    block[(A < y) & (C > y) & (x7 < x) & (x8 > x)] = 0
    block[(pr > y) & (C < y) & (circ(x) > y) & (-circ(x) < y)] = 0

    return block


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
        Distance between the secondary and primary refelctor measured in rad.
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

    # d_z has to be in radians
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
        print('WARNING: U(l, n) can only have positive values for n')

    m = abs(l)
    a = (n + m) // 2
    b = (n - m) // 2

    R = sum(
        (-1) ** s * f(n - s) * rho ** (n - 2 * s) /
        (f(s) * f(a - s) * f(b - s))
        for s in range(0, b + 1)
        )

    if l < 0:
        U = R * np.sin(m * theta)
    else:
        U = R * np.cos(m * theta)

    return U


def phi(theta, rho, K_coeff):
    """
    Generates a series of Zernike polynomials, the aberration function.

    Parameters
    ----------
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).
    K_coeff : ndarray
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
    n = int((np.sqrt(1 + 8 * K_coeff.size) - 3) / 2)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    # Aperture phase distribution function in radians
    phi = sum(
        K_coeff[i] * U(L[i], N[i], theta, rho)
        for i in range(K_coeff.size)
        ) * 2 * np.pi

    return phi


def cart2pol(x, y):
    """
    Transformation for the cartesian coord. to polars. It is needed for the
    aperture function.

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


def aperture(x, y, K_coeff, d_z, I_coeff, illum):
    """
    Aperture function. Multiplication between the antenna truncation, the
    illumination function and the aberration.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.
    K_coeff : ndarray
        Phase coefficients in increasing order.
    d_z : float
        Distance between the secondary and primary refelctor measured in rad.
    I_coeff : ndarray
        Illumination coefficients for pedestal function.
    illum : str
        Illumination function type, gauss, pedestal or nikolic.
    Returns
    -------
    E : ndarray
        Grid value that contains general expression for aperture.
    """

    r, t = cart2pol(x, y)

    pr = 50  # Primary reflector radius

    # It needs to be normalized to be orthogonal undet the Zernike polynomials
    r_norm = r / pr

    _phi = phi(theta=t, rho=r_norm, K_coeff=K_coeff)
    _delta = delta(x, y, d_z=d_z)
    _shape = ant_blockage(x, y)

    # Wavefront aberration distribution (rad)
    wavefront = (_phi + _delta)

    if illum == 'gauss':
        _illum = illumination_gauss(x, y, I_coeff=I_coeff)
    if illum == 'pedestal':
        _illum = illumination_pedestal(x, y, I_coeff=I_coeff)
    if illum == 'nikolic':
        _illum = illumination_nikolic(x, y, I_coeff=I_coeff)

    E = _shape * _illum * np.exp(wavefront * 1j)
    # Aperture: E(x / wavel, y / wavel)

    return E


def wavevector_to_degree(x, wavel):
    """
    Converst wave-vector 1/m to degrees.

    Parameters
    ----------
    x : ndarray
        Wave-vector, result from FFT2.
    wavel : ndarray
        Wavelength.
    Returns
    -------
    wave_vector_degrees : ndarray
        Wave-vector in degrees.
    """

    wave_vector_degrees = np.degrees(x * wavel)

    return wave_vector_degrees


def angular_spectrum(K_coeff, I_coeff, d_z, illum):

    # Arrays to generate angular spectrum model
    box_size = 500
    x = np.linspace(-box_size, box_size, 2 ** 10)
    y = np.linspace(-box_size, box_size, 2 ** 10)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Normalization
    # Nx, Ny = x.size, y.size

    # fft2 doesn't work without a grid
    x_grid, y_grid = np.meshgrid(x, y)

    _aperture = aperture(
        x=x_grid,
        y=y_grid,
        K_coeff=K_coeff,
        d_z=d_z,
        I_coeff=I_coeff,
        illum=illum
        )

    # FFT, normalisation not needed, comparing normalised beam
    F = np.fft.fft2(_aperture)  # * 4 / np.sqrt(Nx * Ny) # Normalisation
    F_shift = np.fft.fftshift(F)

    u, v = np.fft.fftfreq(x.size, dx), np.fft.fftfreq(y.size, dy)
    u_shift, v_shift = np.fft.fftshift(u), np.fft.fftshift(v)

    return u_shift, v_shift, F_shift


def sr_phase(params, notilt):
    # subreflector phase
    K_coeff = params[4:]

    if notilt:
        K_coeff[1] = 0  # For value K(-1, 1) = 0
        K_coeff[2] = 0  # For value K(1, 1) = 0

    pr = 50
    x = np.linspace(-pr, pr, 1e3)
    y = np.linspace(-pr, pr, 1e3)

    x_grid, y_grid = np.meshgrid(x, y)

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / pr

    sr_phi = phi(theta=t, rho=r_norm, K_coeff=K_coeff)

    return sr_phi


def par_variance(res, jac, n_pars):
    # Covariance and correlation matrices
    m = res.size
    d_free = m - n_pars  # degrees of freedom

    # Covarince matrix
    cov = np.dot(res.T, res) / d_free * np.linalg.inv(np.dot(jac.T, jac))

    sigmas2 = np.diag(np.diag(cov))  # sigma ** 2
    D = np.linalg.inv(np.sqrt(sigmas2))  # inv diagonal variance matrix

    # Correlation matrix
    corr = np.dot(np.dot(D, cov), D)

    return cov, corr


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Testing the blockage
    x = np.linspace(-60, 60, 1e3)
    y = x
    xg, yg = np.meshgrid(x, y)

    blockage = ant_blockage(xg, yg)

    plt.imshow(blockage, origin='lower')

    plt.show()

