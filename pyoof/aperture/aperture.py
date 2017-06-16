#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from ..math_functions import cart2pol
from ..zernike import U

# All mathematical function have been adapted for the Effelsberg telescope
__all__ = [
    'illumination_pedestal', 'illumination_gauss', 'delta', 'phi',
    'phase', 'aperture', 'angular_spectrum'
    ]


def illumination_pedestal(x, y, I_coeff, pr, order=2):
    """
    Illumination function, parabolic taper on a pedestal, sometimes called
    amplitude, apodization, taper or window function. Represents the
    distribution of light in the primary reflector. The illumination reduces
    the sidelobes in the FT.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, c_dB, x0, y0]
    pr : int
        Radius from the primary reflector.
    order : int
        Order of the parabolic taper on a pedestal, it is commonly set at 2.

    Returns
    -------
    illumination : ndarray
        Illumination distribution
    """

    i_amp = I_coeff[0]  # amplitude of the illumination distribution
    c_dB = I_coeff[1]  # [dB] Illumination taper it is defined by the feedhorn
    # Number has to be negative, bounds given [-8, -25], see fit
    x0 = I_coeff[2]  # Centre illumination primary reflector
    y0 = I_coeff[3]

    # Parabolic taper on a pedestal
    n = order  # Order quadratic model illumination (Parabolic squared)

    c = 10 ** (c_dB / 20.)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    illumination = i_amp * (c + (1. - c) * (1. - (r / pr) ** 2) ** n)

    return illumination


def illumination_gauss(x, y, I_coeff, pr):
    """
    Illumination function, Gaussian distribution, sometimes called
    amplitude, apodization, taper or window function. Represents the
    distribution of light in the primary reflector. The illumination reduces
    the sidelobes in the FT.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, sigma_dB, x0, y0]
    pr : int
        Radius from the primary reflector.

    Returns
    -------
    illumination : ndarray
        Illumination distribution
    """

    i_amp = I_coeff[0]  # amplitude of the illumination distribution
    sigma_dB = I_coeff[1]  # illumination taper, sigma_dB
    sigma = 10 ** (sigma_dB / 20)  # -15 to -20 dB
    x0 = I_coeff[2]  # Centre illumination primary reflector
    y0 = I_coeff[3]

    illumination = (
        i_amp *
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma * pr) ** 2))
        )

    return illumination


def delta(x, y, d_z):
    """
    Delta or phase change due to defocus function. Given by geometry of
    the telescope and defocus parameter. For Cassegrain/Gregorain geometries.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.
    d_z : float
        Distance between the secondary and primary refelctor measured in rad.
        It is the characteristic measurement to give an offset and an
        out-of-focus image at the end.

    Returns
    -------
    delta : ndarray
    """

    # Cassegrain/Gregorian (at focus) telescope
    f1 = 30  # Focus primary reflector m
    F = 387.66  # Total focus Cassegrain/Gregorian telescope m
    r = np.sqrt(x ** 2 + y ** 2)  # polar coordinates radius
    a = r / (2 * f1)
    b = r / (2 * F)

    # d_z has to be in radians
    delta = d_z * ((1 - a ** 2) / (1 + a ** 2) + (1 - b ** 2) / (1 + b ** 2))

    return delta


def phi(theta, rho, K_coeff):
    """
    Computes the wavefront (aberration) function which coincides with the
    aperture phase distribution in optics theory. The wavefront is an
    approximation from the Zernike
    circle polynomials and their coefficients, K(l, n).

    Parameters
    ----------
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2)
        normalised.
    K_coeff : ndarray
        Constants coefficients for each of them there is only one Zernike
        circle polynomial.
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


def phase(K_coeff, notilt, pr):
    """
    Aperture phase distribution (or wavefront aberration function), for an
    specific telescope primary reflector. In general the tilt (in optics,
    deviation in the direction a beam of light propagates) is substracted from
    its calculation. Function used to show the final results from the fit
    procedure.

    Parameters
    ----------
    K_coeff : ndarray
        Constants coefficients for each of them there is only one Zernike
        circle polynomial.
    notilt : bool
        True or False boolean to include or exclude the tilt coefficients in
        the aperture phase distribution. The Zernike circle polynomials are
        related to tilt through U(l=-1, n=1) and U(l=1, n=1).
    pr : float
        Primary reflector radius.

    Returns
    -------
    phase : ndarray
        Aperture phase ditribution for an specific primary radius.
    """

    # Necessary to copy the array, otherwise the original value will be
    # forgotten, it is important for the fit procedure to keep it
    _K_coeff = K_coeff.copy()

    if notilt:
        _K_coeff[1] = 0  # For value K(-1, 1) = 0
        _K_coeff[2] = 0  # For value K(1, 1) = 0

    # Default resolution for the phase map
    x = np.linspace(-pr, pr, 1e3)
    y = np.linspace(-pr, pr, 1e3)

    x_grid, y_grid = np.meshgrid(x, y)

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / pr

    _phase = phi(theta=t, rho=r_norm, K_coeff=_K_coeff)
    _phase[(x_grid ** 2 + y_grid ** 2 > pr ** 2)] = 0

    return _phase


def aperture(x, y, K_coeff, d_z, I_coeff, illum_func, telgeo):
    """
    Aperture distribution function. Collection of individual functions,
    illumination, telescope geometry, phi and delta.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.
    K_coeff : ndarray
        Constants coefficients for each of them there is only one Zernike
        circle polynomial.
    d_z : float
        Distance between the secondary and primary refelctor measured in rad.
        It is the characteristic measurement to give an offset and an
        out-of-focus image at the end.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, sigma_dB, x0, y0]
    illum_func : function
        Illumination function with parameters (x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage function and the primary radius.
        telego = [function, int].
    Returns
    -------
    E : ndarray
        Grid value that contains general expression for aperture distribution.
    """

    r, t = cart2pol(x, y)

    [blockage, pr] = telgeo
    _blockage = blockage(x=x, y=y)

    # Normalisation to be used in the Zernike circle polynomials
    r_norm = r / pr

    _phi = phi(theta=t, rho=r_norm, K_coeff=K_coeff)
    _delta = delta(x=x, y=y, d_z=d_z)

    # Wavefront aberration distribution (rad)
    wavefront = (_phi + _delta)

    # Selection of the illumination function
    _illumination = illum_func(x=x, y=y, I_coeff=I_coeff, pr=pr)

    E = _blockage * _illumination * np.exp(wavefront * 1j)

    return E


def angular_spectrum(K_coeff, I_coeff, d_z, illum_func, telgeo, resolution):
    """
    Angular spectrum or (field) radiation pattern, it is the FFT2 computation
    of the aperture distribution in an rectangular grid. Passing the mayority
    of arguments to the aperture function except the resolution, which is the
    FFT2 resolution.

    Parameters
    ----------
    K_coeff : ndarray
        Constants coefficients for each of them there is only one Zernike
        circle polynomial.
    d_z : float
        Distance between the secondary and primary refelctor measured in rad.
        It is the characteristic measurement to give an offset and an
        out-of-focus image at the end.
    I_coeff : ndarray
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, sigma_dB, x0, y0]
    illum_func : function
        Illumination function with parameters (x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage function and the primary radius.
        telego = [function, int].
    resolution : int
        Fast Fourier Transform resolution for a rectancular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.

    Returns
    -------
    u_shift : ndarray
        u wave-vector in 1/m units. It belongs to the x coordinate in m from
        the aperture distribution.
    v_shift : ndarray
        v wave-vector in 1/m units. It belongs to the y coordinate in m from
        the aperture distribution.
    F_shift : ndarray
        Output from the FFT2 pack, unnormalized solution in a grid same as
        aperture input computed from a given resolution.
    """

    # Arrays to generate angular spectrum model
    box_size = 500  # Effelsberg pr=50, 1/5 of the length

    # default resolution 2 ** 10
    x = np.linspace(-box_size, box_size, resolution)
    y = x

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
        illum_func=illum_func,
        telgeo=telgeo
        )

    # FFT, normalisation not needed, comparing normalised beam
    F = np.fft.fft2(_aperture)  # * 4 / np.sqrt(Nx * Ny) # Normalisation
    F_shift = np.fft.fftshift(F)

    u, v = np.fft.fftfreq(x.size, dx), np.fft.fftfreq(y.size, dy)
    u_shift, v_shift = np.fft.fftshift(u), np.fft.fftshift(v)

    return u_shift, v_shift, F_shift
