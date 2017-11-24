# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from ..math_functions import cart2pol, rms
from ..zernike import U

__all__ = [
    'illum_pedestal', 'illum_gauss', 'wavefront', 'phase', 'aperture',
    'radiation_pattern', 'e_rse'
    ]


def e_rse(phase):
    '''
    Computes the random-surface-error efficiency using the Ruze's equation.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        The phase error is a two dimensional array (one of the solutions from
        the pyoof package). Its amplitude values are in radians.
    '''

    rms_rad = rms(phase)  # rms value in radians

    return np.exp(-rms_rad ** 2)


def illum_pedestal(x, y, I_coeff, pr, q=2):
    '''
    Illumination function, parabolic taper on a pedestal, sometimes called
    apodization, taper or window function. Represents the distribution of
    light in the primary reflector. The illumination reduces the side lobes in
    the FT and it is a property of the receiver.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Grid value for the x variable in meters.
    y : `~numpy.ndarray`
        Grid value for the y variable in meters.
    I_coeff : `~numpy.ndarray`
        List which contains 4 parameters, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, c_dB, x0, y0]
    pr : float
        Primary reflector radius in meters.
    q : int
        Order of the parabolic taper on a pedestal, it is commonly set at 2.

    Returns
    -------
    Ea : `~numpy.ndarray`
        Illumination function.
    '''

    i_amp, c_dB, x0, y0 = I_coeff

    c = 10 ** (c_dB / 20.)  # c_dB has to be negative, bounds given [-8, -25]
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Parabolic taper on a pedestal
    Ea = i_amp * (c + (1. - c) * (1. - (r / pr) ** 2) ** q)

    return Ea


def illum_gauss(x, y, I_coeff, pr):
    '''
    Illumination function, Gaussian, sometimes called apodization, taper or
    window function. Represents the distribution of light in the primary
    reflector. The illumination reduces the side lobes in the FT and
    it is a property of the receiver.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Grid value for the x variable in meters.
    y : `~numpy.ndarray`
        Grid value for the y variable in meters.
    I_coeff : `~numpy.ndarray`
        List which contains 4 coefficients, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, sigma_dB, x0, y0].
    pr : float
        Primary reflector radius in meters.

    Returns
    -------
    Ea : `~numpy.ndarray`
        Illumination function.
    '''
    i_amp, sigma_dB, x0, y0 = I_coeff
    sigma = 10 ** (sigma_dB / 20)  # -15 to -20 dB
    norm = np.sqrt(2 * np.pi * sigma ** 2)  # normalization Gaussian
    Ea = (
        i_amp * norm *
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma * pr) ** 2))
        )

    return Ea


def wavefront(rho, theta, K_coeff):
    '''
    Computes the wavefront (aberration) distribution. It tells how is the
    error distributed along the primary dish, it is related to the phase error.
    The wavefront (aberration) distribution is described as a parametrization
    of the Zernike circle polynomials multiplied by a set of coefficients.

    Parameters
    ----------
    rho : `~numpy.ndarray`
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2)
        normalized by its maximum radius.
    theta : `~numpy.ndarray`
        Values for the angular component. theta = np.arctan(y / x).
    K_coeff : `~numpy.ndarray`
        Constants coefficients for each of them there is only one Zernike
        circle polynomial. The coefficients are between -2 and 2.

    Returns
    -------
    W : `~numpy.ndarray`
        Zernike circle polynomial already evaluated and multiplied by their
        coefficients. Its values are between -1 and 1.
    '''

    # Total number of Zernike circle polynomials
    n = int((np.sqrt(1 + 8 * K_coeff.size) - 3) / 2)

    # list of tuples with (n, l) allowed values
    nl = [(i, j) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]

    # Wavefront (aberration) distribution
    W = sum(
        K_coeff[i] * U(*nl[i], rho, theta)
        for i in range(K_coeff.size)
        )

    return W


def phase(K_coeff, notilt, pr, resolution=1e3):
    '''
    Aperture phase distribution (or phase error), for an specific telescope
    primary reflector. In general the tilt (average slope in x- and
    y-directions, related to telescope pointings) is subtracted from its
    calculation. Function used to show the final results from the fit
    procedure.

    Parameters
    ----------
    K_coeff : `~numpy.ndarray`
        Constants coefficients for each of them there is only one Zernike
        circle polynomial. The coefficients are between -2 and 2.
    notilt : bool
        Boolean to include or exclude the tilt coefficients in the aperture
        phase distribution. The Zernike circle polynomials are related to tilt
        through U(n=1, l=-1) and U(n=1, l=1).
    pr : float
        Primary reflector radius in meters.
    resolution : int
        Resolution for the phase error map, usually used 1e3 in the pyoof
        package.

    Returns
    -------
    phi : `~numpy.ndarray`
        Aperture phase distribution for an specific primary dish radius,
        measured in radians.
    x : `~numpy.ndarray`
        x-axis dimensions for the primary reflector in meters.
    y : `~numpy.ndarray`
        y-axis dimensions for the primary reflector in meters.
    '''

    _K_coeff = K_coeff.copy()

    # Erasing tilt dependence
    if notilt:
        _K_coeff[1] = 0  # For value K(-1, 1) = 0
        _K_coeff[2] = 0  # For value K(1, 1) = 0

    x = np.linspace(-pr, pr, resolution)
    y = np.linspace(-pr, pr, resolution)
    x_grid, y_grid = np.meshgrid(x, y)

    r, t = cart2pol(x_grid, y_grid)
    r_norm = r / pr  # For orthogonality U(n, l) polynomials

    # Wavefront (aberration) distribution
    W = wavefront(rho=r_norm, theta=t, K_coeff=_K_coeff)
    W[(x_grid ** 2 + y_grid ** 2 > pr ** 2)] = 0

    phi = W * 2 * np.pi  # Aperture phase distribution in radians

    return x, y, phi


def aperture(x, y, K_coeff, I_coeff, d_z, wavel, illum_func, telgeo):
    '''
    Aperture distribution. Collection of individual distribution/functions:
    i.e. illumination function, blockage distribution, aperture phase
    distribution and OPD function. In general is a complex quantity, its
    phase an amplitude are better understood separately. The FT (2-dim) of the
    aperture represents the (field) radiation pattern.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Grid value for the x variable in meters.
    y : `~numpy.ndarray`
        Grid value for the y variable in meters.
    K_coeff : `~numpy.ndarray`
        Constants coefficients for each of them there is only one Zernike
        circle polynomial. The coefficients are between -2 and 2.
    I_coeff : `~numpy.ndarray`
        List which contains 4 coefficients, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, sigma_dB, x0, y0].
    d_z : float
        Radial offset added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It is usually of the order of 1e-2 m.
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with coefficients illum_func(x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (float) in meters.
        telego = [block_dist, opd_func, pr].

    Returns
    -------
    E : `~numpy.ndarray`
        Grid value that contains general expression for aperture distribution.
    '''

    r, t = cart2pol(x, y)

    [block_dist, opd_func, pr] = telgeo
    B = block_dist(x=x, y=y)

    # Normalization to be used in the Zernike circle polynomials
    r_norm = r / pr

    # Wavefront (aberration) distribution
    W = wavefront(rho=r_norm, theta=t, K_coeff=K_coeff)
    delta = opd_func(x=x, y=y, d_z=d_z)  # Optical path difference function
    Ea = illum_func(x=x, y=y, I_coeff=I_coeff, pr=pr)  # Illumination function

    # Transformation: wavefront (aberration) distribution -> phase error
    phi = (W + delta / wavel) * 2 * np.pi  # phase error plus the OPD function

    E = B * Ea * np.exp(phi * 1j)  # Aperture distribution

    return E


def radiation_pattern(
    K_coeff, I_coeff, d_z, wavel, illum_func, telgeo, resolution, box_factor
        ):
    '''
    Spectrum or (field) radiation pattern, it is the FFT2 computation of the
    aperture distribution in a rectangular grid. Passing the majority of
    arguments to the aperture distribution except the FFT2 resolution.

    Parameters
    ----------
    K_coeff : `~numpy.ndarray`
        Constants coefficients for each of them there is only one Zernike
        circle polynomial. The coefficients are between -2 and 2.
    I_coeff : `~numpy.ndarray`
        List which contains 4 coefficients, the illumination amplitude, the
        illumination taper and the two coordinate offset.
        I_coeff = [i_amp, sigma_dB, x0, y0].
    d_z : float
        Radial offset added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It is usually of the order of 1e-2 m.
    wavel : float
        Wavelength of the observation in meters.
    illum_func : function
        Illumination function with coefficients illum_func(x, y, I_coeff, pr).
    telgeo : list
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (float) in meters.
        telego = [block_dist, opd_func, pr].
    resolution : int
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and a
        power of 2 for FFT faster processing.
    box_factor : int
        Related to the FFT resolution, defines the image in the at the pixel
        size level, depending on the data a good value has to be chosen, the
        standard is 5, then the box_size = 5 * pr.

    Returns
    -------
    u_shift : `~numpy.ndarray`
        u wave-vector in 1 / m units. It belongs to the x coordinate in meters
        from the aperture distribution.
    v_shift : `~numpy.ndarray`
        v wave-vector in 1 / m units. It belongs to the y coordinate in meters
        from the aperture distribution.
    F_shift : `~numpy.ndarray`
        Output from the FFT2 pack, unnormalized solution in a grid same as
        aperture input computed from a given resolution.
    '''

    # Arrays to generate (field) radiation pattern
    pr = telgeo[2]
    box_size = pr * box_factor

    x = np.linspace(-box_size, box_size, resolution)
    y = x
    x_grid, y_grid = np.meshgrid(x, y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Aperture distribution model
    E = aperture(
        x=x_grid,
        y=y_grid,
        K_coeff=K_coeff,
        I_coeff=I_coeff,
        d_z=d_z,
        wavel=wavel,
        illum_func=illum_func,
        telgeo=telgeo
        )

    # Normalization may not be needed
    # Nx, Ny = x.size, y.size

    F = np.fft.fft2(E)  # * 4 / np.sqrt(Nx * Ny)
    F_shift = np.fft.fftshift(F)  # (field) radiation pattern

    # wave-vectors in 1 / m
    u, v = np.fft.fftfreq(x.size, dx), np.fft.fftfreq(y.size, dy)
    u_shift, v_shift = np.fft.fftshift(u), np.fft.fftshift(v)

    return u_shift, v_shift, F_shift
