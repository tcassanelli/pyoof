# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from astropy import units as apu
from ..math_functions import cart2pol, rms
from ..zernike import U

__all__ = [
    'illum_pedestal', 'illum_gauss', 'wavefront', 'phase', 'aperture',
    'radiation_pattern', 'e_rs'
    ]


def e_rs(phase):
    """
    Computes the random-surface-error efficiency,
    :math:`\\varepsilon_\\mathrm{rs}`, using the Ruze's equation.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        The phase error, :math:`\\varphi(x, y)`, is a two dimensional array (
        one of the solutions from the pyoof package). Its amplitude values are
        in radians.

    Notes
    -----
    Ruze's equation is derived empirically from a fat reflector with Gaussian
    distributed errors, and it expressed as,

    .. math::
        \\varepsilon_\\mathrm{rs} =
        \\mathrm{e}^{-(4\\pi\\delta_\\mathrm{rms}/\\lambda)^2}.

    Where :math:`\\delta_\\mathrm{rms}` corresponds to the root-mean-squared
    deviation. The Python function uses the key **phase** because the term
    :math:`4\\pi\\delta_\\mathrm{rms}/\\lambda` corresponds to the phase error.

    Examples
    --------
    >>> import numpy as np
    >>> pr = 50  # m, Effelsberg primary dish radius
    >>> box_factor = 5
    >>> resolution = 2 ** 8
    >>> # then the x and y array will be defined
    >>> x = np.linspace(-5 * pr, 5 * pr, resolution)
    >>> x.size
    256
    """

    rms_rad = rms(phase)  # rms value in radians

    with apu.set_enabled_equivalencies(apu.dimensionless_angles()):
        return np.exp(-rms_rad ** 2)


def illum_pedestal(x, y, I_coeff, pr, q=2):
    """
    Illumination function, :math:`E_\\mathrm{a}(x, y)`, parabolic taper on a
    pedestal, sometimes called apodization, taper or window. Represents the
    distribution of light in the primary reflector. The illumination reduces
    the side lobes in the FT and it is a (feed) receiver property.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Grid value for the :math:`x` variable in meters.
    y : `~numpy.ndarray`
        Grid value for the :math:`y` variable in meters.
    I_coeff : `~numpy.ndarray`
        List which contains 4 parameters, the illumination amplitude,
        :math:`A_{E_\\mathrm{a}}`, the illumination taper,
        :math:`c_\\mathrm{dB}` and the two coordinate offset, :math:`(x_0,
        y_0)`. The illumination coefficients must be listed as follows,
        ``I_coeff = [i_amp, c_dB, x0, y0]``.
    pr : `float`
        Primary reflector radius in meters.
    q : `int`
        Order of the parabolic taper on a pedestal, it is commonly set at
        :math:`q = 2`.

    Returns
    -------
    Ea : `~numpy.ndarray`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`.

    Notes
    -----
    The parabolic taper on a pedestal has the following mathematical formula,

    .. math::

        E_{\\mathrm{a}} (\\rho') = C + (1 - C) \\cdot
        \\left[ 1 - \\left( \\frac{\\rho'}{R} \\right)^2 \\right]^q,

    .. math::

        C=10^{\\frac{c_\\mathrm{dB}}{20}}.
    """

    i_amp, c_dB, x0, y0 = I_coeff

    # workaround for units dB
    if type(c_dB) == apu.quantity.Quantity:
        c = 10 ** (c_dB / 20. / apu.dB)
    else:
        c = 10 ** (c_dB / 20.)

    # c_dB has to be negative, bounds given [-8, -25]

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Parabolic taper on a pedestal
    Ea = i_amp * (c + (1. - c) * (1. - (r / pr) ** 2) ** q)

    return Ea


def illum_gauss(x, y, I_coeff, pr):
    """
    Illumination function, :math:`E_\\mathrm{a}(x, y)`, Gaussian, sometimes
    called apodization, taper or window. Represents the distribution of light
    in the primary reflector. The illumination reduces the side lobes in the
    FT and it is a property of the receiver.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Grid value for the :math:`x` variable in meters.
    y : `~numpy.ndarray`
        Grid value for the :math:`y` variable in meters.
    I_coeff : `~numpy.ndarray`
        List which contains 4 parameters, the illumination amplitude,
        :math:`A_{E_\\mathrm{a}}`, the illumination taper,
        :math:`\\sigma_\\mathrm{dB}` and the two coordinate offset, :math:`(
        x_0, y_0)`. The illumination coefficients must be listed as follows,
        ``I_coeff = [i_amp, sigma_dB, x0, y0]``.
    pr : `float`
        Primary reflector radius in meters.

    Returns
    -------
    Ea : `~numpy.ndarray`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`.

    Notes
    -----
    The Gaussian illumination function has the same formula of a normalized
    Gaussian distribution.
    """
    i_amp, sigma_dB, x0, y0 = I_coeff
    sigma = 10 ** (sigma_dB / 20)  # -15 to -20 dB
    norm = np.sqrt(2 * np.pi * sigma ** 2)  # normalization Gaussian
    Ea = (
        i_amp * norm *
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma * pr) ** 2))
        )

    return Ea


def wavefront(rho, theta, K_coeff):
    """
    Computes the wavefront (aberration) distribution, :math:`W(x, y)`. It
    tells how is the error distributed along the primary dish, it is related
    to the phase error. The wavefront (aberration) distribution is described
    as a parametrization of the Zernike circle polynomials multiplied by a set
    of coefficients, :math:`K_{n\\ell}`.

    Parameters
    ----------
    rho : `~numpy.ndarray`
        Values for the radial component. :math:`\\sqrt{x^2 + y^2} /
        \\varrho_\\mathrm{max}` normalized by its maximum radius.
    theta : `~numpy.ndarray`
        Values for the angular component. :math:`\\vartheta = \\mathrm{arctan}(
        y / x)`.
    K_coeff : `~numpy.ndarray`
        Constants coefficients, :math:`K_{n\\ell}`, for each of them there is
        only one Zernike circle polynomial, :math:`U^\\ell_n(\\varrho,
        \\varphi)`. The coefficients are between :math:`[-2, 2]`.

    Returns
    -------
    W : `~numpy.ndarray`
        Wavefront (aberration) distribution, :math:`W(x, y)`. Zernike circle
        polynomials already evaluated and multiplied by their coefficients.

    Notes
    -----
    The wavefront (aberration) distribution it strictly related to the Zernike
    circle polynomials through the expression,

    .. math::
        W(\\varrho, \\vartheta) = \\sum_{n,
        \\ell}K_{n\\ell}U^\\ell_n(\\varrho, \\vartheta).
    """

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
    """
    Aperture phase distribution (or phase error), :math:`\\varphi(x, y)`, for
    an specific telescope primary reflector. In general the tilt (average
    slope in :math:`x`- and :math:`y`-directions, related to telescope
    pointings) is subtracted from its calculation. Function used to show the
    final results from the fit procedure.

    Parameters
    ----------
    K_coeff : `~numpy.ndarray`
        Constants coefficients, :math:`K_{n\\ell}`, for each of them there is
        only one Zernike circle polynomial, :math:`U^\\ell_n(\\varrho,
        \\varphi)`. The coefficients are between :math:`[-2, 2]`.
    notilt : `bool`
        Boolean to include or exclude the tilt coefficients in the aperture
        phase distribution. The Zernike circle polynomials are related to tilt
        through :math:`U^{-1}_1(\\varrho, \\varphi)` and
        :math:`U^1_1(\\varrho, \\varphi)`.
    pr : `float`
        Primary reflector radius in meters.
    resolution : `int`
        Resolution for the phase error map, usually used ``resolution = 1e3``
        in the pyoof package.

    Returns
    -------
    x : `~numpy.ndarray`
        :math:`x`-axis dimensions for the primary reflector in meters.
    y : `~numpy.ndarray`
        :math:`y`-axis dimensions for the primary reflector in meters.
    phi : `~numpy.ndarray`
        Aperture phase distribution, :math:`\\varphi(x, y)`, for an specific
        primary dish radius, measured in radians.

    Notes
    -----
    The aperture phase distribution or phase error, :math:`\\varphi(x, y)` is
    related to the wavefront (aberration) distribution, from classical optics,
    through the expression,

    .. math::
        \\varphi(x, y) = 2\\pi \\cdot W(x, y) = 2\\pi \\cdot \\sum_{n, \\ell}
        K_{n\\ell}U^\\ell_n(\\varrho, \\vartheta).

    Examples
    --------
    To use compute the aperture phase distribution, :math:`\\varphi(x, y)`,
    first a set of coefficients need to be generated, ``K_coeff``, then simply
    execute the `~pyoof.aperture.phase` function.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pyoof import aperture
    >>> pr = 50  # primary relfector m
    >>> n = 5  # order polynomial
    >>> N_K_coeff = (n + 1) * (n + 2) // 2  # max polynomial number
    >>> K_coeff = np.random.normal(0., .1, N_K_coeff)
    >>> x, y, phi = aperture.phase(K_coeff=K_coeff, notilt=True, pr=50.)
    """

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

    phi = W * 2 * np.pi * apu.rad  # Aperture phase distribution in radians

    return x, y, phi


def aperture(x, y, K_coeff, I_coeff, d_z, wavel, illum_func, telgeo):
    """
    Aperture distribution, :math:`\\underline{E_\\mathrm{a}}(x, y)`.
    Collection of individual distribution/functions: i.e. illumination
    function, :math:`E_\\mathrm{a}(x, y)`, blockage distribution, :math:`B(x,
    y)`, aperture phase distribution, :math:`\\varphi(x, y)` and OPD
    function, :math:`\\delta(x, y;d_z)`. In general, it is a complex
    quantity, its phase an amplitude are better understood separately. The FT
    (2-dim) of the aperture represents the (field) radiation pattern, :math:`F(
    u, v)`.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Grid value for the :math:`x` variable in meters.
    y : `~numpy.ndarray`
        Grid value for the :math:`y` variable in meters.
    K_coeff : `~numpy.ndarray`
        Constants coefficients, :math:`K_{n\\ell}`, for each of them there is
        only one Zernike circle polynomial, :math:`U^\\ell_n(\\varrho,
        \\varphi)`. The coefficients are between :math:`[-2, 2]`.
    I_coeff : `~numpy.ndarray`
        List which contains 4 parameters, the illumination amplitude,
        :math:`A_{E_\\mathrm{a}}`, the illumination taper,
        :math:`c_\\mathrm{dB}` and the two coordinate offset, :math:`(x_0,
        y_0)`. The illumination coefficients must be listed as follows,
        ``I_coeff = [i_amp, c_dB, x0, y0]``.
    d_z : `float`
        Radial offset, :math:`d_z`, added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It is usually of the order of centimeters.
    wavel : `float`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telgeo : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (`float`) in meters. The list
        must have the following order, ``telego = [block_dist, opd_func, pr]``.

    Returns
    -------
    E : `~numpy.ndarray`
        Grid value that contains general expression for aperture distribution,
        :math:`\\underline{E_\\mathrm{a}}(x, y)`.

    Notes
    -----
    The aperture distribution is a collection of distributions/functions, its
    structure follows,

    .. math::
        \\underline{E_\\mathrm{a}}(x, y) = B(x, y)\\cdot E_\\mathrm{a}(x, y)
        \\cdot \\mathrm{e}^{\\mathrm{i} \\{\\varphi(x, y) +
        \\frac{2\\pi}{\\lambda}\\delta(x,y;d_z)\\}}.

    Where it does represent a complex number, with phase: aperture phase
    distribution, plus OPD function and amplitude the blockage distribution
    and illumination function.

    """

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
    phi = (W + delta / wavel) * 2 * np.pi * apu.rad
    # phase error plus the OPD function

    with apu.set_enabled_equivalencies(apu.dimensionless_angles()):
        E = B * Ea * np.exp(phi * 1j)  # Aperture distribution

    return E


def radiation_pattern(
    K_coeff, I_coeff, d_z, wavel, illum_func, telgeo, resolution, box_factor
        ):
    """
    Spectrum or (field) radiation pattern, :math:`F(u, v)`, it is the FFT2
    computation of the aperture distribution, :math:`\\underline{E_\\mathrm{a}}
    (x, y)`, in a rectangular grid. Passing the majority of
    arguments to the aperture distribution except the FFT2 resolution.

    Parameters
    ----------
    K_coeff : `~numpy.ndarray`
        Constants coefficients, :math:`K_{n\\ell}`, for each of them there is
        only one Zernike circle polynomial, :math:`U^\\ell_n(\\varrho,
        \\varphi)`. The coefficients are between :math:`[-2, 2]`.
    I_coeff : `~numpy.ndarray`
        List which contains 4 parameters, the illumination amplitude,
        :math:`A_{E_\\mathrm{a}}`, the illumination taper,
        :math:`c_\\mathrm{dB}` and the two coordinate offset, :math:`(x_0,
        y_0)`. The illumination coefficients must be listed as follows,
        ``I_coeff = [i_amp, c_dB, x0, y0]``.
    d_z : `float`
        Radial offset, :math:`d_z`, added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It is usually of the order of centimeters.
    wavel : `float`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with the key **I_coeff**. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.
    telgeo : `list`
        List that contains the blockage distribution, optical path difference
        (OPD) function, and the primary radius (`float`) in meters. The list
        must have the following order, ``telego = [block_dist, opd_func, pr]``.
    resolution : `int`
        Fast Fourier Transform resolution for a rectangular grid. The input
        value has to be greater or equal to the telescope resolution and with
        power of 2 for faster FFT processing. It is recommended a value higher
        than ``resolution = 2 ** 8``.
    box_factor : `int`
        Related to the FFT resolution (**resolution** key), defines the image
        pixel size level. It depends on the primary radius, ``pr``, of the
        telescope, e.g. a ``box_factor = 5`` returns ``x = np.linspace(-5 *
        pr, 5 * pr, resolution)``, an array to be used in the FFT2
        (`~numpy.fft.fft2`).

    Returns
    -------
    u_shift : `~numpy.ndarray`
        :math:`u` wave-vector in 1 / m units. It belongs to the :math:`x`
        coordinate in meters from the aperture distribution,
        :math:`\\underline{E_\\mathrm{a}}(x, y)`.
    v_shift : `~numpy.ndarray`
        :math:`v` wave-vector in 1 / m units. It belongs to the y coordinate
        in meters from the aperture distribution,
        :math:`\\underline{E_\\mathrm{a}}(x, y)`.
    F_shift : `~numpy.ndarray`
        Output from the FFT2 (`~numpy.fft.fft2`), :math:`F(u, v)`,
        unnormalized solution in a grid, defined by **resolution** and
        **box_factor** keys.

    Notes
    -----
    The (field) radiation pattern is the direct Fourier Transform in two
    dimensions of the aperture distribution, hence,

    .. math::

        F(u, v) = \\mathcal{F} \\left[ \\underline{E_\\mathrm{a}}(x, y)
        \\right].
    """

    # Arrays to generate (field) radiation pattern
    pr = telgeo[2]
    box_size = pr * box_factor

    x = np.linspace(-box_size, box_size, resolution)
    y = x.copy()
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

    F = np.fft.fft2(E)
    F_shift = np.fft.fftshift(F)  # (field) radiation pattern

    # wave-vectors in 1 / m
    u, v = np.fft.fftfreq(x.size, dx), np.fft.fftfreq(y.size, dy)

    # workaround units
    if type(x) == apu.quantity.Quantity:
        u_shift = np.fft.fftshift(u) * u.unit
        v_shift = np.fft.fftshift(v) * v.unit
    else:
        u_shift, v_shift = np.fft.fftshift(u), np.fft.fftshift(v)

    return u_shift, v_shift, F_shift
