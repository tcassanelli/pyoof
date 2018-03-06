#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from math import factorial as f


__all__ = [
    'U', 'R'
    ]


def R(n, m, rho):
    """
    Radial Zernike polynomials generator (:math:`R^m_n(\\varrho)` from Born &
    Wolf definition). The :math:`m`, :math:`n` are integers, :math:`n\geqslant
    0` and :math:`n - m` even. Only used to compute the general expression for
    the Zernike circle polynomials, `~pyoof.zernike.U`.

    Parameters
    ----------
    n : `int`
        It is :math:`n \geqslant 0`. Order of the radial component.
    m : `int`
        Positive number, relative to the angle component.
    rho : `~numpy.ndarray`
        Values for the radial component, :math:`\\varrho = \sqrt{x^2 + y^2}`.

    Returns
    -------
    radial_poly : `~numpy.ndarray`
        Radial Zernike polynomial already evaluated, :math:`R^m_n(\\varrho)`.

    Notes
    -----
    The original generating formula for the radial polynomials is given by,

    .. math::

        R^{\\pm m}_n (\\varrho) = \\frac{1}{\\left(\\frac{n-m}{2}\\right)!\cdot
        \\varrho^m}\\left\{\\frac{\\mathrm{d}}{\\mathrm{d}\\left(\\varrho^2
        \\right)} \\right\}^{\\frac{n-m}{2}} \\left\{  \\left( \\varrho^2
        \\right)^{\\frac{n+m}{2}} \cdot  \\left( \\varrho^2 -1
        \\right)^{\\frac{n-m}{2}} \\right\},

    Which can also be expressed as a polynomial sum.

    Examples
    --------
    To start using the radial polynomials simply call the package.

    >>> import numpy as np
    >>> from pyoof import zernike
    >>> r = np.linspace(-1, 1, 5)  # only orthogonal under unitary circle
    >>> zernike.R(n=4, m=2, rho=r)
    array([ 1. , -0.5,  0. , -0.5,  1. ])
    """

    a = (n + m) // 2
    b = (n - m) // 2

    radial_poly = sum(
        (-1) ** s * f(n - s) * rho ** (n - 2 * s) /
        (f(s) * f(a - s) * f(b - s))
        for s in range(0, b + 1)
        )

    return radial_poly


def U(n, l, rho, theta):
    """
    Zernike circle polynomials generator (:math:`U^\ell_n(\\varrho, \\varphi)`
    from Born & Wolf definition). The :math:`\ell`, :math:`n` are integers,
    :math:`n \geqslant 0` and :math:`n - |\ell|` even. Expansion of a complete
    set of orthonormal polynomials in a unitary circle, specially useful when
    computing for the wavefront (aberration) distribution, :math:`W(x, y)`.
    The total number of polynomials is given by :math:`(n + 1)(n + 2) / 2.`

    Parameters
    ----------
    n : `int`
        It is :math:`n \geqslant 0`. Relative to radial component.
    l : `int`
        l Can be positive or negative, relative to angle component.
    rho : `~numpy.ndarray`
        Values for the radial component. :math:`\\varrho = \\sqrt{x^2 + y^2}`.
    theta : `~numpy.ndarray`
        Values for the angular component. For a rectangular grid x and y are
        evaluated as :math:`\\vartheta = \\mathrm{arctan}(y / x)`.

    Returns
    -------
    zernike_circle_poly : `~numpy.ndarray`
        Zernike circle polynomial already evaluated, :math:`U^\ell_n(\\varrho,
        \\varphi)`.

    Notes
    -----
    The generating formula for the Zernike circle polymials make use of the
    radial polynomials, then,

    .. math::
        U^\ell_n(\\varrho, \\vartheta) = R^m_n(\\varrho) \\cdot \\cos
        m\\vartheta \qquad \\ell \geq 0,

    .. math::

        U^\ell_n(\\varrho, \\vartheta) = R^m_n(\\varrho) \\cdot \sin
        m\\vartheta \\qquad \\ell < 0.

    Examples
    --------
    Same as the radial polynomials, just start with the package and then apply
    the order, :math:`n`, and angular dependence, :math:`\\ell`, on the
    function.

    >>> import numpy as np
    >>> from pyoof import zernike, cart2pol
    >>> x = np.linspace(-1, 1, 5)
    >>> r, t = cart2pol(x, x)  # polar coordinates
    >>> zernike.U(n=4, l=-2, rho=r, theta=t)
    array([10. , -0.5,  0. , -0.5, 10. ])
    """

    if not isinstance(l, int):
        raise TypeError(
            'Polynomial angular component (l) has to be an integer'
            )
    if not (n >= 0 and isinstance(n, int)):
        raise TypeError('Polynomial order (n) has to be a positive integer')

    m = abs(l)
    radial = R(n, m, rho)

    if l < 0:
        zernike_circle_poly = radial * np.sin(m * theta)
    else:
        zernike_circle_poly = radial * np.cos(m * theta)

    return zernike_circle_poly
