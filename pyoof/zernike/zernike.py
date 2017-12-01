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
    Radial Zernike polynomials generator (:math:`R^m_n` from Born & Wolf
    definition). The :math:`m`, :math:`n` are integers, :math:`n \geqslant 0`
    and :math:`n - m` even. Only used to compute the general expression for
    the Zernike circle polynomials, `~pyoof.zernike.U`.

    Parameters
    ----------
    n : int
        It is :math:`n \geqslant 0`. Order of the radial component.
    m : int
        Positive number, relative to the angle component.
    rho : `~numpy.ndarray`
        Values for the radial component, :math:`\sqrt{x^2 + y^2}`.

    Returns
    -------
    radial_poly : `~numpy.ndarray`
        Radial Zernike polynomial already evaluated.
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
    Zernike circle polynomials generator (:math:`U^\ell_n` from Born & Wolf
    definition). The :math:`\ell`, :math:`n` are integers, :math:`n \geqslant
    0` and :math:`n - |\ell|` even. Expansion of a complete set of orthonormal
    polynomials in a unitary circle, for the wavefront (aberration)
    distribution. The total number of polynomials is given by :math:`(n + 1)(n
    + 2) / 2.`

    Parameters
    ----------
    n : int
        It is n >= 0. Relative to radial component.
    l : int
        l Can be positive or negative, relative to angle component.
    rho : `~numpy.ndarray`
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).
    theta : `~numpy.ndarray`
        Values for the angular component. For a rectangular grid x and y are
        evaluated as theta = np.arctan(y / x).

    Returns
    -------
    zernike_circle_poly : `~numpy.ndarray`
        Zernike circle polynomial already evaluated.
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
