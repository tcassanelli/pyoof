#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from math import factorial as f


__all__ = [
    'U', 'R'
    ]


def R(m, n, rho):
    """
    Radial Zernike polynomials generator (R from Born & Wolf definition).
    The m, n are integers, n >= 0 and n - m even. Only used to compute the
    general expression for the Zernike circle polynomials, U.

    Parameters
    ----------
    m : int
        Positive number, relative to the angle component.
    n : int
        It is n >= 0. Order of the radial component.
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).

    Returns
    -------
    radial : ndarray
        Radial Zernike polynomial already evaluated.
    """

    if type(m) == float or type(n) == float:
        print('WARNING: l and n can only be integer numbers')

    if n < 0:
        print('WARNING: n can only have integer positive values')

    a = (n + m) // 2
    b = (n - m) // 2

    radial = sum(
        (-1) ** s * f(n - s) * rho ** (n - 2 * s) /
        (f(s) * f(a - s) * f(b - s))
        for s in range(0, b + 1)
        )

    return radial


def U(l, n, theta, rho):
    """
    Zernike circle polynomials generator (U from Born & Wolf definition).
    The l, n are integers, n >= 0 and n - |l| even.
    Expansion of a complete set of orthonormal polynomials in a unitary circle,
    for the aberration function.
    The n value determines the number of polynomials (n + 1) * (n + 2) / 2.

    Parameters
    ----------
    l : int
        Can be positive or negative, relative to angle component.
    n : int
        It is n >= 0. Relative to radial component.
    theta : ndarray
        Values for the angular component. For a rectangular grid x and y are
        evaluated as theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).

    Returns
    -------
    zernike_circle : ndarray
        Zernike circle polynomial already evaluated.
    """

    if type(l) == float or type(n) == float:
        print('WARNING: l and n can only be integer numbers')

    if n < 0:
        print('WARNING: n can only have integer positive values')

    m = abs(l)

    radial = R(m, n, rho)

    if l < 0:
        zernike_circle = radial * np.sin(m * theta)
    else:
        zernike_circle = radial * np.cos(m * theta)

    return zernike_circle
