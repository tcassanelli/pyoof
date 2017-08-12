#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from ..math_functions import linear_equation

__all__ = [
    'delta_effelsberg', 'delta_manual', 'block_manual',
    'block_effelsberg'
    ]


def delta_effelsberg(x, y, d_z):
    """
    Optical path difference or delta function. Given by geometry of
    the telescope and defocus parameter. For Cassegrain/Gregorain geometries.
    Foci specific for Effelsberg radio telescope. In the aperture function
    delta is transformed to radians

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.
    d_z : float
        Distance between the secondary and primary reflector measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end.

    Returns
    -------
    delta : ndarray
        Phase change in meters.
    """

    # Cassegrain/Gregorian (at focus) telescope
    Fp = 30  # Focus primary reflector m
    F = 387.39435  # Total focus Gregorian telescope m
    r = np.sqrt(x ** 2 + y ** 2)  # polar coordinates radius
    a = r / (2 * Fp)
    b = r / (2 * F)

    delta = d_z * ((1 - a ** 2) / (1 + a ** 2) + (1 - b ** 2) / (1 + b ** 2))

    return delta


def delta_manual(Fp, F):
    """
    Optical path difference or delta function. Given by geometry of
    the telescope and defocus parameter. For Cassegrain/Gregorain geometries.
    Primary and total (or effective) foci are required. In the aperture
    function delta is transformed to radians

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.
    d_z : float
        Distance between the secondary and primary reflector measured in
        meters (radial offset). It is the characteristic measurement to give
        an offset and an out-of-focus image at the end.
    Fp : float
        Focus primary reflector (main) in meters.
    F : float
        Effective or totla focus for the telescope mirror configuration.

    Returns
    -------
    delta : func
        It returns the function delta(x, y, d_z), which depends only on the
        grid and radial offset values.
    """

    def delta(x, y, d_z):
        # Cassegrain/Gregorian (at focus) telescope
        r = np.sqrt(x ** 2 + y ** 2)  # polar coordinates radius
        a = r / (2 * Fp)
        b = r / (2 * F)

        _delta = d_z * (
            (1 - a ** 2) / (1 + a ** 2) + (1 - b ** 2) / (1 + b ** 2)
            )

        return _delta

    return delta


def block_manual(pr, sr, a, L):
    """
    Truncation for the aperture function, manual set up for the primary radius
    (pr), secondary radius (sr), hald thickness of a support leg (a) and
    length of the support leg (L) measured from the edge of the sr. It has been
    considered 4 support legs. To omit sr, a and L set them to zero.

    Parameters
    ----------
    pr : float
        Primary reflector radius.
    sr : float
        Seconday reflector radius.
    a : float
        Half thickness of a support leg.
    L : float
        Length of a support leg, measured from the edge of the sr to its end.

    Returns
    -------
    block : func
        It returns the function block(x, y), which depends only on the grid
        values.
    """

    def block(x, y):
        _block = np.zeros(x.shape)  # or y.shape same
        _block[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
        _block[(-(sr + L) < x) & (x < (sr + L)) & (-a < y) & (y < a)] = 0
        _block[(-(sr + L) < y) & (y < (sr + L)) & (-a < x) & (x < a)] = 0
        return _block

    return block


def block_effelsberg(x, y):
    """
    Truncation in the aperture function, given by the hole generated for the
    secondary reflector, the supporting structure and shade efects in the
    Effelsberg telescope.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the x variable.

    Returns
    -------
    block : ndarray
    """

    # default Effelsberg geometry
    pr = 50  # Primary reflector radius
    sr = 3.25  # secondary reflector radius

    L = 20   # length support structure (from the edge of the sr)
    a = 1  # half thickness support structure

    # angle shade effect in aperture
    alpha = np.radians(10)  # triangle angle

    block = np.zeros(x.shape)  # or y.shape same
    block[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
    block[(-(sr + L) < x) & (x < (sr + L)) & (-a < y) & (y < a)] = 0
    block[(-(sr + L) < y) & (y < (sr + L)) & (-a < x) & (x < a)] = 0

    csc2 = np.sin(alpha) ** (-2)  # cosecant squared

    # base of the triangle
    d = (-a + np.sqrt(a ** 2 - (a ** 2 - pr ** 2) * csc2)) / csc2

    # points for the triangle coordinates
    A = sr + L
    B = a
    C = d / np.tan(alpha)
    D = a + d

    y1 = linear_equation((A, B), (C, D), x)
    y2 = linear_equation((A, -B), (C, -D), x)
    x3 = linear_equation((-A, B), (-C, D), y)
    x4 = linear_equation((-A, -B), (-C, -D), y)
    y5 = linear_equation((-A, -B), (-C, -D), x)
    y6 = linear_equation((-A, B), (-C, D), x)
    x7 = linear_equation((A, -B), (C, -D), y)
    x8 = linear_equation((A, B), (C, D), y)

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
