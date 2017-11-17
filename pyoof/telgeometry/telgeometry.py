#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from ..math_functions import line_equation

__all__ = [
    'opd_effelsberg', 'opd_manual', 'block_manual', 'block_effelsberg'
    ]


def opd_effelsberg(x, y, d_z):
    """
    Optical path difference (OPD) function. Given by the geometry of the
    telescope and radial offset parameter (d_z). This function is specific
    for the Effelsberg telescope.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable in meters.
    y : ndarray
        Grid value for the y variable in meters.
    d_z : float
        Radial offset added to the sub-reflector in meters. This
        characteristic measurement adds the classical interference pattern to
        the beam maps, normalized squared (field) radiation pattern, which is
        an out-of-focus property. It is usually of the order of 1e-2 m.

    Returns
    -------
    opd : ndarray
        Optical path difference function.
    """

    # Cassegrain/Gregorian (at focus) telescope
    Fp = 30  # Focus primary reflector m
    F = 387.39435  # Total focus Gregorian telescope m
    r = np.sqrt(x ** 2 + y ** 2)  # polar coordinates radius
    a = r / (2 * Fp)
    b = r / (2 * F)

    opd = d_z * ((1 - a ** 2) / (1 + a ** 2) + (1 - b ** 2) / (1 + b ** 2))

    return opd


def opd_manual(Fp, F):
    """
    Optical path difference (OPD) function. Given by geometry of the telescope
    and defocus parameter. For Cassegrain/Gregorian geometries. Primary and
    total (or effective) foci are required. In the aperture function delta is
    transformed to radians

    Parameters
    ----------
    Fp : float
        Focus primary reflector (main) in meters.
    F : float
        Effective or total focus for the telescope mirror configuration.

    Returns
    -------
    opd_func : function
        It returns the function opd_func(x, y, d_z), which depends only on the
        grid and radial offset values. For arguments see opd_effelsberg.
    """

    def opd_func(x, y, d_z):
        # Cassegrain/Gregorian (at focus) telescope
        r = np.sqrt(x ** 2 + y ** 2)  # radial polar coordinate
        a = r / (2 * Fp)
        b = r / (2 * F)

        opd = d_z * (
            (1 - a ** 2) / (1 + a ** 2) + (1 - b ** 2) / (1 + b ** 2)
            )

        return opd

    return opd_func


def block_effelsberg(x, y):
    """
    Truncation in the aperture (amplitude) distribution, given by the
    telescope's structure; i.e. support legs, sub-reflector and shade effect
    as seen from the secondary focus of the Effelsberg telecope.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable in meters.
    y : ndarray
        Grid value for the y variable in meters.

    Returns
    -------
    block : ndarray
        Aperture (amplitude) distribution truncation. Values that are zero
        correspond to blocked values.
    """

    # Default Effelsberg geometry
    pr = 50  # Primary reflector radius
    sr = 3.25  # Sub-reflector radius

    L = 20   # Length support structure (from the edge of the sr)
    a = 1  # Half-width support structure

    # Angle shade effect in aperture plane
    alpha = np.radians(10)  # triangle angle

    block = np.zeros(x.shape)  # or y.shape same
    block[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
    block[(-(sr + L) < x) & (x < (sr + L)) & (-a < y) & (y < a)] = 0
    block[(-(sr + L) < y) & (y < (sr + L)) & (-a < x) & (x < a)] = 0

    csc2 = np.sin(alpha) ** (-2)  # squared cosecant

    # base of the triangle
    d = (-a + np.sqrt(a ** 2 - (a ** 2 - pr ** 2) * csc2)) / csc2

    # points for the triangle coordinates
    A = sr + L
    B = a
    C = d / np.tan(alpha)
    D = a + d

    y1 = line_equation((A, B), (C, D), x)
    y2 = line_equation((A, -B), (C, -D), x)
    x3 = line_equation((-A, B), (-C, D), y)
    x4 = line_equation((-A, -B), (-C, -D), y)
    y5 = line_equation((-A, -B), (-C, -D), x)
    y6 = line_equation((-A, B), (-C, D), x)
    x7 = line_equation((A, -B), (C, -D), y)
    x8 = line_equation((A, B), (C, D), y)

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


def block_manual(pr, sr, a, L):
    """
    Truncation for the aperture (amplitude) distribution, manual set up for
    the primary radius (pr), sub-reflector radius (sr), half-width of a
    support leg (a) and length of the support leg (L) measured from the edge
    of the sr. It has been considered 4 support legs. To exclude sr, a or L
    set them to zero.

    Parameters
    ----------
    pr : float
        Primary reflector radius.
    sr : float
        Sub-reflector radius.
    a : float
        Half-width of a support leg.
    L : float
        Length of a support leg, measured from the edge of the sub-reflector
        towards its end.

    Returns
    -------
    block_func : function
        It returns the function block_func(x, y), which depends only on the
        grid values. For arguments see block_effelsberg.
    """

    def block_func(x, y):
        block = np.zeros(x.shape)  # or y.shape same
        block[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1
        block[(-(sr + L) < x) & (x < (sr + L)) & (-a < y) & (y < a)] = 0
        block[(-(sr + L) < y) & (y < (sr + L)) & (-a < x) & (x < a)] = 0
        return block

    return block_func
