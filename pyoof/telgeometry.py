#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from functools import partial
from .math_functions import linear_equation

__all__ = [
    'telescope_geo', 'blockage_manual', 'blockage_effelsberg'
    ]


def telescope_geo(telescope):
    """
    Telescope geometry selection and possible manual setting of a particular
    radio telescope radius
    """

    if telescope == 'effelsberg':
        blockage = blockage_effelsberg
        pr = 50  # Primary reflector radius

    elif type(telescope) == tuple:
        (pr, sr) = telescope
        if pr <= sr:
            print('Primary radius cannot be smaller than secondary raidus! \n')
        blockage = partial(blockage_manual, (pr, sr))

    else:
        print(
            'Select an existing telescope geometry: `effelsberg` or a primary',
            'and secondary disc radius (pr, sr). The radii must be an integer',
            '\n'
            )
        raise SystemExit

    return blockage, pr


def blockage_manual(geo, x, y):

    (pr, sr) = geo
    block = np.zeros(x.shape)  # or y.shape same
    block[(x ** 2 + y ** 2 < pr ** 2) & (x ** 2 + y ** 2 > sr ** 2)] = 1

    return block


def blockage_effelsberg(x, y):
    """
    Truncation in the aperture function, given by the hole generated for the
    secondary reflector, the supporting structure and shade efects in the
    Effelsberg telescope.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.
    geometry : list
        Characteristic geometry for the Effelsberg telescope,
        [pr, sr, L, a, alpha].

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
