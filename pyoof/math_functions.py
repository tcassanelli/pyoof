#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np
from scipy import interpolate
from astropy import units as apu

__all__ = ['cart2pol', 'co_matrices', 'line_equation', 'rms', 'norm', 'snr']


def norm(P):
    """
    Data normalization. This is a pre-process right before starting the least
    squares minimization.

    Parameters
    ----------
    P : `~numpy.ndarray`
        Two-dimensional power pattern (or beam). Units are arbitrary, this
        same process is applied to the observed and generated power pattern.

    Returns
    -------
    P_norm : `~numpy.ndarray`
        Normalized power pattern.
    """

    # normalization by it's maximum
    # P_norm = P / P.max()

    # normalization
    P_norm = (P - P.min()) / (P.max() - P.min())

    # standardization
    # P_norm = (P - P.mean()) / P.std()

    return P_norm


def cart2pol(x, y):
    """
    Transformation from Cartesian to polar coordinates, in two dimensions.

    Parameters
    ----------
    x : `~numpy.ndarray` or `~astropy.units.quantity.Quantity`
        Grid value for the :math:`x` variable in length units.
    y : `~numpy.ndarray` or `~astropy.units.quantity.Quantity`
        Grid value for the :math:`y` variable in length units.

    Returns
    -------
    rho : `~numpy.ndarray` or `~astropy.units.quantity.Quantity`
        Grid value for the radial variable.
    theta : `~numpy.ndarray` or `~astropy.units.quantity.Quantity`
        Grid value for the angular variable, in radians.
    """

    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return rho, theta


def co_matrices(res, jac, n_pars):
    """
    Computes the estimated Variance-Covariance and the Correlation matrices.
    Assuming an estimator normally distributed (and consistent).

    Parameters
    ----------
    res : `~numpy.ndarray`
        Last residual evaluation from a fit procedure (least squares
        minimization), residual understood as data - model.
    jac : `~numpy.ndarray`
        Last Jacobian matrix evaluation from a fit procedure.
    n_pars : `int`
        Total number of parameters in the model (only the ones that have been
        fitted). It is related to the degrees of freedom.

    Returns
    -------
    cov : `~numpy.ndarray`
        Variance-Covariance matrix. Dimensions :math:`n \\times p`.
    corr : `~numpy.ndarray`
        Correlation matrix. Dimensions :math:`n \\times p`.
    """

    m = res.size  # Number of data points used in the fit
    p = m - n_pars  # Degrees of freedom

    # Variance-Covariance matrix
    cov = np.dot(res.T, res) / p * np.linalg.inv(np.dot(jac.T, jac))

    sigmas2 = np.diag(np.diag(cov))  # Estimated error variance (sigma ** 2)
    D = np.linalg.inv(np.sqrt(sigmas2))  # Inverted diagonal variance matrix
    corr = np.dot(np.dot(D, cov), D)  # Correlation matrix

    return cov, corr


def line_equation(P1, P2, x):
    """
    Computes the linear equation solution for the :math:`y` vector values
    given two data points, :math:`P_1 = (x_1, y_1)` and :math:`P_2 = (x_2,
    y_2)`, and  the :math:`x` vector.

    Parameters
    ----------
    P1 : `tuple`
        First point that belongs to the desired linear equation.
    P2 : `tuple`
        Second point that belongs to the desired linear equation.
    x : `~numpy.ndarray`
        Data points from the :math:`x`-axis.

    Returns
    -------
    y  : `~numpy.ndarray`
        Linear equation :math:`y`-data points.
    """

    (x1, y1) = P1
    (x2, y2) = P2

    y = (y2 - y1) / (x2 - x1) * (x - x1) + y1

    return y


def rms(phase, circ=False):
    """
    Computes the root-mean-square value from a aperture phase distribution
    map, :math:`\\varphi(x, y)`.

    Parameters
    ----------
    phase : `~numpy.ndarray` or `~astropy.units.quantity.Quantity`
        One or two dimensional array for the aperture phase distribution.
    circ : `bool`
        If `True` it will take the `phase.shape` as the diameter of a circle
        and calculate the root-mean-square only in that portion.
    """

    if circ:
        x = np.linspace(-1, 1, phase.shape[0])
        y = np.linspace(-1, 1, phase.shape[1])
        xx, yy = np.meshgrid(x, y)

        phase[xx ** 2 + yy ** 2 > 1 ** 2] = np.nan
        phase_real = phase[~np.isnan(phase)]
        _rms = np.sqrt(np.nansum(np.square(phase_real)) / phase_real.size)
    else:
        _rms = np.sqrt(np.nansum(np.square(phase)) / phase.size)

    return _rms


def snr(
    u_data, v_data, beam_data, centre=0.03 * apu.deg, radius=0.01 * apu.deg
        ):

    """
    Computes a simple signal-to-noise ratio estimate for a centered beam (or
    in focus) power patter.

    Parameters
    ----------
    beam_data_norm : `np.ndarray`
        The ``beam_data`` is an array with a single observed beam map in-focus,
        :math:`P^\\mathrm{obs}(u, v)`, it can be normalized or not.
    u_data : `~astropy.units.quantity.Quantity`
        :math:`x` axis value for the single in-focus beam map.
    v_data : `~astropy.units.quantity.Quantity`
        :math:`y` axis value for the single in-focus beam map.
    centre : `~astropy.units.quantity.Quantity`
        Position where to measure the standard deviation, measured in angles
        (map sky angles).
    radius : `~astropy.units.quantity.Quantity`
        Section in units of angles on where to measure the standard deviation.

    Returns
    -------
    snr : `float`
        The signal-to-noise ratio.
    """

    if beam_data.ndim == 1:
        u_ng = np.linspace(u_data.min(), u_data.max(), 300)
        v_ng = np.linspace(v_data.min(), v_data.max(), 300)

        beam_ng = interpolate.griddata(
            # coordinates of grid points to interpolate from.
            points=(u_data, v_data),
            values=beam_data,
            # coordinates of grid points to interpolate to.
            xi=tuple(np.meshgrid(u_ng, v_ng)),
            method='cubic'
            )

        uu, vv = np.meshgrid(u_ng, v_ng)

        std = np.nanstd(
            beam_ng[(uu - centre) ** 2 + (vv - centre) ** 2 < radius ** 2]
            )
        snr = np.nanmax(beam_ng) / std

    else:
        uu, vv = np.meshgrid(u_data, v_data)
        std = np.nanstd(
            beam_data[(uu - centre) ** 2 + (vv - centre) ** 2 < radius ** 2]
            )
        snr = np.nanmax(beam_data) / std

    return snr.decompose().value
