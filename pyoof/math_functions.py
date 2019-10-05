#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np

__all__ = ['cart2pol', 'co_matrices', 'line_equation', 'rms', 'norm']


def norm(P):
    """
    Data normalization. This is a pre-process right before starting the least
    squares minimization.

    Parameters
    ----------
    P :

    Returns
    -------
    P_norm :

    """

    # normalization by it's maximum
    P_norm = P / P.max()

    # normalization
    # P_norm = (P - P.min()) / (P.max() - P.min())

    # standarization
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


def rms(phase, radius=None):
    """
    Computes the root-mean-square value from a aperture phase distribution
    map, :math:`\\varphi(x, y)`.

    Parameters
    ----------
    phase : `~numpy.ndarray` or `~astropy.units.quantity.Quantity`
        One or two dimensional array for the aperture phase distribution.
    radius : `bool`
        The limit radios where the phase error map is contained in length
        units. By default it is set to None, meaning that will include the
        entire array.
    """

    if radius is not None:
        x = np.linspace(-radius, radius, phase.shape[0])
        y = np.linspace(-radius, radius, phase.shape[1])
        xx, yy = np.meshgrid(x, y)

        phase[xx ** 2 + yy ** 2 > radius ** 2] = np.nan
        phase_real = phase[~np.isnan(phase)]
        _rms = np.sqrt(np.nansum(np.square(phase_real)) / phase_real.size)
    else:
        _rms = np.sqrt(np.nansum(np.square(phase)) / phase.size)

    return _rms
