#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np

__all__ = [
    'cart2pol', 'wavevector2degrees', 'wavevector2radians', 'co_matrices',
    'line_equation', 'rms'
    ]


def cart2pol(x, y):
    """
    Transformation from Cartesian to polar coordinates, in two dimensions.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable.
    y : ndarray
        Grid value for the y variable.

    Returns
    -------
    rho : ndarray
        Grid value for the radial variable.
    theta : ndarray
        Grid value for the angular variable, in radians.
    """

    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return rho, theta


def wavevector2degrees(u, wavel):
    """
    Transformation from a wave-vector 1 / m units to degrees.

    Parameters
    ----------
    u : ndarray
        Wave-vector, result from FFT2 in 1 / m units.
    wavel : ndarray
        Wavelength in meters.

    Returns
    -------
    wavevector_degrees : ndarray
        Wave-vector in degrees.
    """

    wavevector_degrees = np.degrees(u * wavel)

    return wavevector_degrees


def wavevector2radians(u, wavel):
    """
    Transformation from a wave-vector 1 / m units to radians.

    Parameters
    ----------
    u : ndarray
        Wave-vector, result from FFT2 in 1 / m units.
    wavel : ndarray
        Wavelength in meters.

    Returns
    -------
    wavevector_degrees : ndarray
        Wave-vector in radians.
    """

    wavevector_radians = wavevector2degrees(u, wavel) * np.pi / 180

    return wavevector_radians


def co_matrices(res, jac, n_pars):
    """
    Computes the estimated Variance-Covariance and the Correlation matrices.
    Assuming an estimator normally distributed (and consistent).

    Parameters
    ----------
    res : ndarray
        Last residual evaluation from a fit procedure (least squares
        minimization), residual understood as data - model.
    jac : ndarray
        Last Jacobian matrix evaluation from a fit procedure.
    n_pars: int
        Total number of parameters in the model (only the ones that have been
        fitted). It is related to the degrees of freedom.

    Returns
    -------
    cov : ndarray
        Variance-Covariance matrix. Dimensions n x p.
    corr : ndarray
        Correlation matrix. Dimensions n x p.
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
    Computes the linear equation solution for the y vector values given two
    data points, P1 = (x1, y1) and P2 = (x2, y2), and the x vector.

    Parameters
    ----------
    P1 : tuple
        First point that belongs to the desired linear equation.
    P2 : tuple
        Second point that belongs to the desired linear equation.
    x : ndarray
        Data points from the x-axis.

    Returns
    -------
    y  : ndarray
        Linear equation y-data points.
    """

    (x1, y1) = P1
    (x2, y2) = P2

    y = (y2 - y1) / (x2 - x1) * (x - x1) + y1

    return y


def rms(x):
    """
    Computes the root-mean-square value from a aperture phase distribution
    map.

    Parameters
    ----------
    x : ndarray
        One or two dimensional array for the phase distribution.
    """

    # To remove elements out limit radius
    nonzero_values = x[np.nonzero(x)]

    return np.sqrt(np.sum(np.square(nonzero_values)) / nonzero_values.size)
