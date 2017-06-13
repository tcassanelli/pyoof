#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np

__all__ = [
    'cart2pol', 'wavevector2degrees', 'wavevector2radians', 'co_matrices',
    'linear_equation', 'angle_selection'
    ]


def angle_selection(angle):

    if angle == 'degrees':
        wavevector_change = wavevector2degrees

    elif angle == 'radians':
        wavevector_change = wavevector2radians

    else:
        print('Select `radians` or `degrees` \n')
        raise SystemExit

    return wavevector_change, angle


def cart2pol(x, y):
    """
    Transformation for the cartesian coord. to polars. It is needed for the
    aperture function.

    Parameters
    ----------
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.

    Returns
    -------
    rho : ndarray
        Grid value for the radial variable, same as the contour plot.
    theta : ndarray
        Grid value for the angular variable, same as the contour plot.
    """

    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return rho, theta


def wavevector2degrees(x, wavel):
    """
    Converst wave-vector 1/m to degrees.

    Parameters
    ----------
    x : ndarray
        Wave-vector, result from FFT in 2-dim.
    wavel : ndarray
        Wavelength.

    Returns
    -------
    wave_vector_degrees : ndarray
        Wave-vector in degrees.
    """

    wave_vector_degrees = np.degrees(x * wavel)

    return wave_vector_degrees


def wavevector2radians(x, wavel):

    wave_vector_radians = wavevector2degrees(x, wavel) * np.pi / 180

    return wave_vector_radians


def co_matrices(res, jac, n_pars):
    """
    Computes the estimated Variance-Covariance and the Correlation matrices.
    Assuming an estimator normally distributed (and consistent).

    Parameters
    ----------
    res : ndarray
        Last residual evaluation from a fit procedure, residual understood as
        model - data.
    jac : ndarray
        Last jacobian matrix evaluation from a fit procedure.
    n_pars: int
        Total number of the fittef parameters in the model. It is related to
        the degrees of freedom.

    Returns
    -------
    cov : ndarray
        Variance-Covariance matrix (n x p).
    corr : ndarray
        Correlation matrix (n x p).
    """

    m = res.size  # number of data points used in the fit
    p = m - n_pars  # degrees of freedom

    # Variance-Covarince matrix
    cov = np.dot(res.T, res) / p * np.linalg.inv(np.dot(jac.T, jac))

    # Estimated error variance (sigma ** 2)
    sigmas2 = np.diag(np.diag(cov))

    D = np.linalg.inv(np.sqrt(sigmas2))  # inv diagonal variance matrix

    # Correlation matrix
    corr = np.dot(np.dot(D, cov), D)

    return cov, corr


def linear_equation(P1, P2, x):
    """
    Computes the linear equation solution for the y values given the x data
    points. P1 = (x1, y1) and P2 = (x2, y2).

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
