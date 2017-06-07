#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import numpy as np


__all__ = [
    'cart2pol', 'wavevector_to_degree', 'par_variance',
    ]


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

    rho = np.sqrt(x ** 2 + y ** 2)  # radius normalization
    theta = np.arctan2(y, x)

    return rho, theta


def wavevector_to_degree(x, wavel):
    """
    Converst wave-vector 1/m to degrees.

    Parameters
    ----------
    x : ndarray
        Wave-vector, result from FFT2.
    wavel : ndarray
        Wavelength.
    Returns
    -------
    wave_vector_degrees : ndarray
        Wave-vector in degrees.
    """

    wave_vector_degrees = np.degrees(x * wavel)

    return wave_vector_degrees


def par_variance(res, jac, n_pars):
    # Covariance and correlation matrices
    m = res.size
    d_free = m - n_pars  # degrees of freedom

    # Covarince matrix
    cov = np.dot(res.T, res) / d_free * np.linalg.inv(np.dot(jac.T, jac))

    sigmas2 = np.diag(np.diag(cov))  # sigma ** 2
    D = np.linalg.inv(np.sqrt(sigmas2))  # inv diagonal variance matrix

    # Correlation matrix
    corr = np.dot(np.dot(D, cov), D)

    return cov, corr
