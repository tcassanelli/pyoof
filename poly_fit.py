import numpy as np
from math import factorial as f

# Generating a function to fit a polynomial in 2D.


def gen_poly(params, x, y):
    """
    Generates a polynomial in 2-dim given the parameters.

    Parameters
    ----------
    params : ndarray
        Give one/two dim ndarray for each parameter that gos between x & y.
    x : ndarray
        Grid value for the x variable, same as the contour plot.
    y : ndarray
        Grid value for the x variable, same as the contour plot.

    Returns
    -------
    polynomial: ndarray
        Gives one/two dimentional array
    """
    polynomial = sum(
        params[i, j] * x ** i * y ** j
        for i in range(params.shape[0])
        for j in range(params.shape[1])
        )
    return polynomial


def build_X(x, y, param_shape):
    """
    Builder for the X matrix, or the coefficient matrix to be used in least
    squated method.

    Parameters
    ----------
    param_shape : list
        Parameter shape of the desired polynomial (params.shape).
    x : ndarray
        Grid value for the x variable, this must be flattened.
    y : ndarray
        Grid value for the x variable, this must be flattened.

    Returns
    -------
    X : ndarray
        Gives the coefficient matrix ready for the least squared method.
    """
    X = np.column_stack(
        x ** i * y ** j
        for i in range(param_shape[0])
        for j in range(param_shape[1])
        )
    return X


def U(l, n, theta, rho):
    """
    Zernike polynomial generator. l, n are intergers, n >= 0 and n - |l| even.
    Expansion of a complete set of orthonormal polynomials in a unitary circle,
    for the aberration function.
    The n value determines the total amount of polynomials 0.5(n+1)(n+2).

    Parameters
    ----------
    l : int
        Can be positive or negative, relative to angle component.
    n : int
        It is n >= 0. Relative to radial component.
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).

    Returns
    -------
    U : ndarray
        Zernile polynomail already evaluated.
    """

    m = abs(l)
    a = int((n + m) / 2)
    b = int((n - m) / 2)

    R = sum(
    (-1) ** s * f(n - s) * rho ** (n - 2 * s) / (f(s) * f(a - s) * f(b - s))
    for s in range(0, b + 1)
    )

    if l < 0:
        U = R * np.sin(m * theta)
    else:
        U = R * np.cos(m * theta)

    return U


def gen_polyU(theta, rho, params, n=6):
    """
    Generates a series of Zernike polynomials, the aberration function.

    Parameters
    ----------
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).
    params : ndarray
        Constants organized by the ln list, which gives the possible values.
    n : int
        It is n >= 0. Determines the size of the polynomial, see ln.
    Returns
    -------
    polyU : ndarray
        Zernile polynomail already evaluated and multiplied by its parameter
        or constant.
    """
    # List which contains the allowed values for the U function.
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    polyU = sum(
        params[i] * U(L[i], N[i], theta, rho)
        for i in range(params.size)
        )

    return polyU


def build_UX(theta, rho, params, n=6):
    """
    Coefficient matrix ready to use in the least squared method and calculate
    the fit parameters.

    Parameters
    ----------
    theta : ndarray
        Values for the angular component. theta = np.arctan(y / x).
    rho : ndarray
        Values for the radial component. rho = np.sqrt(x ** 2 + y ** 2).
    params : ndarray
        Constants organized by the ln list, which gives the possible values.
    n : int
        It is n >= 0. Determines the size of the polynomial, see ln.
    Returns
    -------
    X : ndarray
        Coefficient matrix.
    """

    # List which contains the allowed values for the U function.
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    X = np.column_stack(
        U(L[i], N[i], theta, rho)
        for i in range(params.size)
        )

    return X


# Test functions

if False:

    import matplotlib.pyplot as plt

    params = np.array([[2, 4, 5], [5, 3, 2]])
    x_lin = np.linspace(-10, 10, 100)
    y_lin = np.linspace(-5, 5, 200)

    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    data2d = gen_poly(params, x_grid, y_grid)  # Ready for contour plot

    data2d_noisy = data2d + np.random.normal(0, 3., data2d.shape)

    # Making the fit, remember to flatten data
    x_lins, y_lins = x_grid.flatten(), y_grid.flatten()
    data2d_flat = data2d_noisy.flatten()

    X = build_X(x_lins, y_lins, params.shape)  # Coeff Matrix

    result = np.linalg.lstsq(X, data2d_flat)
    params_fit = result[0].reshape(params.shape)

    fitted_data2d = gen_poly(params_fit, x_grid, y_grid)

    fig, ax = plt.subplots(figsize=(14, 5), ncols=3)
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.96, top=0.9, wspace=0.28
        , hspace=0.2)

    # Plot for the three cases!
    ax[0].set_title('2D Polynomial Real')
    cax = ax[0].contourf(x_grid, y_grid, data2d)
    ax[0].set_xlabel('$X$')
    ax[0].set_ylabel('$Y$')

    ax[1].set_title('2D Polynomial Data')
    cax_ = ax[1].contourf(x_grid, y_grid, data2d_noisy)
    ax[1].set_xlabel('$X$')
    ax[1].set_ylabel('$Y$')

    ax[2].set_title('2D Polynomial Fit')
    cax_ = ax[2].contourf(x_grid, y_grid, fitted_data2d)
    ax[2].set_xlabel('$X$')
    ax[2].set_ylabel('$Y$')

    plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    from texttable import Texttable

    theta = np.radians(np.linspace(0, 360, 20))
    rho = np.linspace(0, 1, 20)

    t_grid, r_grid = np.meshgrid(theta, rho)
    t_lins, r_lins = t_grid.flatten(), r_grid.flatten()

    params = np.array([0, 0, 0, 0, 0, 1, 0])
    data2d = gen_polyU(t_grid, r_grid, params)
    data2d_noisy = data2d + np.random.normal(0, 0.3, data2d.shape)

    data2d_flat = data2d_noisy.flatten()

    X = build_UX(t_lins, r_lins, params)  # Coeff Matrix
    result = np.linalg.lstsq(X, data2d_flat)
    params_fit = result[0]

    data2d_fit = gen_polyU(t_grid, r_grid, params_fit)

    # Making nice table :)
    tab = Texttable()
    tab.add_rows([['Poly. Term', 'Given Coeff.', 'Fit Coeff.']])

    n = 6
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0][:len(params)]
    N = np.array(ln)[:, 1][:len(params)]

    for i in range(len(params)):
        tab.add_row(['U(' + str(L[i]) + ',' + str(N[i]) + ')', params[i], params_fit[i]])

    print(tab.draw())

    fig, ax = plt.subplots(figsize=(14, 5), ncols=3,
        subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.96, top=0.9, wspace=0.28
        , hspace=0.2)

    # Plot for the three cases!
    ax[0].set_title('2D Polynomial Real')
    cax0 = ax[0].contourf(t_grid, r_grid, data2d)
    ax[1].set_title('2D Polynomial Data')
    cax1 = ax[1].contourf(t_grid, r_grid, data2d_noisy)
    ax[2].set_title('2D Polynomial Fit')
    cax1 = ax[2].contourf(t_grid, r_grid, data2d_fit)

    plt.show()
