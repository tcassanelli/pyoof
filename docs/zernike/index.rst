.. pyoof-zernike:

:tocdepth: 2

********************************************
Zernike circle polynomials (`pyoof.zernike`)
********************************************

.. currentmodule:: pyoof.zernike

Introduction
============

The Zernike circle polynomials [Virendra]_ were introduced by `Frits Zernike <https://en.wikipedia.org/wiki/Frits_Zernike>`_ (winner Nobel prize in physics 1953), for testing his phase contrast method in circular mirror figures. The polynomials were used by Ben Nijboer to study the effects of small aberrations on diffracted images with a rotationally symmetric origin on circular pupils.

The studies on the wavefront (aberration) distribution, have been classically described as a power series, where it is possible to expand it in terms of `orthonormal polynomials in the unitary circle`. Although, there is a wide range of polynomials that fulfill this condition, the Zernike circle polynomials have certain properties of invariance that make them unique.

Mathematical definitions
========================

There are no conventions assigned on how to enumerate the Zernike circle polynomials. In the `~pyoof` package uses the same letters and conventions from [Born]_. The Zernike circle polynomials can be expressed as a multiplication of radial and angular sections. The radial section it is called radial polynomials, and its generating function is given by,

.. math::

    R^{\pm m}_n (\varrho) = \frac{1}{\left(\frac{n-m}{2}\right)!\cdot\varrho^m}\left\{\frac{\text{d}}{\text{d}\left(\varrho^2\right)} \right\}^{\frac{n-m}{2}} \left\{  \left( \varrho^2 \right)^{\frac{n+m}{2}} \cdot  \left( \varrho^2 -1 \right)^{\frac{n-m}{2}} \right\}.

The above expression only works when :math:`n-m` is even, otherwise it vanishes. Then the Zernike circle polynomials, :math:`U^\ell_n(\varrho, \vartheta)`, are,

.. math::
    U^\ell_n(\varrho, \vartheta) = R^m_n(\varrho) \cdot \cos m\vartheta \qquad \ell \geq 0, \\
    U^\ell_n(\varrho, \vartheta) = R^m_n(\varrho) \cdot \sin m\vartheta \qquad \ell < 0.

The angular dependence is given by :math:`\ell`, and :math:`m=\ell`, and the order of the polynomial is given by :math:`n`. Low orders of the Zernike circle polynomials are closely related to classical aberrations. Up to order :math:`n=4` there are 15 polynomials, this is given by the formula,

.. math::
    \frac{(n+1)(n+2)}{2}.

The following table shows the Zernike circle polynomials up to order four.

================= ==================== ================================================= ======
`Order` :math:`n` `Angle` :math:`\ell` `Polynomial` :math:`U^\ell_n(\varrho, \vartheta)` `Name`
================= ==================== ================================================= ======
:math:`0`         :math:`0`            :math:`1`                                         Piston
:math:`1`         :math:`-1`           :math:`\varrho\sin\vartheta`                      :math:`y`-tilt
:math:`1`         :math:`1`            :math:`\varrho\cos\vartheta`                      :math:`x`-tilt
:math:`2`         :math:`-2`           :math:`\varrho^2\sin2\vartheta`
:math:`2`         :math:`0`            :math:`2\varrho^2-1`                              Defocus
:math:`2`         :math:`2`            :math:`\varrho^2\cos2\vartheta`                   Astigmatism
:math:`3`         :math:`-3`           :math:`\varrho^3\sin3\vartheta`
:math:`3`         :math:`-1`           :math:`(3\varrho^3-2\varrho)\sin\vartheta`        Primary :math:`y`-coma
:math:`3`         :math:`1`            :math:`(3\varrho^3-2\varrho)\cos\vartheta`        Primary :math:`x`-coma
:math:`3`         :math:`3`            :math:`\varrho^3\cos3\vartheta`
:math:`4`         :math:`-4`           :math:`\varrho^4\cos4\vartheta`
:math:`4`         :math:`-2`           :math:`(4\varrho^4-3\varrho^2)\sin2\vartheta`
:math:`4`         :math:`0`            :math:`6\varrho^4-6\varrho^2+1`                   Primary spherical
:math:`4`         :math:`2`            :math:`(4\varrho^4-3\varrho^2)\cos2\vartheta`     Second astigmatism
:math:`4`         :math:`4`            :math:`\varrho^4\cos4\vartheta`
================= ==================== ================================================= ======

The Zernike circle polynomials can be used to represent, in a convenient way, the wavefront (aberration) distribution, :math:`W(x, y)`, [Wyant]_. It is convenient, because low orders are related to classical aberrations (from optic physics), as well as their lower orders are able to represent mid- to low-resolution aperture phase distribution maps, :math:`\varphi(x, y)`. The general expression concerning them is given by,

.. math::
    \varphi(x, y) = 2\pi \cdot W(x, y) = 2\pi \cdot \sum_{n, \ell} K_{n\ell}U^\ell_n(\varrho, \vartheta),

where :math:`K_{n\ell}` are the Zernike circle polynomial coefficients. The final output from the `~pyoof` package is to find such coefficients and then make a representation of the aberration in the (telescope) aperture plane. The wavefront (aberration) distribution, `~pyoof.aperture.wavefront`, is a function listed in the `~pyoof.aperture` sub-package.
The order of the :math:`K_{n\ell}` coefficients will vary from the order of the polynomials. Commonly for the `~pyoof` package their values are between :math:`[-2, 2]`.

.. warning::
    The order of magnitude of the Zernike circle polynomial coefficients (:math:`K_{n\ell}`) will vary on what conventions are used to generate them. There are some conventions that require a normalization constant.

Using `~pyoof.zernike`
======================

To use the polynomials from the `~pyoof.zernike` is really easy. First import the sub-package and start using the default structure::

    >>> import numpy as np
    >>> from pyoof import zernike  # calling the sub-package

    >>> rho = np.linspace(0, 1, 10)
    >>> R40 = zernike.R(n=4, m=0, rho=rho)
    >>> R40
    array([ 1.        ,  0.92684042,  0.71833562,  0.40740741,  0.04892547,
           -0.28029264, -0.48148148, -0.43392775,  0.00502972,  1.        ])

To plot the polynomials first it is required to construct a grid.

.. plot::
    :include-source:

    import matplotlib.pyplot
    import numpy as np
    from pyoof import zernike, cart2pol

    radius = 1  # m
    x = np.linspace(-radius, radius, 1e3)
    xx, yy = np.meshgrid(x, x)
    rho, theta = cart2pol(xx, yy)
    rho_norm = rho / radius  # polynomials only work in the unitary circle

    U31 = zernike.U(n=3, l=1, rho=rho_norm, theta=theta)
    extent = [x.min(), x.max()] * 2

    # restricting to a circle shape
    U31[xx ** 2 + yy ** 2 > radius ** 2] = 0

    fig, ax = plt.subplots()
    ax.imshow(U31, extent=extent, origin='lower', cmap='viridis')
    ax.contour(xx, yy, U31, cmap='viridis')
    ax.set_title('Primary $x$-coma $U^1_3(\\varrho, \\ell)$')
    ax.set_xlabel('$x$ m')
    ax.set_ylabel('$y$ m')


.. note::
    At the time of using the function `~pyoof.zernike.U` make sure that the radius is normalized by its maximum. The Zernike circle polynomials are only orthonormal under the unitary circle. It is to avoid the use of extra constants, use :math:`\varrho / \varrho_\text{max}`.

For a more in-depth example of their usage go to the Jupyter notebook `zernike.ipynb <https://github.com/tcassanelli/pyoof/blob/master/notebooks/zernike.ipynb>`_.


An example of wavefront (aberration) distribution could be the following,

.. math::

    W(x, y) = U^0_0 + \frac{1}{10}\cdot U^{-1}_1 + \frac{1}{5} \cdot U^{2}_4

Then a plot of such function will be,

.. plot::
    :include-source:

    import matplotlib.pyplot
    import numpy as np
    from pyoof import zernike, cart2pol

    radius = 1  # m
    x = np.linspace(-radius, radius, 1e3)
    xx, yy = np.meshgrid(x, x)
    rho, theta = cart2pol(xx, yy)
    rho_norm = rho / radius  # polynomials only work in the unitary circle

    K_coeff = [1/10, 1 / 5, 1/ 5]
    _U = [
        zernike.U(n=0, l=0, rho=rho_norm, theta=theta),
        zernike.U(n=1, l=-1, rho=rho_norm, theta=theta),
        zernike.U(n=4, l=2, rho=rho_norm, theta=theta)
        ]

    W = sum(K_coeff[i] * _U[i] for i in range(3))  # wavefront distribution
    extent = [x.min(), x.max()] * 2

    # restricting to a circle shape
    W[xx ** 2 + yy ** 2 > radius ** 2] = 0

    fig, ax = plt.subplots()
    ax.imshow(W, extent=extent, origin='lower', cmap='viridis')
    ax.contour(xx, yy, W, cmap='viridis')
    ax.set_title('Wavefront (aberration) distribution')
    ax.set_xlabel('$x$ m')
    ax.set_ylabel('$y$ m')


References
==========

.. [Born] Born, M. and Wolf, E., 1959. Principles of Optics: Electromagnetic Theory of Propagation, Interference and Diffraction of Light. Pergamon Press.

.. [Virendra] Virendra N. Mahajan, "Zernike Circle Polynomials and Optical Aberrations of Systems with Circular Pupils," Appl. Opt. 33, 8121-8124 (1994)

.. [Wyant] Wyant, James C., and Katherine Creath. "Basic Wavefront Aberration Theory for Optical Metrology." (1992).

Reference/API
=============

.. automodapi:: pyoof.zernike
