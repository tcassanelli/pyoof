.. pyoof-aperture:

***************************
Aperture (`pyoof.aperture`)
***************************

.. currentmodule:: pyoof.aperture

Introduction
============

The `~pyoof.aperture` sub-package contains related distributions/functions to the aperture distribution, :math:`\underline{E_\text{a}}(x, y)`. The aperture distribution is a two dimension complex distribution, hence has an amplitude and phase. The amplitude are represented by the blockage distribution, :math:`B(x, y)`, and the illumination function, :math:`E_\text{a}(x, y)`. The phase is given by the phase aperture distribution, :math:`\varphi(x, y)` and the optical path difference function, :math:`\delta(x,y;d_z)`.
The collection of all distribution/functions for the aperture distribution is then,

.. math::
    \underline{E_\text{a}}(x, y) = B(x, y)\cdot E_\text{a}(x, y) \cdot \mathrm{e}^{\mathrm{i} \{\varphi(x, y) + \frac{2\pi}{\lambda}\delta(x,y;d_z)\}}.

Some of the presented functions are strictly dependent on the telescope geometry, therefore are separated in the sub-package.
Besides this, there are some other important function related to the aperture distribution, these are its Fourier Transform (field radiation pattern) and the root-mean-squared value for other applications.

.. note::
    All mentioned Python functions are for a two dimensional analysis and they require as an input meshed values. They can also be used in one dimensional analysis by adding ``y=0`` to each of the Python functions. Although their analysis in the `pyoof` package is strictly two dimensional.

To generate meshed values simply use the `~numpy.meshgrid` routine.
    >>> import numpy as np
    >>> x = np.linspace(-50, 50, 1e3)
    >>> xx, yy = np.meshgrid(x, x)  # values in a grid

.. note::
    A Jupyter notebook tutorial for the `~pyoof.aperture` sub-package is provided in the `aperture.ipynb <https://github.com/tcassanelli/pyoof/blob/master/notebooks/aperture.ipynb>`_ on GitHub.

The most extended function on this sub-package corresponds to the (field) radiation pattern, :math:`F(u, v)` which corresponds to the Fast Fourier Transform (FFT) in two dimensions for the aperture distribution, :math:`\underline{E_\text{a}}(x, y)`.
The FFT is computed using the standard Python package `~numpy.fft.fft2`.

Using `pyoof.aperture`
======================

Using the `~pyoof.aperture` package is really simple, for example the illumination function, :math:`E_\text{a}(x, y)`,

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyoof import aperture

    pr = 50  # primary relfector m
    x = np.linspace(-pr, pr, 1e3)
    xx, yy = np.meshgrid(x, x)

    I_coeff = [1, -14, 0, 0]  # [amp, c_dB, x0, y0]

    Ea = aperture.illum_pedestal(xx, yy, I_coeff, pr)
    Ea[xx ** 2 + yy ** 2 > pr ** 2] = 0  # circle shape

    fig, ax = plt.subplots()
    ax.imshow(Ea, extent=[-pr, pr] * 2)
    ax.set_xlabel('$x$ m')
    ax.set_ylabel('$y$ m')
    ax.set_title('Illumination function')

It is only requires the standard Python libraries to use `~pyoof.aperture`. What needs special consideration are the Python functions with the parameter ``K_coeff``.

Wavefront (aberration) distribution
-----------------------------------
The wavefront (aberration) distribution, :math:`W(x, y)`, is strictly related to the aperture phase distribution (see Jupyter notebook, `zernike.ipynb <https://github.com/tcassanelli/pyoof/blob/master/notebooks/zernike.ipynb>`_ on GitHub), and it is the base of the nonlinear least squares minimization done by the `pyoof` package.

.. warning::
    The Zernike circle coefficients, given by ``K_coeff`` are the basic structure for the aperture phase distribution, :math:`\varphi(x, y)=2\pi W(x, y)` and they are usually between :math:`[-2, 2]`. Besides this they have a fixed number for the polynomial order, :math:`\frac{(n+1)(n+2)}{2}`, other number in ``K_coeff`` will rise an error.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyoof import aperture

    pr = 50  # primary relfector m
    n = 5  # order polynomial
    N_K_coeff = (n + 1) * (n + 2) // 2  # max polynomial number
    K_coeff = np.random.normal(0., .1, N_K_coeff)

    x, y, phi = aperture.phase(K_coeff, True, pr)

    levels = np.linspace(-2, 2, 9)  # half-radian interval

    fig, ax = plt.subplots()
    ax.contour(x, y, phi, levels=levels, colors='k', alpha=0.3)
    ax.imshow(phi, extent=[-pr, pr] * 2, origin='lower')

    ax.set_title('Aperture phase distribution')
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')

See Also
========

* `Antenna aperture <https://en.wikipedia.org/wiki/Antenna_aperture>`_
* `Zernike circle polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_

Reference/API
=============

.. automodapi:: pyoof.aperture
