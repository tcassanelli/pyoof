.. pyoof-aperture:

:tocdepth: 2

***************************
Aperture (`pyoof.aperture`)
***************************

.. currentmodule:: pyoof.aperture

Introduction
============

The `~pyoof.aperture` sub-package contains related distributions/functions to the aperture distribution, :math:`\underline{E_\text{a}}(x, y)`. The aperture distribution is a two dimensional complex distribution, hence has an amplitude and a phase. The amplitude is represented by the blockage distribution, :math:`B(x, y)`, and the illumination function, :math:`E_\text{a}(x, y)`. The phase is given by the aperture phase distribution, :math:`\varphi(x, y)` and the optical path difference (OPD) function, :math:`\delta(x,y;d_z)`.
The collection of all distributions/functions for the aperture distribution is then,

.. math::
    \underline{E_\text{a}}(x, y) = B(x, y)\cdot E_\text{a}(x, y) \cdot \mathrm{e}^{\mathrm{i} \{\varphi(x, y) + \frac{2\pi}{\lambda}\delta(x,y;d_z)\}}.

Between them, the blockage distribution and OPD function are strictly dependent on the telescope geometry, which is the reason why they are gathered in the `~pyoof.telgeometry` sub-package.

.. note::
    All mentioned Python functions are for a two dimensional analysis and they require as an input meshed values. They can also be used in one dimensional analysis by adding ``y=0`` to each of the Python functions. Although their analysis in the `~pyoof` package is strictly two dimensional.

To generate meshed values simply use the `~numpy.meshgrid` routine.
    >>> import numpy as np
    >>> x = np.linspace(-50, 50, 1e3)
    >>> xx, yy = np.meshgrid(x, x)  # values in a grid

.. note::
    A Jupyter notebook tutorial for the `~pyoof.aperture` sub-package is provided in the `aperture.ipynb <https://github.com/tcassanelli/pyoof/blob/master/notebooks/aperture.ipynb>`_ on the GitHub repository.

Lastly there is another distribution left which is the (field) radiation pattern, :math:`F(u, v)`, which is the direct Fourier Transform (in two dimensions) of the aperture distribution (both of them being complex numbers).

.. math::
    F(u, v) = \mathcal{F}\left[\underline{E_\text{a}}(x, y)\right]

The (field) radiation pattern is the most extensive function in this sub-package. The standard Python package `~numpy.fft.fft2` is used to compute the FFT in two dimensions.

Using `~pyoof.aperture`
=======================

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

    ax.imshow(Ea, extent=[-pr, pr] * 2, cmap='viridis')
    ax.set_xlabel('$x$ m')
    ax.set_ylabel('$y$ m')
    ax.set_title('Illumination function')


The `~pyoof.aperture` only uses standard Python libraries, but what needs special consideration are the Python functions with the parameter ``K_coeff``, coming up.

Wavefront (aberration) distribution :math:`W(x, y)`
---------------------------------------------------

The wavefront (aberration) distribution, :math:`W(x, y)`, is strictly related to the aperture phase distribution (see Jupyter notebook, `zernike.ipynb <https://github.com/tcassanelli/pyoof/blob/master/notebooks/zernike.ipynb>`_ on GitHub), and it is the basis of the nonlinear least squares minimization done by the `~pyoof` package.

.. note::
    The Zernike circle coefficients, given by ``K_coeff`` are the basic structure for the aperture phase distribution. It is important to note that for an order :math:`n` of the polynomial there are :math:`(n+1)(n+2)/2` total number of polynomials. See `~pyoof.zernike`.

One basic example is to plot :math:`W(x, y)` with a random set of Zernike circle polynomial coefficients.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyoof import aperture, cart2pol

    pr = 50  # primary relfector m
    n = 10  # order polynomial
    N_K_coeff = (n + 1) * (n + 2) // 2  # max polynomial number
    K_coeff = np.random.normal(0., .1, N_K_coeff)

    # Generating the mesh
    x = np.linspace(-pr, pr, 1e3)
    xx, yy = np.meshgrid(x, x)
    r, t = cart2pol(xx, yy)
    r_norm = r / pr  # polynomials uner unitary circle

    W = aperture.wavefront(rho=r_norm, theta=t, K_coeff=K_coeff)
    W[xx ** 2 + yy ** 2 > pr ** 2] = 0

    fig, ax = plt.subplots()
    ax.contour(xx, yy, W, colors='k', alpha=0.3)
    ax.imshow(W, extent=[-pr, pr] * 2, origin='lower', cmap='viridis')

    ax.set_title('Wavefront (aberration) distribution')
    ax.set_ylabel('$y$ m')
    ax.set_xlabel('$x$ m')


Aperture phase distribution :math:`\varphi(x, y)`
-------------------------------------------------

The calculation of the aperture phase distribution, `~pyoof.aperture.phase`, follows the same guidelines as the wavefront (aberration) distribution, `~pyoof.aperture.wavefront`. In general, the problem will only focus on the aperture phase distribution, and not on the wavefront (aberration) distribution. To compute the phase simply,

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyoof import aperture

    pr = 50  # primary relfector m
    n = 5  # order polynomial
    N_K_coeff = (n + 1) * (n + 2) // 2  # max polynomial number
    K_coeff = np.random.normal(0., .1, N_K_coeff)

    x, y, phi_notilt = aperture.phase(K_coeff=K_coeff, notilt=True, pr=pr)
    phi_tilt = aperture.phase(K_coeff=K_coeff, notilt=False, pr=pr)[2]

    levels = np.linspace(-2, 2, 9)

    fig, ax = plt.subplots(ncols=2)
    #fig.subplots_adjust(wspace=0.6)

    for data, i in zip([phi_notilt, phi_tilt], range(2)):
        ax[i].imshow(
            data, extent=[-pr, pr] * 2, origin='lower', cmap='viridis'
            )
        ax[i].contour(x, y, data, levels=levels, alpha=0.3, colors='k')
        ax[i].set_xlabel('$x$ m')
        ax[i].set_ylabel('$y$ m')

    ax[0].set_title('$\\varphi(x, y)$ no-tilt')
    ax[1].set_title('$\\varphi(x, y)$')


To study the aberration in the aperture phase distribution, first it is necessary to remove some telescope effects. These are the tilt terms and are related to the telescope's pointing and become irrelevant. The tilt terms also represent the average slope in the :math:`x` and :math:`y` directions. In the Zernike circle polynomials the tilt terms are :math:`K^1_1` and :math:`K^{-1}_1`. To erase their dependence they are set to zero.

Aperture distribution :math:`\underline{E_\text{a}}(x, y)`
----------------------------------------------------------

To compute the aperture distribution, two extra functions from the `~pyoof.telgeometry` package are required. Assuming the reader knows them, it is easy to compute :math:`\underline{E_\text{a}}(x, y)`,

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyoof import aperture, telgeometry

    pr = 50  # primary relfector m
    n = 5  # order polynomial
    N_K_coeff = (n + 1) * (n + 2) // 2  # max polynomial number
    K_coeff = np.random.normal(0., .1, N_K_coeff)

    taper = np.random.randint(-20, -8)  # random illumination taper
    I_coeff = [1, taper, 0, 0]

    # Generating the mesh
    x = np.linspace(-pr, pr, 1e3)
    xx, yy = np.meshgrid(x, x)

    # For these functions see telgeometry sub-package
    telgeo = [telgeometry.block_effelsberg, telgeometry.opd_effelsberg, pr]

    Ea = []
    for d_z in [-.022, 0, .022]:

        Ea.append(
            aperture.aperture(
                x=xx, y=yy,
                K_coeff=K_coeff,
                I_coeff=I_coeff,
                d_z=d_z,  # radial offset
                wavel=0.009,  # observation wavelength
                illum_func=aperture.illum_pedestal,
                telgeo=telgeo  # see telgeometry sub-package
                )
            )

    extent = [-pr, pr] * 2

    fig, axes = plt.subplots(ncols=3, nrows=2)
    #fig.subplots_adjust(hspace=0.05, wspace=0.8)

    ax = axes.flat

    for i in range(3):
        ax[i].imshow(Ea[i].real, cmap='viridis', origin='lower', extent=extent)
        ax[i + 3].imshow(
            Ea[i].imag, cmap='viridis', origin='lower', extent=extent
            )
        ax[i].contour(xx, yy, Ea[i].real, cmap='viridis')
        ax[i + 3].contour(xx, yy, Ea[i].imag, cmap='viridis')

    ax[0].set_title('Aper real $d_z^-$')
    ax[1].set_title('Aper real $d_z$')
    ax[2].set_title('Aper real $d_z^+$')
    ax[3].set_title('Aper imag $d_z^-$')
    ax[4].set_title('Aper imag $d_z$')
    ax[5].set_title('Aper imag $d_z^+$')

    # Turn off tick labels
    for _ax in ax:
        _ax.set_yticklabels([])
        _ax.set_xticklabels([])


As mentioned before the aperture distribution is complex, which also depends on the radial offset added to defocus the telescope. Depending on that its shape in the real and imaginary parts will change. In general, the aperture distribution will not be used for the OOF holography study, only the power pattern and phase error will be used for visual inspection.

In contrast the (field) radiation pattern, has the same inputs, except for the `~numpy.fft.fft2` routine, which requires two more important parameters. These are ``resolution`` and ``box_factor``. Hence, it is simply executed by,

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pyoof import aperture, telgeometry

    pr = 50  # primary relfector m
    n = 5  # order polynomial
    N_K_coeff = (n + 1) * (n + 2) // 2  # max polynomial number
    K_coeff = np.random.normal(0., .1, N_K_coeff)

    taper = np.random.randint(-20, -8)  # random illumination taper
    I_coeff = [1, taper, 0, 0]

    telgeo = [telgeometry.block_effelsberg, telgeometry.opd_effelsberg, pr]

    F = aperture.radiation_pattern(
        K_coeff=K_coeff,
        I_coeff=I_coeff,
        d_z=0,  # in the cm order
        wavel=0.009,  # observation wavelength
        illum_func=aperture.illum_pedestal,
        telgeo=telgeo,
        resolution=2 ** 8,
        box_factor=5 * pr
        )


See Also
========

* `Antenna aperture <https://en.wikipedia.org/wiki/Antenna_aperture>`_

Reference/API
=============

.. automodapi:: pyoof.aperture
