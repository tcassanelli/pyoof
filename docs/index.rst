
:tocdepth: 2

*******************
pyoof Documentation
*******************

Introduction (`pyoof`)
======================

Welcome to the `~pyoof` documentation. `~pyoof` is a Python package which computes out-of-focus (OOF) holography, for beam maps of a single-dish radio telescope. The method was developed by `B. Nikolic et al <https://www.aanda.org/articles/aa/ps/2007/14/aa5603-06.ps.gz>`_. The OOF holography is a phase-retrieval holography procedure used to find the aperture phase distribution, :math:`\varphi(x, y)`, (or simply the phase error) and the associated errors on a telescope's surface (primary dish). The main advantage of this method, over the traditional with-phase holography, is that it does not require additional equipment to perform observations and it can be used for a wide elevation range. These two allow OOF holography to study and model gravitational deformations, the most well behaved and prominent source of deformation (other sources could also be thermal and wind deformations which are non-repeatable), on the telescope's primary dish.

The method requires the use of a compact source (point-like), with a good signal-to-noise (:math:`\geq200`) and root-mean-squared noise (:math:`\geq 14 \, \mathrm{mJy}`). Then a set of continuum observations (on-the-fly mapping) are required, two of them out-of-focus and one in-focus. The OOF observations are performed by adding a known radial offset (:math:`d_z`), of the order of centimeters, to the telescope's sub-reflector.
The defocused terms are needed to break the degeneracy between the power pattern (observed quantity :math:`P(u, v)`), and the aperture distribution, :math:`\underline{E_\text{a}}(x, y)`, without OOF observations the problem becomes under-determined. Such a relation is given by,

.. math::
    P(u, v) = \left| \mathcal{F}\left[\underline{E_\text{a}}(x, y) \right]  \right|^2.
    :label: power-pattern-definition

The aperture phase distribution, related to the power pattern (observed beam maps), can be expressed as a sum of weighted polynomials. This is,

.. math::
    \varphi(x, y) = 2\pi \cdot W(x, y) = 2\pi \cdot \sum_{n, \ell} K_{n\ell}U^\ell_n(\varrho, \vartheta).
    :label: phase-error-definition

The parametrization of the aperture phase distribution allows its construction by using a nonlinear least squares minimization (`~pyoof.fit_beam`), due to the highly nonlinear relation between the aperture distribution and the power pattern (degenerated). The `~pyoof` package takes as an input the observed beam maps and computes the :math:`K_{n\ell}` coefficients to find the aperture phase distribution (on the primary reflector).

Among the basic operations that the `~pyoof` package does, there are: construction aperture distribution (`~pyoof.aperture.aperture`), calculation of its Fast Fourier Transform in two dimensions (`~pyoof.aperture.radiation_pattern`), calculation of a residual between observed and modeled power pattern, and nonlinear least squares minimization (`~pyoof.fit_beam`).

OOF holography parameters
=========================

The OOF holography makes use of several constants, distributions and functions, and some of them are related to each other. Please keep in mind their mathematical symbols and what they represent.

Aperture distribution: :math:`\underline{E_\text{a}}(x, y)` (`~pyoof.aperture.aperture`)
---------------------------------------------------------------------------------------------------------

Collection of sub-functions and -distributions. It is the point of connection between the observational data and the theoretical model. See Eq. :eq:`power-pattern-definition`. It is defined as,

.. math::
    \underline{E_\text{a}}(x, y) = B(x, y)\cdot E_\text{a}(x, y) \cdot \mathrm{e}^{\mathrm{i} \{\varphi(x, y) + \frac{2\pi}{\lambda}\delta(x,y;d_z)\}}.

Aperture phase distribution (phase error): :math:`\varphi(x, y)` (`~pyoof.aperture.phase`)
------------------------------------------------------------------------------------------------------

The aperture phase distribution represents the aberrations of an optical system, measured in radians. It is also related to the wavefront (aberration) distribution (see Eq. :eq:`phase-error-definition`), :math:`W(x, y)`, which is the classical function used in optics to define aberrations.

Blockage distribution: :math:`B(x, y)` (`~pyoof.telgeometry.block_manual`)
--------------------------------------------------------------------------

It is the truncation or blockage that the telescope structure (or a shade effect) does to the aperture plane. It is usually best represented as a two dimensional modified Heaviside step function. Since it depends on the geometry, it will depend on the telescope used.

(Field) radiation pattern: :math:`F(u, v)` (`~pyoof.aperture.radiation_pattern`)
--------------------------------------------------------------------------------

The (field) radiation pattern is a complex distribution, it is the direct Fourier Transform (in two dimensions) of the aperture distribution, :math:`F(u, v) = \mathcal{F}[\underline{E_\text{a}}(x, y)]`. Represents the angular variation of the radiation around the antenna.

Illumination function: :math:`E_\text{a}(x, y)` (`~pyoof.aperture.illum_pedestal` or `~pyoof.aperture.illum_gauss`)
------------------------------------------------------------------------------------------------------------------------------------------------

The illumination function is a characteristic of a receiver and it is used to reduce the level of the side lobes. It can be modeled with different types of functions, such as a Gaussian or a cosine. The default one used by the `~pyoof` package is the parabolic taper on a pedestal (order :math:`q=2`),

.. math::
    E_\text{a}(\rho') = C + (1 - C)\cdot \left[ 1-\left(\frac{\rho'}{R}\right)^2 \right]^q, \qquad C=10^{\frac{c_\text{dB}}{20}},

the `~pyoof` package includes the option of introducing a new illumination function.

Illumination coefficients (``I_coeff``)
---------------------------------------

The illumination coefficients correspond to four constants that need to be added to the illumination function, :math:`E_\text{a}`. These coefficients are the illumination amplitude, :math:`A_{E_\text{a}}`, the illumination offset (:math:`x_0, y_0`), and the illumination taper in decibels, :math:`c_\text{dB}` or :math:`\sigma_\text{dB}` (depending on the type of illumination function used). The coefficients, ``I_coeff``, are organized in the following order:

.. math::

    \left[A_{E_\text{a}}, c_\text{dB}, x_0, y_0 \right]^\intercal

From these coefficients the most relevant of them is the illumination taper. It gives a measure of how strong the signal is from the center of the aperture plane with respect to its edges.

Observation wavelength: :math:`\lambda` (``wavel``)
---------------------------------------------------

For one OOF holography observation it is required to perform two of the out-of-focus and one in-focus continuum scans. All of them need to be observed with the same receiver at a certain :math:`\lambda`. The wavelength is also important when calculating the aberrations in the primary dish.

Optical path difference (OPD) function: :math:`\delta(x, y;d_z)` (`~pyoof.telgeometry.opd_manual`)
--------------------------------------------------------------------------------------------------------------

The OPD function is the extra path that light has to travel every time a radial offset, :math:`d_z`, (known a priori) is added to the sub-reflector (to make OOF observations). The OPD function depends strictly on the telescope geometry.

Power pattern (beam map): :math:`P(u, v)` (``power_pattern``)
-------------------------------------------------------------

The power pattern or beam map is the observed quantity. For one OOF holography observation, it is required to perform two of the out-of-focus and one in-focus continuum scans (on-the-fly mapping). This will produce two beam maps with a clear interference pattern and one with the common in-focus beam size.

Radial offset: :math:`d_z` (``d_z``)
------------------------------------

The radial offset is the defocused term added to the sub-reflector. It is usually of the order of centimeters (it will vary from telescope to telescope). A small value of the :math:`d_z` may not add enough change to the out-of-focus beam with respect to the in-focus beam, increasing the degeneracy on the least squares minimization. On the contrary, a large value of :math:`d_z` will decrease the signal-to-noise on the source, making the Zernike circle polynomial coefficients have high uncertainties.
This value can only be set by observing and studying the output.

Wavefront (aberration) distribution: :math:`W(x, y)` (`~pyoof.aperture.wavefront`)
----------------------------------------------------------------------------------------------------------

The wavefront (aberration) distribution is the classical approach used to represent an aberrated wavefront. From optical physics it is known that :math:`W(x, y)` can be expressed as a sum of a convenient set of orthonormal polynomials. The Zernike circle polynomials, :math:`U^\ell_n(\varrho,\vartheta)`, fulfill the required mathematical conditions, see Eq. :eq:`phase-error-definition`.

Zernike circle polynomial: :math:`U^\ell_n(\varrho, \vartheta)` (`~pyoof.zernike.R` and `~pyoof.zernike.U`)
------------------------------------------------------------------------------------------------------------------------

The Zernike circle polynomials were introduced by Frits Zernike for testing his phase contrast method in circular mirror figures. The polynomials were used by Ben Nijboer to study the effects of small aberrations on diffracted images with a rotationally symmetric origin on circular pupils.

The polynomials can be separated in terms of their radial and angular components and their respective indices are: order :math:`n` and its angular dependence :math:`\ell`. The radial part of the polynomials, is given by,

.. math::

    R^{\pm m}_n (\varrho) = \frac{1}{\left(\frac{n-m}{2}\right)!\cdot\varrho^m}\left\{\frac{\text{d}}{\text{d}\left(\varrho^2\right)} \right\}^{\frac{n-m}{2}} \left\{  \left( \varrho^2 \right)^{\frac{n+m}{2}} \cdot  \left( \varrho^2 -1 \right)^{\frac{n-m}{2}} \right\}.

With :math:`m = |\ell|`. The complete Zernike circle polynomials are:

.. math::
    U^\ell_n(\varrho, \vartheta) = R^m_n(\varrho) \cdot \cos m\vartheta \qquad \ell \geq 0, \\
    U^\ell_n(\varrho, \vartheta) = R^m_n(\varrho) \cdot \sin m\vartheta \qquad \ell < 0.

For more about the Zernike circle polynomials and the `~pyoof` package see the Jupyter notebook `zernike.ipynb <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/zernike.ipynb>`_ examples from the repository.


Zernike circle polynomial coefficient: :math:`K_{n\, \ell}` (``K_coeff``)
-------------------------------------------------------------------------

The Zernike circle polynomial coefficients, ``K_coeff``, are the final parameters to be found by the least squares minimization (as well as ``I_coeff``). By finding them it is possible to reconstruct the aperture phase distribution, :math:`\varphi(x, y)`, see Eq. :eq:`phase-error-definition`. Depending on the polynomial order, :math:`n`, there will be a different number of polynomials. In general, they are gathered as follows,

.. math::
    \left[K_{0\,0}, K_{1\,-1}, K_{1\,0}, K_{1\,1}, \dotso , K_{n\, \ell}\right]^\intercal.

Getting Started
===============

.. toctree::
    :maxdepth: 2

    install
    Tutorials <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/>

User Documentation
==================

To learn the in-depth structure of the `~pyoof` package, first visit the sub-packages in the following order:

* The use of the Zernike circle polynomials, in `~pyoof.zernike`
* The telescope structure (geometry and blockage), in `~pyoof.telgeometry`
* Mathematical functions for the construction of the aperture distribution, in `~pyoof.aperture`

If there is not enough time for that I encourage you to read the following examples and test the `Jupyter notebooks <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/>`_.

The fits file
-------------

The `~pyoof` package requires a specific format for the fits files. The format must include observational parameters as well as data. These parameters can be seen in the `~pyoof.extract_data_pyoof` function.::

    >>> import pyoof
    >>> from pyoof import aperture, telgeometry
    >>> from astropy.io import fits
    >>> # Generally you would store the filename as string
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> oofh_data = get_pkg_data_filename('pyoof/data/example0.fits')
    >>> hdulist = fits.open(oofh_data)

After generating the data file, the main structure of the fits file has to be as follows::

    >>> hdulist.info()  # doctest: +REMOTE_DATA +IGNORE_OUTPUT
    Filename: pyoof/data/example0.fits
    No.    Name         Type      Cards   Dimensions   Format
    0    PRIMARY     PrimaryHDU      11   ()
    1    MINUS OOF   BinTableHDU     16   9409R x 3C   [E, E, E]
    2    ZERO OOF    BinTableHDU     16   9409R x 3C   [E, E, E]
    3    PLUS OOF    BinTableHDU     16   9409R x 3C   [E, E, E]

Where the data files are separated into ``MINUS OOF``, ``ZERO OOF`` and ``PLUS OOF`` out-of-focus observations. Each of them has to have the radial offset (:math:`d_z`) value which has to be written in the BinTableHDU header with the key ``DZ``. The BinTableHDU header is then::

    >>> hdulist[1].header  # minus file  # doctest: +REMOTE_DATA +IGNORE_OUTPUT
    XTENSION= 'BINTABLE'         / binary table extension
    BITPIX  =                    8 / array data type
    NAXIS   =                    2 / number of array dimensions
    NAXIS1  =                   12 / length of dimension 1
    NAXIS2  =                 9409 / length of dimension 2
    PCOUNT  =                    0 / number of group parameters
    GCOUNT  =                    1 / number of groups
    TFIELDS =                    3 / number of table field
    TTYPE1  = 'U       '
    TFORM1  = 'E       '
    TTYPE2  = 'V       '
    TFORM2  = 'E       '
    TTYPE3  = 'BEAM    '
    TFORM3  = 'E       '
    EXTNAME = 'MINUS OOF'
    DZ      =               -0.022

If ``hdulist[1].data`` is printed, then the ``MINUS OOF`` observation is selected.

    >>> hdulist[1].data  # doctest: +REMOTE_DATA +IGNORE_OUTPUT
    FITS_rec([(-0.00089586416, -0.00089586416, -1007.5198),
           (-0.00087720033, -0.00089586416, 262.23615),
           (-0.00085853651, -0.00089586416, -557.19135), ...,
           (0.00085853651, 0.00089586416, 419.55826),
           (0.00087720033, 0.00089586416, -53.516975),
           (0.00089586416, 0.00089586416, -110.75285)],
          dtype=(numpy.record, [('U', '<f4'), ('V', '<f4'), ('BEAM', '<f4')]))

Where ``U`` and ``V`` are the dimensions of the :math:`x`- and :math:`y`-axis, and the key ``BEAM`` (beam map) corresponds to the observed power pattern. The three of them are flat arrays. Hence, to reconstruct the plot in two dimensions an interpolation is required::

    >>> hdulist.close()

.. warning::
    Always remember to close the fits file after use, ``hdulist.close()``.

.. note::
    The fits format can also be avoided. If the required parameters are added to the `~pyoof.fit_beam` function, then the `~pyoof` package will also work. Although, it is recommended to use the fits format, for ease and clean storage of the data.

It is also possible to try the `~pyoof` by generating your own data, with the build-in function `~pyoof.beam_generator`. The Jupyter notebook `oof_holography.ipynb <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/oof_holography.ipynb>`_  has generated data, plus noise, and it is used as an input for the `~pyoof` package.

Using pyoof
-----------

The main function in the `~pyoof` package is `~pyoof.fit_beam`. This function does all the numerical computation necessary to find the Zernike circle coefficients and the illumination coefficients, stored in ``params_solution`` (`~numpy.ndarray`). To start, first the data needs to be extracted using `~pyoof.extract_data_pyoof`, then its outputs; constants, arrays and strings are given as input to the core function `~pyoof.fit_beam`. Besides the observational data some other parameters need to be added, these are the ones related to the FFT2 (`~numpy.fft.fft2`) and functions related to the telescope geometry (effective focal length, type of antenna, etc).

Following the same example as before, it is possible to make a simple preview of the file, using `~pyoof.extract_data_pyoof`.

.. code-block:: python

    import pyoof
    from scipy import interpolate
    import matplotlib.pyplot as plt
    # Generally you would store the filename as string
    from astropy.utils.data import get_pkg_data_filename
    oofh_data = get_pkg_data_filename('pyoof/data/example0.fits')

    data_info, data_obs = pyoof.extract_data_pyoof(oofh_data)
    [name, obs_object, obs_date, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    u_data, v_data = np.degrees(u_data), np.degrees(v_data)
    vmin = np.min(beam_data)
    vmax = np.max(beam_data)


    fig, ax = plt.subplots(ncols=3)

    for i in range(3):
        # new grid for beam_data
        u_ng = np.linspace(u_data[i].min(), u_data[i].max(), 300)
        v_ng = np.linspace(v_data[i].min(), v_data[i].max(), 300)

        beam_ng = interpolate.griddata(  # new grid beam map
            # coordinates of grid points to interpolate from.
            points=(u_data[i], v_data[i]),
            values=beam_data[i],
            # coordinates of grid points to interpolate to.
            xi=tuple(np.meshgrid(u_ng, v_ng)),
            method='cubic'
            )

        extent = [u_ng.min(), u_ng.max(), v_ng.min(), v_ng.max()]
        ax[i].imshow(beam_ng, extent=extent, vmin=vmin, vmax=vmax)
        ax[i].contour(u_ng, v_ng, beam_ng, 10)

The properties of the receiver and the telescope can be found in the sub-packages `~pyoof.aperture` and `~pyoof.telgeometry`. These are important geometrical aspects that will make the nonlinear least squares minimization process more precise. An example on how to gather the properties of a telescope is below::

    pr = 50. # primary reflector radius m
    # telescope = [block_dist, opd_func, pr, name]
    telescope = dict(
        effelsberg=[
            telgeometry.block_effelsberg,
            telgeometry.opd_effelsberg,
            pr,
            'effelsberg'
            ]
        manual=[
            telgeometry.block_manual(pr=50, sr=3.25, a=1, L=50-3.25),
            telgeometry.opd_manual(Fp=30, F=387),
            pr,
            'manual'
            ]
        )

Besides this an illumination function must be selected, to fulfill the receiver properties. The illumination functions can be found on the `~pyoof.aperture` package. Currently there are two made functions, `pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.

.. note::
    The blockage distribution, ``block_dist``, and the OPD function, ``opd_func`` can be manually created or using the standard manual functions, `~pyoof.telgeometry.block_manual` and `~pyoof.telgeometry.opd_manual`.

.. warning::
    The geometrical aspects of the telescope, such as blockage distribution and OPD function, are fundamental for a good fit in the nonlinear squares minimization. If they are not properly estimated there will be a great difference in the final aperture phase distribution.

After creating the basic structure, the core function, `~pyoof.fit_beam`, can be executed,

.. code-block:: python

    pyoof.fit_beam(
        data_info=data_info,                   # information
        data_obs=[beam_data, u_data, v_data],  # observed beam
        method='trf',                          # opt. algorithm 'trf', 'lm' or 'dogbox'
        order_max=5,                           # it will fit from 1 to order_max
        illum_func=illum_pedestal,             # or illum_gauss
        telescope=telescope['effelsberg'],     # [block_dist, opd_func, pr, name]
        resolution=2**8,                       # standard is 2 ** 8
        box_factor=5,                          # box_size = 5 * pr, pixel resolution
        fit_previous=True,                     # default
        config_params_file=None,               # default
        make_plots=True,                       # default
        verbose=2                              # default
        )

This will show the main properties of the fit as well as its progress. The key ``order_max`` is the maximum order to be fitted in the process. It starts from the polynomial order one until order five. If ``fit_previous=True``, then the algorithm will use coefficients from the previous order (:math:`n`) as the initial coefficients for the next order (:math:`n+1`), this feature is strongly recommended. The keys ``method`` and ``verbose`` are related to the least squares minimization package (`~scipy.optimize.least_squares`). The ``box_factor`` and ``resolution`` are necessary to perform a good FFT2 (more information about this will be updated).

The `~pyoof` package will generate a directory called `pyoof_out/name-000/`, where `name` is the name of the fits file used. The directory will contain:

* `beam_data.csv`: Corresponds to the observed power pattern as a flat array.
* `corr_n#.csv`: Correlation matrix evaluated at the last residual from the least squares minimization, order `#`.
* `cov_n#.csv`: Covariance matrix evaluated at the last residual from the least squares minimization, order `#`.
* `fitpar_n#.csv`: Estimated parameters from the least squares minimization, order `#`. These correspond to the illumination coefficients, ``I_coeff``, and the Zernike circle polynomial coefficients, ``K_coeff``. They are gathered as ``params_solution = I_coeff + K_coeff``. Be aware that some coefficients can be fixed, or excluded from the optimization, such as the illumination offset (:math:`x_0, y_0`) or the illumination amplitude (:math:`A_{E_\text{a}}`). The default configuration is in `config_params.yml` (package directory). You can also provide your own configuration set up.
* `grad_n#.csv`: Gradient of the last residual evaluation, order `#`.
* `phase_n#.csv`: Aperture phase distribution (phase error), in radians, for the primary dish, order `#`. The solution takes the same approach as the `fitpar_n#.csv`, using the same telescope configuration introduced in the `~pyoof.fit_beam`.
* `pyoof_info.yml`: It contains information about the fit and observation parameters.
* `res_n#.csv`: Last evaluation of the residual from the least squares minimization, order `#`.
* `u_data.csv`: :math:`x`-axis vector which contains position coordinates for `beam_data.csv`, in radians.
* `v_data.csv`: :math:`y`-axis vector which contains position coordinates for `beam_data.csv`, in radians.

Finally if the key ``make_plots=True``, then `~pyoof.fit_beam` will create a sub-directory containing the most important plots to study the fit, as well as the phase error maps for the primary dish.

See Also
========

* `Out-of-focus holography at the Green Bank Telescope <https://www.aanda.org/articles/aa/ps/2007/14/aa5765-06.ps.gz>`_
* `Measurement of antenna surfaces from in- and out-of-focus beam maps using astronomical sources <https://www.aanda.org/articles/aa/ps/2007/14/aa5603-06.ps.gz>`_
* `Zernike circle polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
* `Effelsberg 100-m radio telescope <https://en.wikipedia.org/wiki/Effelsberg_100-m_Radio_Telescope>`_
* `Essential Radio Astronomy <http://www.cv.nrao.edu/course/astr534/ERA_old.shtml>`_


Available modules
=================

.. toctree::
    :maxdepth: 2

    aperture/index
    telgeometry/index
    zernike/index

.. automodapi:: pyoof

Project details
===============

.. toctree::
    :maxdepth: 1

    license

Acknowledgments
===============

This code makes use of the excellent work provided by the
`Astropy <http://www.astropy.org/>`__ community. `~pyoof` uses the Astropy package and also the
`Astropy Package Template <https://github.com/astropy/package-template>`__
for the packaging.

As well `bwinkel <https://github.com/bwinkel>`_ for his support and help during the development of `~pyoof`.


