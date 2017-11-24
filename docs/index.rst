
:tocdepth: 3

###################
pyoof Documentation
###################

************
Introduction
************

Welcome to the `pyoof` documentation. `pyoof` is a Python package which computes out-of-focus (OOF) holography for beam maps of a single-dish radio telescope. The method was developed by `B. Nikolic et al <https://www.aanda.org/articles/aa/ps/2007/14/aa5603-06.ps.gz>`_. The OOF holography is a phase-retrieval holography procedure to find the aperture phase distribution, :math:`\varphi(x, y)`, (or simply phase error) and the associated errors on a telescope's surface (primary dish).
The main advantage of the method, compared to the traditional with-phase holography, is the lack of extra equipment needed to perform observations and the wide range in telescope's elevation. These two allow the method to study gravitational deformations on the surface, the most well behaved and prominent source of deformation (other sources could be thermal and wind deformations which are non-repeatable).
The method required the use of a compact source (point-like), with a good signal-to-noise (:math:`\geq200`) and root-mean-squared noise (:math:`\geq 14 \, \mathrm{mJy}`). Then a set of continuum observations (on-the-fly mapping) are required, two of them out-of-focus and one in-focus. The OOF observations are performed adding a known radial offset (:math:`d_z`) of the order of centimeters, to the telescope's sub-reflector.
The defocused terms are needed to break the degeneracy between the power pattern (observed quantity :math:`P(u, v)`), and the aperture distribution, :math:`\underline{E_\text{a}}(x, y)`, without OOF observations the problems becomes under-determined. Such relation is given by,

.. math::
    P(u, v) = \left| \mathcal{F}\left[\underline{E_\text{a}}(x, y) \right]  \right|^2.
    :label: power-pattern-definition

The aperture phase distribution, related to the power pattern (observed beam maps) can be expressed as a sum of polynomials times certain coefficients,

.. math::
    \varphi(x, y) = 2\pi \cdot W(x, y) = 2\pi \cdot \sum_{n, \ell} K_{n\ell}U^\ell_n(\varrho, \vartheta).
    :label: phase-error-definition

The parametrization of the aperture phase distribution allows its construction by using a nonlinear least squares minimization (`~pyoof.fit_beam`), due to the highly nonlinear relation between the aperture distribution and power pattern (degenerated). The `pyoof` package takes as an input the observed beam maps and computes such coefficients to find the aperture phase distribution (on the primary reflector).

Among the basic operations that the `pyoof` package does, there are: construction aperture distribution (`pyoof.aperture.aperture`), calculation of its Fast Fourier Transform in two dimensions (`~pyoof.aperture.radiation_pattern`), calculation of a residual between observed and model power pattern, and nonlinear least squares minimization (`~pyoof.fit_beam`). A simplified flowchart of the `pyoof` package main routine follows,

OOF holography constants/distribution/function
==============================================
The OOF holography make use of several constants, distributions and functions. Some of them are related to each other. Please keep in mind their mathematical symbol and what they represent.

Aperture distribution: :math:`\underline{E_\text{a}}(x, y)` (`~pyoof.aperture.aperture`)
---------------------------------------------------------------------------------------------------------

Collection of sub-functions and -distributions. It is the point of connection between the observational data and the theoretical model. See Eq. :eq:`power-pattern-definition`. Its definition is:

.. math::
    \underline{E_\text{a}}(x, y) = B(x, y)\cdot E_\text{a}(x, y) \cdot \mathrm{e}^{\mathrm{i} \{\varphi(x, y) + \frac{2\pi}{\lambda}\delta(x,y;d_z)\}}.

Aperture phase distribution (phase error): :math:`\varphi(x, y)` (`~pyoof.aperture.phase`)
------------------------------------------------------------------------------------------------------

The aperture phase distribution represents the aberrations of an optical system, measured in radians. It is also related to the wavefront (aberration) distribution (see Eq. :eq:`phase-error-definition`), :math:`W(x, y)`, which is the classical function in optics to define aberrations.

Blockage distribution: :math:`B(x, y)` (`~pyoof.telgeometry.block_manual`)
--------------------------------------------------------------------------

It is the truncation or blockage that the telescope structure (or a shade effect) does to the aperture plane. It is usually best represented as a two dimensional modified Heaviside step function. Since it depends on the geometry, it will depend on the telescope used.

(Field) radiation pattern: :math:`F(u, v)` (`~pyoof.aperture.radiation_pattern`)
--------------------------------------------------------------------------------

The (field) radiation pattern is a complex distribution, it is the direct Fourier Transform of the aperture distribution, :math:`F(u, v) = \mathcal{F}[\underline{E_\text{a}}(x, y)]`. Represents the angular variation of the radiation around the antenna.

Illumination function: :math:`E_\text{a}(x, y)` (`~pyoof.aperture.illum_pedestal` or `~pyoof.aperture.illum_gauss`)
------------------------------------------------------------------------------------------------------------------------------------------------

The illumination function is a characteristic of a receiver and it is used to reduce the level on the side lobes. It can be modeled with different types of functions, such as Gaussian or a cosine. The default one used by the `pyoof` package is the parabolic taper on a pedestal (order :math:`q=2`),

.. math::
    E_\text{a}(\rho') = C + (1 - C)\cdot \left[ 1-\left(\frac{\rho'}{R}\right)^2 \right]^q, \qquad C=10^{\frac{c_\text{dB}}{20}}.

It can also be possible to define a new illumination function.

Illumination coefficients (``I_coeff``)
---------------------------------------

The illumination coefficients correspond to four constants that need to be added to the illumination function, :math:`E_\text{a}`. These coefficients are the illumination amplitude, :math:`A_{E_\text{a}}`, the illumination offset (:math:`x_0, y_0`), and the illumination taper in decibels, :math:`c_\text{dB}` or :math:`\sigma_\text{dB}` (depending on the illumination function used). The coefficients are organized in the following order::
``I_coeff = [i_amp, c_dB, x0, y0]``

From these coefficients the most relevant of them is the illumination taper. It gives a measure on how strong in the signal from the center of the aperture respect to its edges.

Observation wavelength: :math:`\lambda` (``wavel``)
---------------------------------------------------

For one OOF holography observation it is required to perform two of the out-of-focus and one in-focus. All of them need to be observed with the same receiver at a certain :math:`\lambda`. The wavelength is also important at the time to know the aberrations in the primary dish.

Optical path difference (OPD) function: :math:`\delta(x, y;d_z)` (`~pyoof.telgeometry.opd_manual`)
--------------------------------------------------------------------------------------------------------------

The OPD function is the extra path that light has to travel every time a radial offset, :math:`d_z`, (known a priori) is added to the sub-reflector (to make OOF observations). The OPD function depends strictly on the telescope geometry.

Power pattern (beam map): :math:`P(u, v)` (``power_pattern``)
-------------------------------------------------------------

The power pattern or beam map is the observed quantity. For one OOF holography observation it is required to perform two of the out-of-focus and one in-focus continuum scans (on-the-fly mapping). This will produce two beam maps with a clear interference pattern and one with the common in-focus beam size.

Radial offset: :math:`d_z` (``d_z``)
------------------------------------

The radial offset is defocused term added to the sub-reflector. It is usually of the order of centimeters (it will vary from telescope to telescope). A small value of the :math:`d_z` may not add enough change respect to the in-focus beam, increasing the degeneracy on the least squares minimization. A large value of :math:`d_z` will decrease the signal-to-noise on the source, making the Zernike circle polynomials coefficients with high uncertainty.
This value can only be set by observing and analyzing the data.

Wavefront (aberration) distribution: :math:`W(x, y)` (`~pyoof.aperture.wavefront`)
----------------------------------------------------------------------------------------------------------

The wavefront (aberration) distribution is the classical approach to represent an aberrated wavefront. From optical physics it is known that :math:`W(x, y)` can be expressed in a sum of a convenient set of orthonormal polynomials. The Zernike circle polynomials, :math:`U^\ell_n(\varrho,\vartheta)`, fulfill the mathematical conditions, see Eq. :eq:`phase-error-definition`.

Zernike circle polynomial: :math:`U^\ell_n(\varrho, \vartheta)` (`~pyoof.zernike.R` and `~pyoof.zernike.U`)
------------------------------------------------------------------------------------------------------------------------

The Zernike circle polynomials were introduced by Frits Zernike for testing his phase contrast method in circular mirror figures. The polynomials were used by Ben Nijboer to study the effects on small aberrations on diffracted images with a rotationally symmetric origin on circular pupils.

The polynomials can be separated in a radial and angular section and their respective indices are: order :math:`n` and its angular dependence :math:`\ell`. The radial polynomials are given by,

.. math::

    R^{\pm m}_n (\varrho) = \frac{1}{\left(\frac{n-m}{2}\right)!\cdot\varrho^m}\left\{\frac{\text{d}}{\text{d}\left(\varrho^2\right)} \right\}^{\frac{n-m}{2}} \left\{  \left( \varrho^2 \right)^{\frac{n+m}{2}} \cdot  \left( \varrho^2 -1 \right)^{\frac{n-m}{2}} \right\}.

With :math:`m = |\ell|`. The complete Zernike circle polynomials are:

.. math::
    U^\ell_n(\varrho, \vartheta) = R^m_n(\varrho) \cdot \cos m\vartheta \qquad \ell \geq 0, \\
    U^\ell_n(\varrho, \vartheta) = R^m_n(\varrho) \cdot \sin m\vartheta \qquad \ell < 0.

For more about the Zernike circle polynomials and the `pyoof` package see the Jupyter notebook `zernike.ipynb <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/zernike.ipynb>`_ examples from the repository.


Zernike circle polynomial coefficient: :math:`K_{n\ell}` (``K_coeff``)
----------------------------------------------------------------------

The Zernike circle polynomial coefficients are the final parameters to be found by the least squares minimization (as well as ``I_coeff``). By finding them it is possible to reconstruct the aperture phase distribution, :math:`\varphi(x, y)`, see Eq. :eq:`phase-error-definition`. Depending on the polynomial order, :math:`n`, there will be a different number of them. In general the structure to list them is: ``K_coeff = [K(0, 0), K(1, -1), K(1, 0), K(1, 1), ... , K(n, l)]``.

See Also
========

* `Out-of-focus holography at the Green Bank Telescope <https://www.aanda.org/articles/aa/ps/2007/14/aa5765-06.ps.gz>`_
* Out-of-focus holography
* Zernike circle polynomials
* Holography
* Radio astronomy
* Effelsberg telescope
* Effelsberg wiki


Getting Started
===============

.. toctree::
    :maxdepth: 1

    install
    Tutorials <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/>

User Documentation
==================

To start with the `pyoof` package first visit the sub-packages in the following order:

* The use of the Zernike circle polynomials, in zernike/index

The fits file
-------------

The `pyoof` package requires a specific format for the fits files. The format must include observational parameters as well as data. These parameters can be seen in `~pyoof.extract_data_pyoof` function.::

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

Where the data files are separated in ``MINUS OOF``, ``ZERO OOF`` and ``PLUS OOF`` out-of-focus observations. Each of them has to have the radial offset (:math:`d_z`) value which has to be written in the BinTableHDU header with the key ``DZ``. The BinTableHDU header then is::

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

BinTableHDU data has to have another three keys, if ``hdulist[1].data`` is printed, the ``MINUS OOF`` is observation selected, then::

    >>> hdulist[1].data  # doctest: +REMOTE_DATA +IGNORE_OUTPUT
    FITS_rec([(-0.00089586416, -0.00089586416, -1007.5198),
           (-0.00087720033, -0.00089586416, 262.23615),
           (-0.00085853651, -0.00089586416, -557.19135), ...,
           (0.00085853651, 0.00089586416, 419.55826),
           (0.00087720033, 0.00089586416, -53.516975),
           (0.00089586416, 0.00089586416, -110.75285)],
          dtype=(numpy.record, [('U', '<f4'), ('V', '<f4'), ('BEAM', '<f4')]))

were the ``U`` and ``V`` are the dimensions or x- and y-axis. The key ``BEAM`` (beam map) corresponds to the observed power pattern. The three of them are flat arrays. Hence, to reconstruct the plot in two dimensions an interpolation is required.

After using a fits file it needs to be closed.::

    >>> hdulist.close()

.. note::
    The fits format can also be avoided. If the required parameters are added to the `~pyoof.fit_beam` function, then the `pyoof` package will also work. Although, it is recommended to use the fits format, for ease and clean storage of the data.

It is also possible to try the `pyoof` generating your own data, with the build-in function `~pyoof.beam_generator`. The Jupyter notebook `oof_holography.ipynb <http://nbviewer.jupyter.org/github/tcassanelli/pyoof/blob/master/notebooks/oof_holography.ipynb>`_  has generated data, plus noise, and it is used as an input for the `pyoof` package.

Using pyoof
-----------

The main function from the `pyoof` package is `~pyoof.fit_beam`. This functions does all the numerical computation to find the Zernike circle coefficients and the illumination coefficients in the ``params_solution`` (`~numpy.ndarray`). To start first the data needs to be extracted using `~pyoof.extract_data_pyoof`, then the output constant, arrays and strings are given to the core function `~pyoof.fit_beam`. Besides the observational data some other parameters need to be added, these are the ones related to the FFT2 (`~numpy.fft.fft2`) and functions related to the telescope geometry.

Following the same example as before, it is possible to make a simple preview of the file, using `~pyoof.extract_data_pyoof`.

.. plot::
    :include-source:

    import pyoof
    # Generally you would store the filename as string
    from astropy.utils.data import get_pkg_data_filename
    oofh_data = get_pkg_data_filename('pyoof/data/example0.fits')

    data_info, data_obs = pyoof.extract_data_pyoof(oofh_data)
    [name, obs_object, obs_date, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    pyoof.plot_data(
        u_data=u_data,
        v_data=v_data,
        beam_data=beam_data,
        d_z=d_z,
        angle='degrees',
        title='',
        res_mode=False
        )

The properties of the receiver and the telescope can be found in the sub-packages `~pyoof.aperture` and `~pyoof.telgeometry`. These are important geometrical aspects and will make the nonlinear least squares minimization process more precise. For the Effelsberg telescope there are::

    pr = 50. # primary reflector radius m

    # telescope = [blockage, delta, pr, name]
    telescope = dict(
        effelsberg=[
            telgeometry.block_effelsberg,
            telgeometry.opd_effelsberg,
            pr,
            'effelsberg'
            ]
        )

    illumination = dict(
        gaussian=[aperture.illum_gauss, 'gaussian', 'sigma_dB'],
        pedestal=[aperture.illum_pedestal, 'pedestal', 'c_dB']
        )

.. note::
    The blockage distribution, ``block_dist``, and the OPD function, ``opd_func`` can be manually created or using the standard manual functions included, `~pyoof.telgeometry.block_manual` and `~pyoof.telgeometry.opd_manual`.


Available modules
=================

.. toctree::
    :maxdepth: 1

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
`Astropy <http://www.astropy.org/>`__ community. `pyoof` uses the Astropy package and also the
`Astropy Package Template <https://github.com/astropy/package-template>`__
for the packaging.
