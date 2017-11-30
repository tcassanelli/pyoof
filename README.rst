*****
pyoof
*****

- *Version: 0.2*
- *Author: Tomas Cassanelli*
- *User manual:* `stable <https://readthedocs.../>`__ |
  `developer <https://readthedocs.../latest/>`__

.. image:: https://img.shields.io/pypi/v/pyoof.svg
    :target: https://pypi.python.org/pypi/pyoof
    :alt: PyPI tag

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

pyoof is a Python package that contains all needed tools to perform out-of-focus (OOF) holography on astronomical beam maps for single-dish radio telescopes. It is based on the original OOF holography papers,

* [Out-of-focus holography at the Green Bank Telescope](https://www.aanda.org/articles/aa/ps/2007/14/aa5765-06.ps.gz)
* [Measurement of antenna surfaces from in- and out-of-focus beam maps using astronomical sources](https://www.aanda.org/articles/aa/ps/2007/14/aa5603-06.ps.gz)

and [software](https://github.com/bnikolic/oof) developed by [Bojan Nikolic](http://www.mrao.cam.ac.uk/~bn204/oof/).

The pyoof package calculates the aperture phase distribution map from a set of beam maps (telescope observations), at a relatively good signal-to-noise as described by B. Nikolic. By using a nonlinear least squares minimization, a convenient set of polynomials coefficients can be found to represent the aperture distribution. Once this is calculated the aberrations on the primary dish are known.

We are currently testing the pyoof package at the [Effelsberg radio telescope](https://en.wikipedia.org/wiki/Effelsberg_100-m_Radio_Telescope) :satellite:.

Project Status
==============

.. image:: https://travis-ci.org/tcassanelli/pyoof.svg?branch=master
    :target: https://travis-ci.org/tcassanelli/pyoof
    :alt: Pyoof's Travis CI Status

.. image:: https://coveralls.io/repos/github/tcassanelli/pyoof/badge.svg?branch=master
    :target: https://coveralls.io/github/tcassanelli/pyoof?branch=master
    :alt: Pyoof's Coveralls Status

`pyoof` is still in the early-development stage. While much of the
functionality is already working as intended, the API is not yet stable.
Nevertheless, we kindly invite you to use and test the library and we are
grateful for feedback. Note, that work on the documentation is still ongoing.

Usage
=====
For now the installation is only available from the source, clone the repository and then execute

.. code-block:: bash

    python setup.py install


I believe in the future :smile:, so please install Python 3.

License
=======
pyoof is licensed under a 3-clause BSD style license - see the LICENSE.rst file.
