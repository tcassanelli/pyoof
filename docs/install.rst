************
Installation
************

Requirements
============

pyoof has the following strict requirements:

- `Python <http://www.python.org/>`__ 3.5 or later.

- `setuptools <https://pythonhosted.org/setuptools/>`__: Used for the package
  installation.

- `NumPy <http://www.numpy.org/>`__ 1.11 or later.

- `SciPy <https://scipy.org/>`__: 0.15 or later.

- `astropy <http://www.astropy.org/>`__: 1.3 or later (2.0 recommended).

- `pytest <https://pypi.python.org/pypi/pytest>`__ 2.6 or later.

- `matplotlib <http://matplotlib.org/>`__ 1.5 or later: To provide plotting
  functionality.

- `PyYAML <http://pyyaml.org>`__ 3.11 or later.


Installing pyoof
================

.. note::

    Currently the package installation is not working without a prior installation of the `miniconda <https://conda.io/miniconda.html>`_ distribution (or anaconda distribution). In the mean time please install miniconda and follow the instructions below.

Using pip
---------

To install pyoof with `pip <https://pip.pypa.io/en/stable/>`__, simply run

.. code-block:: bash

    pip install pyoof

.. note::

    Use the ``--no-deps`` flag if you already have dependency packages
    installed, since otherwise pip will sometimes try to "help" you
    by upgrading your installation, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`__.

    We recommend to use a Python distribution, such as `Anaconda
    <https://www.continuum.io/downloads>`_.

    Do **not** install pyoof or other third-party packages using ``sudo``
    unless you are fully aware of the risks.

.. _source_install:

Installation from source
------------------------

There are two options, if you want to build pyoof from sources. Either, you
install the tar-ball (`*.tar.gz` file) from `PyPI
<https://pypi.python.org/pypi/pyoof>`_ and extract it to the directory of
your choice, or, if you always want to stay up-to-date, clone the git
repository:

.. code-block:: bash

    git clone https://github.com/tcassanelli/pyoof

Then go into the `~pyoof` source directory and run:

.. code-block:: bash

    python setup.py install

Again, consider the ``--user`` option or even better use a python distribution
such as `Anaconda <https://www.continuum.io/downloads>`_ to avoid messing up
the system-wide Python installation.

.. _testing_installed_pyoof:

Testing an installed pyoof
--------------------------

The easiest way to test your installed version of `pyoof` is running
correctly is to use the `~pyoof.test()` function::

    import pyoof
    pyoof.test()

To run the tests for one sub-package, e.g., `~pyoof.aperture`, only::

    import pyoof
    pyoof.test('aperture')

The tests should run and print out any failures, which you can report at
the `pyoof issue tracker <http://github.com/tcassanelli/pyoof/issues>`__.

.. note::

    This way of running the tests may not work if you do it in the
    `pyoof` source distribution directory.

If you prefer testing on the command line and usually work with the source
code, you can also do

.. code-block:: bash

    python setup.py test

    # to run tests from a sub-package
    python setup.py test -P aperture
