
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys
from distutils.version import LooseVersion

__minimum_python_version__ = "3.6"

__all__ = []


class UnsupportedPythonError(Exception):
    pass


if LooseVersion(sys.version) < LooseVersion(__minimum_python_version__):
    raise UnsupportedPythonError("packagename does not support Python < {}"
                                 .format(__minimum_python_version__))

if not _ASTROPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.
    from .aux_functions import *
    from .math_functions import *
    from .core import *
    from .plot_routines import *
    from .simulate_data import *

    # comment so they don't apper in docs
    # __all__ += aux_functions.__all__
    # __all__ += math_functions.__all__
    # __all__ += plot_routines.__all__

    __all__ += core.__all__
    __all__ += simulate_data.__all__

    from . import aperture
    from . import telgeometry
    from . import zernike
    from . import actuator
