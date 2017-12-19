#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import pytest
import numpy as np
import matplotlib
from astropy.utils.data import get_pkg_data_filename
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import pyoof
import nose


# Initial fits file configuration
n = 7                                    # initial order
N_K_coeff = (n + 1) * (n + 2) // 2       # total numb. polynomials
K_coeff = np.array([0.1] * N_K_coeff)    # random Zernike circle coeff.

@image_comparison(
    baseline_images=['plot_phase'],
    extensions=['pdf']
    )
def test_plot_phase():
    pyoof.plot_phase(
        K_coeff=K_coeff, notilt=False, pr=50, title='test plot'
        )
