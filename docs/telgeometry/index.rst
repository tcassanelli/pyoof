.. pyoof-telgeometry:

****************************************
Telescope geometry (`pyoof.telgeometry`)
****************************************

.. currentmodule:: pyoof.telgeometry

Introduction
============

To apply the OOF holography to any type of antenna or radio telescope it is necessary to use their geometrical aspects. This properties can be the type of telescope, Cassegrain, Gregorian, etc, and how/where is the sub/refelctor located in the aperture plane (if there is any). The type of the telescope will give information about the optical path carried every time OOF observations are performed and the location of the sub-relfector will give a truncation over the aperture plane.

Getting started
===============

To start using the telescope geometry functions is stright forward, simply by defining a meshed array and replacing them in the function.

.. note::
    Unfortunately for now there is only premade functions for the Effelsberg telescope, plus a manual version of the functions. The user is encouraged to develop their own functions for the telescope geometry. In future versions this will be updated.

Blockage distribution :math:`B(x, y)`
-------------------------------------
The blockage distribution corresponds to the elements that block the light in the aperture plane. Thi could be the support legs, the sub-reflector and shade effects. For the Effelsberg telescope the three of them are present. How to construct the blockage, follows

.. plot::
    :include-source:

    import numpy as np
    from pyoof import telgeometry
    import matplotlib.pyplot as plt

    pr = 50  # primary dish radius
    x = np.linspace(-pr, pr, 1e3)
    xx, yy = np.meshgrid(x, x)

    B = telgeometry.block_effelsberg(xx, yy)

    fig, ax  = plt.subplots()
    ax.imshow(B, extent=[-pr, pr] * 2, cmap='viridis')
    ax.set_xlabel('$x$ m')
    ax.set_ylabel('$y$ m')
    ax.set_title('Blockage dist. Effelsberg telescope')

To construct manually the blockage distribution, using the premade function `~pyoof.telgeometry.block_manual`, basic geometrical aspect from the telescope are required, such as the primary and secondary dish radii, dimensions of the support legs, etc.

Optical path difference :math:`\delta(x, y; d_z)`
-------------------------------------------------





See Also
========

Reference/API
=============

.. automodapi:: pyoof.telgeometry
