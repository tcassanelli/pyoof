.. pyoof-telgeometry:

:tocdepth: 2

****************************************
Telescope geometry (`pyoof.telgeometry`)
****************************************

.. currentmodule:: pyoof.telgeometry

Introduction
============

To apply the OOF holography first we need to model the antenna and its geometry. These properties include the type of telescope, Cassegrain, Gregorian, etc, and how/where is the sub-reflector located in the aperture plane (if there is any). The type of the telescope will give information about the optical path carried every time OOF observations are performed and the location of the sub-reflector will give a truncation over the aperture plane. Both og them fundamental to achieve a good least squares minimization.

Using `~pyoof.telgeometry`
==========================

Using the telescope geometry functions is straight forward, simply by defining a meshed array and replacing them in the function.

.. note::
    Unfortunately for now there are only pre-made functions for the Effelsberg telescope, plus a manual version of the functions. The user is encouraged to develop their own functions for the telescope geometry. In future versions this will be updated.

Blockage distribution :math:`B(x, y)`
-------------------------------------
The blockage distribution corresponds to the elements that block incoming radiation in the aperture plane. This could be the support legs, the sub-reflector and shade effects. And for the Effelsberg telescope, all three of them are present. How to construct the blockage follows,

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pyoof import telgeometry

    pr = 50. * u.m  # primary reflector radius
    x = np.linspace(-pr, pr, 1000)
    xx, yy = np.meshgrid(x, x)

    B = telgeometry.block_effelsberg(xx, yy)

    fig, ax  = plt.subplots()
    ax.imshow(B, extent=[-pr.value, pr.value] * 2, cmap='viridis')
    ax.set_xlabel('$x$ m')
    ax.set_ylabel('$y$ m')
    ax.set_title('Blockage dist. Effelsberg telescope')

To construct manually the blockage distribution, using the pre-made function `~pyoof.telgeometry.block_manual`, basic geometrical aspects from the telescope are required, such as the primary and secondary dish radii, dimensions of the support legs, etc.

Optical path difference function :math:`\delta(x, y; d_z)`
----------------------------------------------------------

The optical path difference (OPD) function, in Cassegrain and Gregorian geometries is characterized by their primary reflector focus and their effective focal length. No other information is required and it is a purely theoretical formula (ray-tracing). There is also an included manual version for the OPD function and the user is encourage to develop her/his own version of it.

.. note::
    If you are interested in its theoretical derivation send me an email!

Same as before the OPD function for the Effelsberg telescope is already made,

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pyoof import telgeometry

    pr = 50 * u.m  # primary reflector radius
    x = np.linspace(-pr, pr, 1000)

    delta = []
    for d_z in [-2.2, 0, 2.2] * u.cm:
        delta.append(telgeometry.opd_effelsberg(x=x, y=0, d_z=d_z))

    fig, ax = plt.subplots()

    labels = ['$\\delta(r ;d_z^-)$', '$\\delta(r ;0)$', '$\\delta(r ;d_z^+)$']
    for i in range(3):
        ax.plot(x, delta[i], label=labels[i])

    ax.grid(True, ls='--')
    ax.set_xlabel('$r$ m')
    ax.set_ylabel('$\\delta(r; d_z)$')
    ax.set_title('OPD function Effelsberg telescope')
    ax.legend(loc='upper right')

From the plot, it becomes clear that by adding a radial offset of :math:`d_z=0` m the solution becomes flat.

.. note::
    The presented distribution/functions are a good first order approximation that follow ray-tracing optics. In reality light diffraction has a considerable effect in such measurements. 

See Also
========

* `Out-of-focus holography at the Green Bank Telescope <https://www.aanda.org/articles/aa/ps/2007/14/aa5765-06.ps.gz>`_
* `Measurement of antenna surfaces from in- and out-of-focus beam maps using astronomical sources <https://www.aanda.org/articles/aa/ps/2007/14/aa5603-06.ps.gz>`_
* `The Effelsberg holography campaing - 2001 <https://eff100mwiki.mpifr-bonn.mpg.de/lib/exe/fetch.php?media=information_for_astronomers:eff_holo_rpt.pdf>`_

Reference/API
=============

.. automodapi:: pyoof.telgeometry
