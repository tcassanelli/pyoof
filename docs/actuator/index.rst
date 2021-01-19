.. pyoof-actuator:

:tocdepth: 2

****************************************
Actuator (`pyoof.actuator`)
****************************************

.. currentmodule:: pyoof.actuator

Introduction
============
This is a developer sub-package. The purpose of `~pyoof.actuator` is to provide tools for telescopes that have included in their structure an active surface control system. An active system has a model to correct for gravitational deformations, what we propose here is to use the results from `~pyoof.fit_zpoly` (out-of-focus holography) to compute such model and return the required corrections to apply to every actuator in the active surface, and hence improve the telescope's sensitivity.

The current sub-package only contains corrections for the Effelsberg telescope in the class `~pyoof.actuator.EffelsbergActuator`, future versions will expand this to other types of active surface. 

Active surface at the Effelsberg telescope
------------------------------------------
To use the `~pyoof.actuator` is simple, first let's take a look at the active surface control system and its look-up table at the Effelsberg telescope.

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from astropy import units as u
    from pyoof.actuator import EffelsbergActuator

    ae = EffelsbergActuator(
        frequency=34.75 * u.GHz,  # obs frequency
        nrot=1,                   # convention parameter
        sign=-1,                  # convention parameter
        order=5,                  # Polynomial order
        sr=3.25 * u.m,            # Sub-reflector radius
        pr=50 * u.m,              # Primary reflector radius
        resolution=1000           # Phase-error map resolution
        )

    fig = ae.plot()
    plt.show()

The following table is the transformation from the actuator space (in micrometer displacement) to the phase-error space. Notice that ``nrot`` and ``sign`` are important parameters to be defined. Several tests at the Effelsberg telescope lead us to find those values.

Reference/API
=============

.. automodapi:: pyoof.actuator
    :no-inheritance-diagram: