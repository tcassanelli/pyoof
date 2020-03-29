.. pyoof-actuator:

:tocdepth: 2

****************************************
Actuator (`pyoof.actuator`)
****************************************

.. currentmodule:: pyoof.actuator

Introduction
============
This is a developer sub-package. The purpose of `~pyoof.actuator` is to provide tools for telescopes that have included in their structure an active surface control system. An active system has a model to correct for gravitational deformations, what we propose here is to use the results from `~pyoof.fit_zpoly` (out-of-focus holography) to compute such model and return the required corrections to apply to every actuator in the active surface, and hence improve the telescope's sensitivity.

The current sub-package only contains corrections for the Effelsberg telescope in the class `~pyoof.actuator.EffelsbergActuator`.

Reference/API
=============

.. automodapi:: pyoof.actuator
    :no-inheritance-diagram: