pyoof
-----

* *Version: 0.1*
* *Author: Tomas Cassanelli*

pyoof is a Python package that contains all needed tools to perform out-of-focus (OOF) holography on astronomical beam maps for single dish radio telescopes. It is based on the original OOF holography papers,

* [Out-of-focus holography at the Green Bank Telescope](https://www.aanda.org/articles/aa/ps/2007/14/aa5765-06.ps.gz)
* [Measurement of antenna surfaces from in- and out-of-focus beam maps using astronomical sources](https://www.aanda.org/articles/aa/ps/2007/14/aa5603-06.ps.gz)

and [software](https://github.com/bnikolic/oof) developed by [Bojan Nikolic](http://www.mrao.cam.ac.uk/~bn204/oof/).

The pyoof package calculates the phase error map from a set of beam maps, at a relatively good signal-to-noise as described by B. Nikolic. From here it is possible to compute the phase error at the sub-reflector and eventually make surface corrections in the telescope if an active surface exits. We are currently testing the pyoof package at the [Effelsberg radio telescope](https://en.wikipedia.org/wiki/Effelsberg_100-m_Radio_Telescope) :satellite:.

Usage
-----
For now the installation is only available from the source, clone the repository and then execute

```
$ python install setup.py
```

I believe in the future :smile:, so please install Python 3.

License
-------

I need to read but soon it will be uploaded, sorry for the inconvenience :no_mouth:
