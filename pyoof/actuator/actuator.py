#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import astropy
from astropy import constants
from astropy import units as apu
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_filename
from scipy import interpolate, optimize
from ..aperture import phase

__all__ = ['EffelsbergActuator']


class EffelsbergActuator():
    """
    Several tasks for the Effelsberg telescope and the active surface control
    system located in the 6.5 m sub-reflector. The purpose of all these
    functions is to transform the phase-error maps obtained from the core
    `~pyoof` routines, to an equivalent actuator perpendicular displacement to
    correct those effects seen in the main phase-error maps.

    Attributes
    ----------
    frequency : `~astropy.units.quantity.Quantity`
        Frequency of the observation in Hertz.
    nrot : `int`
        This is a required rotation to apply to the phase maps (obtained
        from `~pyoof.fit_zpoly`) to get the right orientation of the active
        surface look-up table in the active surface control system.
    sign : `int`
        It is the value of the phase-error amplitude as seen from the active
        surface, same as ``nrot`` is a convention for the Effelsberg telescope.
    resolution : `int`
        Resolution for the phase-error map, usually used ``resolution = 1000``
        in the `~pyoof` package.
    limits_amplitude : `~astropy.units.quantity.Quantity`
            This is the maximum and minimum amplitude that the actuators
            can make in a displacement, for the Effelsberg active surface
            control system is :math:`\\pm5 \\mathrm{mm}`.
    path_lookup : `str`
        Path for the current look-up table that controls the active surface
        control system. If `None` it will select the default table from the
        FEM model.
    """

    def __init__(
        self, frequency=34.75 * apu.GHz, nrot=1, sign=-1, order=5,
        sr=3.25 * apu.m, pr=50 * apu.m, resolution=1000,
        limits_amplitude=[-5, 5] * apu.mm, path_lookup=None
            ):
        self.frequency = frequency
        self.wavel = (constants.c / frequency).to(apu.mm)
        self.nrot = nrot
        self.sign = sign
        self.sr = sr
        self.pr = pr
        self.n = order
        self.N_K_coeff = (self.n + 1) * (self.n + 2) // 2
        self.resolution = resolution
        self.limits_amplitude = limits_amplitude

        # angular and radial actuators' position
        theta = np.linspace(7.5, 360 - 7.5, 24) * apu.deg
        R = np.array([3250, 2600, 1880, 1210]) * apu.mm
        # R = np.array([3245, 2600, 1880, 1210]) * apu.mm

        # (x, y) position of actuators
        self.act_x = np.outer(R, np.cos(theta)).reshape(-1)
        self.act_y = np.outer(R, np.sin(theta)).reshape(-1)

        if astropy.__version__ < '4':
            self.act_x *= R.unit
            self.act_y *= R.unit

        if path_lookup is None:
            self.path_lookup = get_pkg_data_filename(
                '../data/lookup_effelsberg.data'
                )
        else:
            self.path_lookup = path_lookup

        self.alpha_lookup, self.actuator_sr_lookup = self.read_lookup(True)
        self.phase_pr_lookup = self.transform(self.actuator_sr_lookup)

    def read_lookup(self, interp):
        """
        Simple reader for the Effelsberg active surface look-up table.

        Parameters
        ----------
        interp : `bool`
            If `True` it will compute the interpolated maps.

        Returns
        -------
        alpha_lookup : `~astropy.units.quantity.Quantity`
            List of angles from the look-up table.
        actuator_sr_lookup : `~astropy.units.quantity.Quantity`
            Actuators surface perpendicular displacement as seen from the
            sub-reflector in the standard grid format from `~pyoof`.
        """

        # transforming zenith angels to elevation angles in look-up table
        file_open = open(self.path_lookup, "r")
        file_read = file_open.read()
        file_open.close()

        alpha_lookup = [7, 10, 20, 30, 32, 40, 50, 60, 70, 80, 90] * apu.deg
        names = [
            'NR', 'N', 'ffff'
            ] + alpha_lookup.value.astype(int).astype(str).tolist()

        lookup_table = QTable.read(
            file_read.split('**ENDE**')[0], names=names, format='ascii'
            )

        for n in names[3:]:
            lookup_table[n] = lookup_table[n] * apu.um

        if interp:
            # Generating new grid same as pyoof output
            x_ng = np.linspace(-self.sr, self.sr, self.resolution)
            y_ng = x_ng.copy()
            xx, yy = np.meshgrid(x_ng, y_ng)
            circ = [(xx ** 2 + yy ** 2) >= (self.sr) ** 2]

            # actuators displacement in the new grid
            actuator_sr_lookup = np.zeros(
                shape=(alpha_lookup.size, self.resolution, self.resolution)
                ) << apu.um

            for j, _alpha in enumerate(names[3:]):
                actuator_sr_lookup[j, :, :] = interpolate.griddata(
                    # coordinates of grid points to interpolate from
                    points=(
                        self.act_x.to_value(apu.m),
                        self.act_y.to_value(apu.m)
                        ),
                    values=lookup_table[_alpha].to_value(apu.um),
                    # coordinates of grid points to interpolate to
                    xi=tuple(
                        np.meshgrid(x_ng.to_value(apu.m), y_ng.to_value(apu.m))
                        ),
                    method='cubic'
                    ) * apu.um
                actuator_sr_lookup = np.nan_to_num(actuator_sr_lookup)
                actuator_sr_lookup[j, :, :][tuple(circ)] = 0

        else:
            actuator_sr_lookup = np.zeros(shape=(11, 96), dtype=int) << apu.um
            for j, _alpha in enumerate(names[3:]):
                actuator_sr_lookup[j, :] = lookup_table[_alpha]

        return alpha_lookup, actuator_sr_lookup

    def grav_deformation(self, g_coeff, alpha):
        """
        Simple decomposition of the telescope elastic structure and
        gravitational force into a gravitational deformation model. The model
        takes into account only the elevation angle and not azimuth (since it
        cancels out).

        Parameters
        ----------
        g_coeff : `~numpy.ndarray` or `list`
            It has the list of three gravitational/elastic coefficients to
            supply the model.
        alpha : `~astropy.units.quantity.Quantity`
            Single angle related to the three ``g_coeff`` coefficients.

        Returns
        -------
        K : `float`
            Single Zernike circle polynomial coefficient related to a single
            elevation angle ``alpha``.
        """

        if type(alpha) == apu.Quantity:
            alpha = alpha.to_value(apu.rad)

        K = (
            g_coeff[0] * np.sin(alpha) + g_coeff[1] * np.cos(alpha) +
            g_coeff[2]
            )

        return K

    def transform(self, actuator_sr):
        """
        Transformation required to get from the actuators displacement in the
        sub-reflector to the phase-error map in the primary dish.

        Parameters
        ----------
        actuator_sr : `~astropy.units.quantity.Quantity`
            Two or three dimensional array, in the `~pyoof` format, for the
            actuators displacement in the sub-reflector. It must have shape
            ``(alpha.size, resolution)`` or ``(alpha.size, resolution,
            resolution)``.

        Returns
        -------
        phase_pr : `~astropy.units.quantity.Quantity`
            Phase-error map for the primary dish. It must have shape
            ``(alpha.size, resolution, resolution)``.
        """

        if actuator_sr.ndim == 3:
            axes = (1, 2)
        else:
            axes = (0, 1)

        factor = self.sign * 4 * np.pi * apu.rad / self.wavel

        phase_pr = (
            factor * np.rot90(
                m=actuator_sr,
                axes=axes,
                k=self.nrot
                )
            ).to(apu.rad)

        return phase_pr

    def itransform(self, phase_pr):
        """
        Inverse transformation for
        `~pyoof.actuator.EffeslbergActuator.transform`.

        Parameters
        ----------
        phase_pr : `~astropy.units.quantity.Quantity`
            Phase-error map for the primary dish. It must have shape
            ``(alpha.size, resolution, resolution)``.

        Returns
        -------
        actuator_sr : `~astropy.units.quantity.Quantity`
            Two or three dimensional array, in the `~pyoof` format, for the
            actuators displacement in the sub-reflector. It must have shape
            ``(alpha.size, resolution)`` or ``(alpha.size, resolution,
            resolution)``.
        """

        if phase_pr.ndim == 3:
            axes = (1, 2)
        else:
            axes = (0, 1)

        factor = self.wavel / (self.sign * 4 * np.pi * apu.rad)

        actuator_sr = np.rot90(
            m=(phase_pr * factor).to(apu.um),
            axes=axes,
            k=-self.nrot
            )

        # replacing larger values for maximum/minimum displacement
        [min_amplitude, max_amplitude] = self.limits_amplitude
        actuator_sr[actuator_sr > max_amplitude] = max_amplitude
        actuator_sr[actuator_sr < min_amplitude] = min_amplitude

        return actuator_sr

    def interp_surface2rings(self, actuator_sr):
        """
        Bivariate spline approximation over a rectangular mesh. It
        interpolates from a two dimensional array to a one dimensional set of
        concentric rings. Result is in the same format as in the default
        look-up table, and it is ready to be written.

        Parameters
        ----------
        actuator_sr : `~astropy.units.quantity.Quantity`
            Three dimensional array, in the `~pyoof` format, for the
            actuators displacement in the sub-reflector. It must have shape
            ``(alpha.size, resolution, resolution)``.

        Returns
        -------
        lookup_table : `~astropy.units.quantity.Quantity`
            Array with shape ``lookup_table.shape = (11, 96)``, in the same
            format as in the standard look-up table at Effelsberg telescope.
        """

        # Generating new grid same as pyoof output
        x = np.linspace(-self.sr, self.sr, self.resolution)
        y = x.copy()

        lookup_table = np.zeros((11, 96)) << apu.um
        for j in range(11):

            intrp = interpolate.RectBivariateSpline(
                x.to_value(apu.mm), y.to_value(apu.mm),
                z=actuator_sr.to_value(apu.um)[j, :, :].T,
                kx=5,
                ky=5
                )

            lookup_table[j, :] = intrp(
                self.act_x.to_value(apu.mm),
                self.act_y.to_value(apu.mm),
                grid=False
                ) * apu.to(u.um)

        return lookup_table

    def write_lookup(self, fname, actuator_sr):
        """
        Easy writer for the active surface standard formatting at the
        Effelsberg telescope. The writer admits the actuator sub-reflector
        perpendicular displacement in the same shape as the `~pyoof` format
        (with the exact angle list as in ``alpha_lookup`` format), then it
        grids the data to the active surface look-up format.

        Parameters
        ----------
        fname : `str`
            String to the name and path for the look-up table to be stored.
        actuator_sr : `~astropy.units.quantity.Quantity`
            Two or three dimensional array, in the `~pyoof` format, for the
            actuators displacement in the sub-reflector. It must have shape
            ``(11, 96)`` or ``(11, resolution, resolution)``. The angles must
            be same as ``alpha_lookup`` (``alpha_lookup.size = 11``).
        """

        if actuator_sr.ndim == 3:
            lookup_table = self.interp_surface2rings(actuator_sr=actuator_sr)
        else:
            lookup_table = actuator_sr.copy()

        [min_amplitude, max_amplitude] = self.limits_amplitude
        lookup_table[lookup_table > max_amplitude] = max_amplitude
        lookup_table[lookup_table < min_amplitude] = min_amplitude
        lookup_table = lookup_table.to_value(apu.um)

        # writing the file row per row in specific format
        with open(fname, 'w') as file:
            for k in range(96):
                row = np.around(
                    lookup_table[:, k], 0).astype(np.int).astype(str)
                file.write(f'NR {k + 1} ffff ' + '  '.join(tuple(row)) + '\n')
            file.write('**ENDE**\n')

    def fit_zpoly(self, phase_pr, alpha, fem=True):
        """
        Simple Zernike circle polynomial fit to a single phase-error map. Do
        not confuse with the `~pyoof.fit_zpoly`, the later calculates the
        phase-error maps from a set of beam maps, in this case we only adjust
        polynomials to the phase, an easier process.

        Parameters
        ----------
        phase_pr : `~astropy.units.quantity.Quantity`
            Phase-error map for the primary dish. It must have shape
            ``(alpha.size, resolution, resolution)``.
        alpha : `~astropy.units.quantity.Quantity`
            List of elevation angles related to ``phase_pr.shape[0]``.
        fem : `bool`
            If ``fem`` (Finite Element Method) is `True` then the Zernike
            circle polynomials coefficients will be adjusted without tilt and
            and overall amplitude. If False it will adjust all available
            polynomials given by the ``order`` given to the class.

        Returns
        -------
        K_coeff_alpha : `~numpy.ndarray`
            Two dimensional array for the Zernike circle polynomials
            coefficients. The shape is ``(alpha.size, N_K_coeff)``.
        """
        start_time = time.time()
        print('\n ***** PYOOF FIT POLYNOMIALS ***** \n')

        if fem:
            def residual_phase(K_coeff, phase_data):
                phase_model = phase(
                    K_coeff=np.insert(K_coeff, [0] * 3, [0.] * 3),
                    pr=self.pr,
                    piston=False,
                    tilt=False,
                    resolution=self.resolution
                    )[2].to_value(apu.rad).flatten()
                return phase_data - phase_model
            K_coeff_init = np.array([0.1] * (self.N_K_coeff - 3))
            K_coeff_alpha = np.zeros((alpha.size, self.N_K_coeff - 3))

        else:
            def residual_phase(K_coeff, phase_data):
                phase_model = phase(
                    K_coeff=K_coeff,
                    pr=self.pr,
                    piston=False,
                    tilt=True,
                    resolution=self.resolution
                    )[2].to_value(apu.rad).flatten()
                return phase_data - phase_model
            K_coeff_init = np.array([0.1] * self.N_K_coeff)
            K_coeff_alpha = np.zeros((alpha.size, self.N_K_coeff))

        for _alpha in range(alpha.size):
            res_lsq_K = optimize.least_squares(
                fun=residual_phase,
                x0=K_coeff_init,
                args=(phase_pr[_alpha, ...].to_value(apu.rad).flatten(),),
                method='trf',
                tr_solver='exact'
                )
            K_coeff_alpha[_alpha, :] = res_lsq_K.x

        if K_coeff_alpha.shape[1] != self.N_K_coeff:
            K_coeff_alpha = np.insert(K_coeff_alpha, [0] * 3, [0.] * 3, 1)

        final_time = np.round((time.time() - start_time) / 60, 2)
        print(f'\n ***** PYOOF FIT COMPLETED AT {final_time} mins *****\n')

        return K_coeff_alpha

    def fit_grav_deformation(self, K_coeff_alpha, alpha):
        """
        Finds the full set for a gravitational deformation model given a list
        of elevations in ``alpha``. The list of Zernike circle polynomials
        coefficients, ``K_coeff_alpha``, must be given in the same order as
        ``alpha``.

        Parameters
        ----------
        K_coeff_alpha : `~numpy.ndarray`
            Two dimensional array for the Zernike circle polynomials
            coefficients. The shape is ``(alpha.size, N_K_coeff)``.
        alpha : `~astropy.units.quantity.Quantity`
            List of elevation angles related to ``phase_pr.shape[0]``.

        Returns
        -------
        g_coeff : `~numpy.ndarray`
            Two dimensional array for the gravitational deformation
            coefficients found in the least-squares minimization. The shape of
            the array will be given by the Zernike circle polynomial order
            ``n`` and the size of the ``g_coeff`` coefficients in
            `~pyoof.actuator.EffelsbergActuator.grav_deformation`.
        """
        start_time = time.time()
        print('\n ***** PYOOF FIT GRAVITATIONAL DEFORMATION MODEL ***** \n')

        def residual_grav_deformation(g_coeff, Knl, alpha):
            Knl_model = self.grav_deformation(g_coeff, alpha)
            return Knl - Knl_model

        # removing tilt and amplitude (K(0, 0)) terms from fit
        g_coeff = np.zeros((self.N_K_coeff, 3))
        for N in range(self.N_K_coeff):

            res_lsq_g = optimize.least_squares(
                fun=residual_grav_deformation,
                x0=[0.1, 0.1, 0.1],
                args=(K_coeff_alpha[:, N], alpha,),
                method='trf',
                tr_solver='exact'
                )
            g_coeff[N, :] = res_lsq_g.x

        final_time = np.round((time.time() - start_time) / 60, 2)
        print(f'\n ***** PYOOF FIT COMPLETED AT {final_time} mins *****\n')

        return g_coeff

    def fit_all(self, phase_pr, alpha):
        """
        Wrapper for all least-squares minimizations, Zernike circle
        polynomials (`~pyoof.actuator.EffelsbergActuator.fit_zpoly`) and
        gravitational deformation model
        (`~pyoof.actuator.EffelsbergActuator.fit_grav_deformation`).

        Parameters
        ----------
        phase_pr : `~astropy.units.quantity.Quantity`
            Phase-error map for the primary dish. It must have shape
            ``(alpha.size, resolution, resolution)``.
        alpha : `~astropy.units.quantity.Quantity`
            List of elevation angles related to ``phase_pr.shape[0]``.

        Returns
        -------
        g_coeff : `~numpy.ndarray`
            Two dimensional array for the gravitational deformation
            coefficients found in the least-squares minimization. The shape of
            the array will be given by the Zernike circle polynomial order
            ``n`` and the size of the ``g_coeff`` coefficients in
            `~pyoof.actuator.EffelsbergActuator.grav_deformation`.
        K_coeff_alpha : `~numpy.ndarray`
            Two dimensional array for the Zernike circle polynomials
            coefficients. The shape is ``(alpha.size, N_K_coeff)``.
        """

        K_coeff_alpha = self.fit_zpoly(phase_pr=phase_pr, alpha=alpha)
        g_coeff = self.fit_grav_deformation(
            K_coeff_alpha=K_coeff_alpha,
            alpha=alpha
            )
        return g_coeff, K_coeff_alpha

    def generate_phase_pr(self, g_coeff, alpha, eac):
        """
        Generate a set of phase for the primary reflector ``phase_pr``, given
        the gravitational deformation coefficients ``g_coeff`` for a new set
        of elevations ``alpha``.

        Parameters
        ----------
        g_coeff : `~numpy.ndarray`
            Two dimensional array for the gravitational deformation
            coefficients found in the least-squares minimization. The shape of
            the array will be given by the Zernike circle polynomial order
            ``n`` and the size of the ``g_coeff`` coefficients in
            `~pyoof.actuator.EffelsbergActuator.grav_deformation`.
        alpha : `~astropy.units.quantity.Quantity`
            List of new elevation angles.
        eac : `bool`
            If `True` it will activate the ellipsoidal actuator correction, described in `~pyoof.actuator.EffelsbergActuator.ellipsoidal_actuator_correction`.

        Returns
        -------
        phase_pr : `~astropy.units.quantity.Quantity`
            Phase-error map for the primary dish. It must have shape
            ``(alpha.size, resolution, resolution)``.
        """

        phases = np.zeros(
            shape=(alpha.size, self.resolution, self.resolution)
            ) << apu.rad

        for a, _alpha in enumerate(alpha):

            K_coeff = np.zeros((self.N_K_coeff))
            for k in range(self.N_K_coeff):

                K_coeff[k] = self.grav_deformation(
                    g_coeff=g_coeff[k, :],
                    alpha=_alpha
                    )

            phases[a, :, :] = phase(
                K_coeff=K_coeff,
                pr=self.pr,
                piston=False,
                tilt=False,
                resolution=self.resolution
                )[2]

            if eac:
                phases *= self.ellipsoidal_actuator_correction()

        return phases

    def ellipsoidal_actuator_correction(
        self, r=None, a=14.3050 * apu.m, b=7.3872 * apu.m
            ):
        """
        The truncated ellipsoidal in the sub-reflector has its actuators
        located across all its surface in 4 concentric rings. This correction
        takes into account the direction of the applied actuator displacement
        in order to decompose it in a only vertical component with respect to
        the telescope pointing axis (:math:`z_f`) and not with respect to the
        sub-reflector normal surface vector.

        Parameters
        ----------
        r : `~astropy.units.quantity.Quantity`
            Grid value for the radial variable in length units. If `None` it
            will use the default configuration.
        a : `~astropy.units.quantity.Quantity`
            Major axis ellipse in length units.
        b : `~astropy.units.quantity.Quantity`
            Minor axis ellipse in length units.

        Returns
        -------
        correction : `~numpy.ndarray`
            Two dimensional grid correction to be applied to the phase-error.
            It is just a simply multiplication of the two.
        """

        if r is None:
            x = np.linspace(-self.sr, self.sr, self.resolution)
            y = x.copy()
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx ** 2 + yy ** 2)

        # slope of the normal vector to the ellipse (sub-reflector)
        m = b ** 2 / a * np.sqrt(1 - r ** 2 / b ** 2) / r
        correction = np.sin(np.arctan(np.abs(m)))

        return correction.value

    def plot(self, data_r=None, figsize=(16, 5.5), title=None):
        """
        Simple plot function for the 11 FEM look-up tables. If ``data_r`` is
        `None` it will compute the standard FEM look-up table from Effelsberg.

        Parameters
        ----------
        data_r : `~astropy.units.quantity.Quantity` or `None`
            Phase-error map for the primary dish or actuator displacement
            sub-reflector. It must have shape ``(alpha.size, resolution,
            resolution)``. The array ``data_r`` can be in units of radians or
            length.
        figsize : `list` or `tuple`
            Width, height in inches.
        title : `str` or `None`
            Title name.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            FEM look-up table figure.
        """

        if data_r is None:
            data_r = self.phase_pr_lookup.copy()

        levels = np.linspace(-2, 2, 9) * apu.rad

        if data_r.decompose().unit == apu.rad:
            _unit = apu.rad
            radius = self.pr.copy()
            cb_title = 'Actuator phase-error rad'
        else:
            _unit = apu.um
            radius = self.sr.copy()
            cb_title = 'Actuator displacement $\\mu$s'
            factor = self.wavel / (self.sign * 4 * np.pi * apu.rad)
            levels = np.sort(levels * factor).to(_unit)

        nrow = 2
        ncol = 6

        # Make a new figure
        fig = plt.figure(constrained_layout=True, figsize=figsize)

        # Design your figure properties
        gs = GridSpec(
            nrow, ncol + 1,
            figure=fig,
            width_ratios=[1] * ncol + [0.1],
            height_ratios=[1] * nrow
            )

        ax = []
        for i in range(nrow):
            for j in range(ncol):
                ax.append(fig.add_subplot(gs[i, j]))
        ax.append(fig.add_subplot(gs[:, ncol]))

        x = np.linspace(-radius, radius, self.resolution)
        y = x.copy()

        extent = [-radius.to_value(apu.m), radius.to_value(apu.m)] * 2
        vmin, vmax = data_r.min(), data_r.max()

        for j, _alpha in enumerate(self.alpha_lookup):

            im = ax[j].imshow(
                data_r[j, ...].to_value(_unit),
                extent=extent,
                aspect='auto',
                vmin=vmin.to_value(_unit),
                vmax=vmax.to_value(_unit),
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax[j].contour(
                    x.to_value(apu.m), y.to_value(apu.m),
                    data_r[j, ...].to_value(_unit),
                    colors='k',
                    alpha=0.3,
                    levels=levels.to_value(_unit)
                    )

            patch = Patch(label=f'$\\alpha={_alpha.to_value(apu.deg)}$ deg')
            ax[j].legend(handles=[patch], loc='lower right', handlelength=0)

            ax[j].grid(False)
            ax[j].xaxis.set_major_formatter(plt.NullFormatter())
            ax[j].yaxis.set_major_formatter(plt.NullFormatter())
            ax[j].xaxis.set_ticks_position('none')
            ax[j].yaxis.set_ticks_position('none')

        cb = fig.colorbar(im, cax=ax[-1])
        cb.set_label(cb_title)
        fig.delaxes(ax[-2])

        if title is not None:
            fig.suptitle(title, fontsize=10)

        return fig
