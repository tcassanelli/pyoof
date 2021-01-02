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

    Parameters
    ----------
    wavel : `~astropy.units.quantity.Quantity`
        Wavelength, :math:`\\lambda`, of the observation in meters.
    nrot : `int`
        This is a required rotation to apply to the phase maps (obtained
        from `~pyoof.fit_zpoly`) to get the right orientation of the active
        surface lookup table in the active surface control system.
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
        Path for the current look up table that controls the active surface
        control system. If `None` it will select the default table from the
        FEM model.
    """

    def __init__(
        self, wavel=7 * apu.mm, nrot=1, sign=-1, order=5, sr=3.25 * apu.m,
        pr=50 * apu.m, resolution=1000, limits_amplitude=[-5, 5] * apu.mm,
        path_lookup=None
            ):
        self.wavel = wavel
        self.nrot = nrot
        self.sign = sign
        self.sr = sr
        self.pr = pr
        self.n = order
        self.N_K_coeff = (self.n + 1) * (self.n + 2) // 2
        self.resolution = resolution
        self.limits_amplitude = limits_amplitude

        if path_lookup is None:
            self.path_lookup = get_pkg_data_filename(
                '../data/lookup_effelsberg.data'
                )
        else:
            self.path_lookup = path_lookup

        self.alpha_lookup, self.actuator_sr_lookup = self.read_lookup()
        self.phase_pr_lookup = self.transform(self.actuator_sr_lookup)

    def read_lookup(self):
        """
        Simple reader for the Effelsberg active surface look-up table.

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

        # Generating the mesh from technical drawings
        theta = np.linspace(7.5, 360 - 7.5, 24) * apu.deg
        R = np.array([3250, 2600, 1880, 1210]) * apu.mm

        # Actuator positions
        act_x = np.outer(R, np.cos(theta)).reshape(-1)
        act_y = np.outer(R, np.sin(theta)).reshape(-1)
        # np.outer may not preserve units in some astropy version

        # workaround units and the new astropy version
        if astropy.__version__ < '4':
            act_x *= R.unit
            act_y *= R.unit

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
                points=(act_x.to_value(apu.m), act_y.to_value(apu.m)),
                values=lookup_table[_alpha].to_value(apu.um),
                # coordinates of grid points to interpolate to
                xi=tuple(
                    np.meshgrid(x_ng.to_value(apu.m), y_ng.to_value(apu.m))
                    ),
                method='cubic'
                ) * apu.um
            actuator_sr_lookup = np.nan_to_num(actuator_sr_lookup)
            actuator_sr_lookup[j, :, :][tuple(circ)] = 0

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
            g_coeff[2] * alpha + g_coeff[3]
            )

        return K

    def transform(self, actuator_sr):
        """
        Transformation required to get from the actuators displacement in the
        sub-reflector to the phase-error map in the primary dish.

        Parameters
        ----------
        actuator_sr : `~astropy.units.quantity.Quantity`
            Two dimensional array, in the `~pyoof` format, for the actuators
            displacement in the sub-reflector. It must have shape
            ``(alpha.size, resolution, resolution)``.

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
            Two dimensional array, in the `~pyoof` format, for the actuators
            displacement in the sub-reflector. It must have shape
            ``(alpha.size, resolution, resolution)``.
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

    def write_lookup(self, fname, actuator_sr):
        """
        Easy writer for the active surface standard formatting at the
        Effelsberg telescope. The writer admits the actuator sub-reflector
        perpendicular displacement in the same shape as the `~pyoof` format (
        with the exact angle list as in ``alpha_lookup`` format), then it
        grids the data to the active surface look-up format.

        Parameters
        ----------
        fname : `str`
            String to the name and path for the look-up table to be stored.
        actuator_sr : `~astropy.units.quantity.Quantity`
            Two dimensional array, in the `~pyoof` format, for the actuators
            displacement in the sub-reflector. It must have shape
            ``(alpha.size, resolution, resolution)``. The angles must be same
            as ``alpha_lookup``.
        """

        # Generating the mesh from technical drawings
        theta = np.linspace(7.5, 360 - 7.5, 24) * apu.deg

        # slightly different at the edge
        R = np.array([3245, 2600, 1880, 1210]) * apu.mm

        # Actuator positions
        act_x = np.outer(R, np.cos(theta)).reshape(-1)
        act_y = np.outer(R, np.sin(theta)).reshape(-1)

        # Generating new grid same as pyoof output
        x = np.linspace(-self.sr, self.sr, self.resolution)
        y = x.copy()

        lookup_table = np.zeros((11, 96))
        for j in range(11):

            intrp = interpolate.RectBivariateSpline(
                x.to_value(apu.mm), y.to_value(apu.mm),
                z=actuator_sr.to_value(apu.um)[j, :, :].T,
                kx=5, ky=5
                )

            lookup_table[j, :] = intrp(
                act_x.to_value(apu.mm),
                act_y.to_value(apu.mm),
                grid=False
                )

        # after interpolation there may be values that have higher amplitude
        # than the lookup table maximum, we need to correct this
        [min_amplitude, max_amplitude] = self.limits_amplitude.to_value(apu.um)
        lookup_table[lookup_table > max_amplitude] = max_amplitude
        lookup_table[lookup_table < min_amplitude] = min_amplitude

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
        print(
            '\n***** PYOOF FIT COMPLETED AT {} mins *****\n'.format(final_time)
            )

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
        g_coeff = np.zeros((self.N_K_coeff, 4))
        for N in range(self.N_K_coeff):

            res_lsq_g = optimize.least_squares(
                fun=residual_grav_deformation,
                x0=[0, 0, 0, 0],
                args=(K_coeff_alpha[:, N], alpha,),
                method='trf',
                tr_solver='exact'
                )
            g_coeff[N, :] = res_lsq_g.x

        final_time = np.round((time.time() - start_time) / 60, 2)
        print(
            '\n***** PYOOF FIT COMPLETED AT {} mins *****\n'.format(final_time)
            )

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

    def generate_phase_pr(self, g_coeff, alpha):
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
                )[2]

        return phases

    def plot(self, phases=None, figsize=(16, 5.5)):

        if phases is None:
            phases = self.phase_pr_lookup

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

        x = np.linspace(-self.pr, self.pr, self.resolution)
        y = x.copy()
        levels = np.linspace(-2, 2, 9) * apu.rad
        extent = [-self.pr.to_value(apu.m), self.pr.to_value(apu.m)] * 2
        vmin, vmax = phases.min(), phases.max()

        for j, _alpha in enumerate(self.alpha_lookup):

            im = ax[j].imshow(
                phases[j, ...].to_value(apu.rad),
                extent=extent,
                aspect='auto',
                vmin=vmin.to_value(apu.rad),
                vmax=vmax.to_value(apu.rad),
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax[j].contour(
                    x.to_value(apu.m), y.to_value(apu.m),
                    phases[j, ...].to_value(apu.rad),
                    colors='k',
                    alpha=0.3,
                    levels=levels.to_value(apu.rad)
                    )

            patch = Patch(label=f'$\\alpha={_alpha.to_value(apu.deg)}$ deg')
            ax[j].legend(handles=[patch], loc='lower right', handlelength=0)

            ax[j].grid(False)
            ax[j].xaxis.set_major_formatter(plt.NullFormatter())
            ax[j].yaxis.set_major_formatter(plt.NullFormatter())
            ax[j].xaxis.set_ticks_position('none')
            ax[j].yaxis.set_ticks_position('none')

        cb = fig.colorbar(im, cax=ax[-1])
        cb.set_label('Phase rad')
        fig.delaxes(ax[-2])

        return fig
