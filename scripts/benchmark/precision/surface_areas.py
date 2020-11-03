from elisa.binary_system.system import BinarySystem
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.base.star import Star
from elisa import units as u
from elisa import const, settings

import matplotlib.pyplot as plt
import numpy as np


def sigmas(percentiles):
    return np.round([100 * (percentiles[2] - percentiles[1]) / percentiles[1],
                     -100 * (percentiles[1] - percentiles[0]) / percentiles[1]], 2)


def main():
    # surface_potential = 2.9
    # surface_potential = 3.055
    # surface_potential = 3.06
    # surface_potential = 3.065
    surface_potential = 3.2
    primary = Star(
        mass=2.0 * u.solMass,
        surface_potential=surface_potential,
        synchronicity=1.0,
        t_eff=10000 * u.K,
        gravity_darkening=1.0,
        discretization_factor=5,  # angular size (in degrees) of the surface elements
        albedo=0.6,
        metallicity=0.0,
    )

    secondary = Star(
        mass=1.2 * u.solMass,
        surface_potential=surface_potential,
        synchronicity=1.0,
        t_eff=7000 * u.K,
        gravity_darkening=1.0,
        albedo=0.6,
        metallicity=0,
    )

    # setattr(primary, "_mass", None)
    bs = BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=90 * u.deg,
        gamma=100 * u.km / u.s,
        period=0.5 * u.d,
        eccentricity=0.0,
        inclination=85 * u.deg,
        primary_minimum_time=2440000.0 * u.d,
        phase_shift=0.0,
        additional_light=0.1
    )

    # component = ['primary', 'secondary']
    component = ['primary']
    # component = ['secondary']

    discretization_methods = ['trapezoidal', 'improved_trapezoidal']
    alphas = [1.0, 0.75]
    # clrs = ['blue', 'green']
    clrs = ['gray', 'black']

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 5)

    bins = 100
    for jj, pot in enumerate([3.2, 2.9]):
        bs.primary.surface_potential = pot
        bs.secondary.surface_potential = pot
        bs.init()
        container = \
            OrbitalPositionContainer.from_binary_system(binary_system=bs, position=const.Position(0, 1.0, 0, 0, 0))

        for ii, discretization_method in enumerate(discretization_methods):
            settings.configure(MESH_GENERATOR=discretization_method)
            container.build_mesh()
            container.build_faces()
            container.build_surface_areas()

            areas = np.concatenate([getattr(container, com).areas for com in component])
            percentiles = np.percentile(areas, [100-68.27, 50, 68.27])
            percentiles3 = np.percentile(areas, [100-95, 50, 95])

            s1 = sigmas(percentiles)
            s3 = sigmas(percentiles3)
            print(f'1 sigma deviation in triangle sizes for {discretization_method} method: +{s1[0]}, {s1[1]} %')
            print(f'95 percentile in triangle sizes for {discretization_method} method: +{s3[0]}, {s3[1]} %')
            if pot == 2.9 and discretization_method == 'trapezoidal':
                continue
            hist = ax[jj].hist(areas, bins=bins, label=discretization_method.replace('_', ' '), alpha=alphas[ii],
                               color=clrs[ii])
            bins = hist[1]
        bins += 0.0002
        ax[jj].legend()

    # plt.legend()
    ax[0].text(0.00040, 350, s='a', fontsize=18)
    ax[1].text(0.00062, 146, s='b', fontsize=18)
    
    ax[0].set_xlim(0.00037, 0.00083)
    ax[1].set_xlim(0.00059, 0.00105)

    ax[0].set_ylabel('No. of faces')
    ax[0].set_xlabel('Areas')
    ax[1].set_xlabel('Areas')
    ax[1].yaxis.tick_right()
    plt.subplots_adjust(left=0.07, right=0.95, top=0.98, bottom=0.09, wspace=0.02)

    plt.show()

    # settings.configure(MESH_GENERATOR='trapezoidal')
    # settings.configure(MESH_GENERATOR='improved_trapezoidal')
    # bs.plot.mesh(components_to_plot=component,)
    # bs.plot.wireframe(components_to_plot=component,)
    bs.plot.surface(
        # components_to_plot='primary',
        components_to_plot='secondary',
        phase=0.8,
        inclination=75,
        edges=True,
        colormap=None,
        surface_colors=('gray', 'gray'),
                    )


if __name__ == '__main__':
    main()
