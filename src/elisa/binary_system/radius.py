import numpy as np

from . import model
from .. import (
    const,
    umpy as up
)
from .. opt.fsolver import fsolve


def calculate_radius(synchronicity, mass_ratio, surface_potential, component, *args):
    """
    Function calculates radius of the star in given direction of arbitrary direction vector (in spherical
    coordinates) starting from the centre of the star.

    :param component: str; `primary` or `secondary`,
    :param synchronicity: float;
    :param mass_ratio: float;
    :param surface_potential: float; if None compoent surface potential is assumed
    :param args: Tuple;

    ::

        (
            components_distance: float - distance between components in SMA units,
            phi: float - longitudonal angle of direction vector measured from point under L_1 in
                         positive direction (in radians)
            theta: float - latitudonal angle of direction vector measured from north pole (in radians)
         )

    :return: float; radius
    """
    if component == 'primary':
        fn = model.potential_primary_fn
        precalc = model.pre_calculate_for_potential_value_primary
    elif component == 'secondary':
        fn = model.potential_secondary_fn
        precalc = model.pre_calculate_for_potential_value_secondary
    else:
        raise ValueError(f'Invalid value of `component` argument {component}. \n'
                         f'Expecting `primary` or `secondary`.')

    precalc_args = (synchronicity, mass_ratio) + args
    scipy_solver_init_value = np.array([1e-4])
    argss = ((mass_ratio,) + precalc(*precalc_args), surface_potential)
    solution, _, ier, _ = fsolve(fn, scipy_solver_init_value, full_output=True, args=argss, xtol=1e-10)

    # check for regular solution
    if ier == 1 and not up.isnan(solution[0]) and 30 >= solution[0] >= 0:
        return solution[0]
    else:
        if not (0 < solution[0] < 1.0):
            raise ValueError(f'Invalid value of radius {solution} was calculated.')
        return solution[0]


def calculate_polar_radius(synchronicity, mass_ratio, components_distance, surface_potential, component):
    """
    Radius of the star in the direction of the pole.

    :param synchronicity: float;
    :param mass_ratio: float;
    :param components_distance: float;
    :param surface_potential: float;
    :param component: str; `primary` a `secondary`
    :return: float;
    """
    args = (components_distance, 0.0, 0.0)
    return calculate_radius(synchronicity, mass_ratio, surface_potential, component, *args)


def calculate_side_radius(synchronicity, mass_ratio, components_distance, surface_potential, component):
    """
    Radius of the star in the direction perpendicular to the pole and component join vector.

    :param synchronicity: float;
    :param mass_ratio: float;
    :param components_distance: float;
    :param surface_potential: float;
    :param component: str; `primary` a `secondary`
    :return: float;
    """
    args = (components_distance, const.HALF_PI, const.HALF_PI)
    return calculate_radius(synchronicity, mass_ratio, surface_potential, component, *args)


def calculate_backward_radius(synchronicity, mass_ratio, components_distance, surface_potential, component):
    """
    Radius of the star in the direction away from the companion.

    :param synchronicity: float;
    :param mass_ratio: float;
    :param components_distance: float;
    :param surface_potential: float;
    :param component: str; `primary` a `secondary`
    :return: float;
    """
    args = (components_distance, const.PI, const.HALF_PI)
    return calculate_radius(synchronicity, mass_ratio, surface_potential, component, *args)


def calculate_forward_radius(synchronicity, mass_ratio, components_distance, surface_potential, component):
    """
    Radius of the star in the direction towards the companion.

    :param synchronicity: float;
    :param mass_ratio: float;
    :param components_distance: float;
    :param surface_potential: float;
    :param component: str; `primary` a `secondary`
    :return: float;
    """
    args = (components_distance, 0.0, const.HALF_PI)
    return calculate_radius(synchronicity, mass_ratio, surface_potential, component, *args)


def calculate_forward_radii(distances, surface_potential, mass_ratio, synchronicity, component):
    """
    Calculates forward radii for given object for given array of distances.

    :param distances: Union[numpy.array, List]: array of component distances at which to calculate
                      the forward radii of given component(s)
    :param surface_potential: float;
    :param mass_ratio: float;
    :param synchronicity: float;
    :param component: str;
    :return: Dict; Dict[str, numpy.array];
    """
    return [calculate_forward_radius(synchronicity, mass_ratio, d, surface_potential[ii], component)
            for ii, d in enumerate(distances)]
