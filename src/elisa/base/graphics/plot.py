import numpy as np

from elisa import units, settings, utils
from elisa.utils import transform_values
from elisa.ld import limb_darkening_factor
from elisa.pulse import container_ops


def add_colormap_to_plt_kwargs(*args, **kwargs):
    """
    Returns a colormap that can be passed to surface plot kwargs.

    :param args: tuple;
    :**args options**:
        * **colormap** *: str; 'gravity_acceleration', 'temperature', 'velocity', 'radial_velocity', 'normal_radiance',
                          'radiance'
        * **star** *:  elisa.base.container.StarContainer;
        * **phase** *:  float; photometric phase
        * **com_x** *:  float; centre of mass
        * **system_scale** *:  float; scaling factor of a system
    :param kwargs: Dict;
    :**kwargs options**:
        * **scale** *: str; `log` or `linear`
        * **unit** *: astropy.units.Unit;
        * **subtract_equilibrium** *: bool; if True; equilibrium values are subtracted from surface colormap
    :return: numpy.array;
    """
    colorbar_fn = {
        'radius': r_cmap,
        'gravity_acceleration': g_cmap,
        'temperature': t_cmap,
        'velocity': v_cmap,
        'radial_velocity': v_rad_cmap,
        'normal_radiance': norm_radiance_cmap,
        'radiance': radiance_cmap,
    }

    colormap, star, phase, com_x, model_scale = args
    scale = kwargs.get('scale', 'linear')
    unit = kwargs.get('unit', 'default')
    subtract_equilibrium = kwargs.get('subtract_equilibrium', False)

    retval = None
    if colormap is None:
        return retval

    if colormap not in colorbar_fn.keys():
        raise KeyError(f'Unknown `colormap` argument {colormap}. Options: {colorbar_fn.keys()}')
    if subtract_equilibrium:
        if not star.has_pulsations():
            raise ZeroDivisionError('You are trying to display surface colormap with `subtract_equilibrium`=True but '
                                    'surface of the star does not oscillate.')

    retval = colorbar_fn[colormap](star, scale, unit, subtract_equilibrium, model_scale)

    return retval


def r_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning the radius of the points as a colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :param model_scale: float; scale of the system
    :return:
    """
    if not subtract_equilibrium:
        value = star.points_spherical[:, 0]
    else:
        points_unperturbed = utils.spherical_to_cartesian(star.points_spherical)
        perturbation = container_ops.position_perturbation(star, None, update_container=False, return_perturbation=True)

        value = utils.cartesian_to_spherical(points_unperturbed + perturbation)[:, 0] - star.points_spherical[:, 0]

    value = value[star.faces].mean(axis=1) * model_scale
    unt = units.DISTANCE_UNIT if unit == 'default' else unit
    value = transform_values(value, units.DISTANCE_UNIT, unt)
    return to_log(value, scale)


def g_cmap(star, scale, unit, subtract_equilibrium):
    """
    Returning gravity acceleration colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    log_g = getattr(star, 'log_g')
    g = np.power(10, log_g)
    value = transform_values(g, units.ACCELERATION_UNIT, unit)
    return to_log(value, scale)


def t_cmap(star, scale, unit, subtract_equilibrium):
    """
    Returning temperature colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    temperatures = getattr(star, 'temperatures')
    value = transform_values(temperatures, units.TEMPERATURE_UNIT, unit)
    return to_log(value, scale)


def v_cmap(star, scale, unit, subtract_equilibrium):
    """
    Returning speed colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    phase = 0
    velocities = container_ops.velocity_perturbation(star, phase, update_container=True, return_perturbation=True) \
        if subtract_equilibrium else getattr(star, 'velocities')
    velocities = np.linalg.norm(velocities, axis=1)
    unt = units.km / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    return to_log(value, scale)


def v_rad_cmap(star, scale, unit, subtract_equilibrium):
    """
    Returning radial velocity colormap (with respect to the observer).

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    velocities = getattr(star, 'velocities')[:, 0]
    unt = units.km / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    if scale in ['log', 'logarithmic']:
        raise Warning("`log` scale is not allowed for radial velocity colormap.")
    return value


def norm_radiance_cmap(star, scale, unit, subtract_equilibrium):
    """
    Returning radiance in the direction of surface normal vector.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    normal_radiance = getattr(star, 'normal_radiance')['bolometric']
    value = transform_values(normal_radiance, units.RADIANCE_UNIT, unit)
    return to_log(value, scale)


def radiance_cmap(star, scale, unit, subtract_equilibrium):
    """
    Returning radiance in the direction of the observer.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    normal_radiance = getattr(star, 'normal_radiance')['bolometric']
    los_cosines = getattr(star, 'los_cosines')
    indices = getattr(star, 'indices')
    ld_cfs = getattr(star, 'ld_cfs')['bolometric'][indices]
    ld_cors = limb_darkening_factor(coefficients=ld_cfs,
                                    limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                    cos_theta=los_cosines[indices])
    retval = np.zeros(normal_radiance.shape)
    retval[indices] = normal_radiance[indices] * los_cosines[indices] * ld_cors

    value = transform_values(retval, units.RADIANCE_UNIT, unit)
    if scale in ['log', 'logarithmic']:
        raise Warning("`log` scale is not allowed for radiance colormap.")
    return value


def to_log(value, scale):
    """
    Function transforms to logarithmic scale.

    :param value: value: Union[float, numpy.array]; input values
    :param scale: str; `log`, `logarithmic`, `linear`
    :return: numpy.array;
    """
    return np.log10(value) if scale in ['log', 'logarithmic'] else value
