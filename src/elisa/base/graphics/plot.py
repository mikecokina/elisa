import numpy as np

from elisa import units, settings, utils
from elisa.utils import transform_values
from elisa.ld import limb_darkening_factor
from elisa.pulse import container_ops, utils as putils


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
        'horizontal_displacement': horizonatal_displacement_cmap,
        'gravity_acceleration': g_cmap,
        'temperature': t_cmap,
        'velocity': v_cmap,
        'horizontal_acceleration': horizontal_g_pert_cmap,
        'v_r_perturbed': v_rad_pert_cmap,
        'v_horizontal_perturbed': v_horizontal_pert_cmap,
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
        points = star.points - star.com[None, :]
        value = utils.cartesian_to_spherical(points)[:, 0]
    else:
        args = (star, 0.0, False, True, True)
        perturbation = container_ops.position_perturbation(*args)
        value = perturbation[:, 0]

    value = value[star.faces].mean(axis=1) * model_scale
    unt = units.DISTANCE_UNIT if unit == 'default' else unit
    value = transform_values(value, units.DISTANCE_UNIT, unt)
    return to_log(value, scale)


def horizonatal_displacement_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """

    :param star:
    :param scale:
    :param unit:
    :param subtract_equilibrium:
    :param model_scale:
    :return:
    """
    if not subtract_equilibrium and not star.has_pulsations():
        raise ValueError('Horizontal displacement colormap is relevant only for stars with pulsations.')
    args = (star, 0.0, False, True, True)
    perturbation = container_ops.position_perturbation(*args)

    value = putils.horizontal_component(perturbation, star.points_spherical)
    value = value[star.faces].mean(axis=1) * model_scale
    unt = units.DISTANCE_UNIT if unit == 'default' else unit
    value = transform_values(value, units.DISTANCE_UNIT, unt)
    return to_log(value, scale)


def v_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning speed colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :param model_scale: float; scale of the system
    :return: numpy.array;
    """
    args = (star, model_scale, False, True, False)
    velocities = container_ops.velocity_perturbation(*args) if subtract_equilibrium else getattr(star, 'velocities')
    velocities = np.linalg.norm(velocities, axis=1)
    unt = units.m / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    return to_log(value, scale)


def v_rad_pert_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning radial component of the velocity perturbation.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :param model_scale: float; scale of the system
    :return: numpy.array;
    """
    if not subtract_equilibrium and not star.has_pulsations():
        raise ValueError('`v_r_perturbed` is relevant only for stars with pulsations.')
    args = (star, model_scale, False, True, True)
    velocities = container_ops.velocity_perturbation(*args)[:, 0]
    unt = units.m / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    return to_log(value, scale)


def v_horizontal_pert_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning horizontal component of the velocity perturbation.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :param model_scale: float; scale of the system
    :return: numpy.array;
    """
    if not subtract_equilibrium and not star.has_pulsations():
        raise ValueError('`v_horizontal_perturbed` colormap is relevant only for stars with pulsations.')
    args = (star, model_scale, False, True, True)
    velocities = container_ops.velocity_perturbation(*args)
    face_centres_sph = utils.cartesian_to_spherical(star.face_centres)
    velocities = putils.horizontal_component(velocities, face_centres_sph) * model_scale
    unt = units.m / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    return to_log(value, scale)


def g_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning gravity acceleration colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    if subtract_equilibrium:
        if scale in ['log', 'logarithmic']:
            raise ValueError('Logarithmic scale is not permitted with the `subtract_equilibrium` = True.')
        args = (star, False, True, True)
        g = container_ops.gravity_acc_perturbation(*args)[:, 0]
    else:
        log_g = getattr(star, 'log_g')
        g = np.power(10, log_g)
    value = transform_values(g, units.ACCELERATION_UNIT, unit)
    return to_log(value, scale)


def horizontal_g_pert_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning horizontal component of the acceleration perturbation.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :param model_scale: float; scale of the system
    :return: numpy.array;
    """
    if not subtract_equilibrium and not star.has_pulsations():
        raise ValueError('`horizontal_acceleration` colormap is relevant only for stars with pulsations.')
    args = (star, False, True, True)
    acceleration = container_ops.gravity_acc_perturbation(*args)
    face_centres_sph = utils.cartesian_to_spherical(star.face_centres)
    acceleration = putils.horizontal_component(acceleration, face_centres_sph, treat_poles=True) * model_scale
    unt = units.ACCELERATION_UNIT if unit == 'default' else unit
    value = transform_values(acceleration, units.ACCELERATION_UNIT, unt)
    return to_log(value, scale)


def t_cmap(star, scale, unit, subtract_equilibrium, model_scale):
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


def v_rad_cmap(star, scale, unit, subtract_equilibrium, model_scale):
    """
    Returning radial velocity colormap (with respect to the observer).

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :param subtract_equilibrium: bool; if true, return only perturbation from equilibrium state
    :return: numpy.array;
    """
    args = (star, model_scale, False, True, False)
    velocities = container_ops.velocity_perturbation(*args) if subtract_equilibrium else getattr(star, 'velocities')
    velocities = velocities[:, 0]
    unt = units.m / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    if scale in ['log', 'logarithmic']:
        raise Warning("`log` scale is not allowed for radial velocity colormap.")
    return value


def norm_radiance_cmap(star, scale, unit, subtract_equilibrium, model_scale):
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


def radiance_cmap(star, scale, unit, subtract_equilibrium, model_scale):
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
