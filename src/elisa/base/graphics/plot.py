import numpy as np

from elisa import units, settings
from elisa.utils import transform_values
from elisa.ld import limb_darkening_factor


def add_colormap_to_plt_kwargs(colormap, star, scale='linear', unit='default'):
    """
    Returns a colormap that can be passed to surface plot kwargs.

    :param colormap: str; 'gravity_acceleration', 'temperature', 'velocity', 'radial_velocity', 'normal_radiance',
    'radiance'
    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    colorbar_fn = {
        'gravity_acceleration': g_cmap,
        'temperature': t_cmap,
        'velocity': v_cmap,
        'radial_velocity': v_rad_cmap,
        'normal_radiance': norm_radiance_cmap,
        'radiance': radiance_cmap,
    }
    retval = None
    if colormap is not None:
        retval = colorbar_fn[colormap](star, scale, unit)
        if colormap not in colorbar_fn.keys():
            raise KeyError(f'Unknown `colormap` argument {colormap}. Options: {colorbar_fn.keys()}')

    return retval


def g_cmap(star, scale, unit):
    """
    Returning gravity acceleration colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    log_g = getattr(star, 'log_g')
    g = np.power(10, log_g)
    value = transform_values(g, units.ACCELERATION_UNIT, unit)
    return to_log(value, scale)


def t_cmap(star, scale, unit):
    """
    Returning temperature colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    temperatures = getattr(star, 'temperatures')
    value = transform_values(temperatures, units.TEMPERATURE_UNIT, unit)
    return to_log(value, scale)


def v_cmap(star, scale, unit):
    """
    Returning speed colormap.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    velocities = np.linalg.norm(getattr(star, 'velocities'), axis=1)
    unt = units.km / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    return to_log(value, scale)


def v_rad_cmap(star, scale, unit):
    """
    Returning radial velocity colormap (with respect to the observer).

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    velocities = getattr(star, 'velocities')[:, 0]
    unt = units.km / units.s if unit == 'default' else unit
    value = transform_values(velocities, units.VELOCITY_UNIT, unt)
    if scale in ['log', 'logarithmic']:
        raise Warning("`log` scale is not allowed for radial velocity colormap.")
    return value


def norm_radiance_cmap(star, scale, unit):
    """
    Returning radiance in the direction of surface normal vector.

    :param star: elisa.base.container.StarContainer;
    :param scale: str; log or linear
    :param unit: astropy.units.Unit;
    :return: numpy.array;
    """
    normal_radiance = getattr(star, 'normal_radiance')['bolometric']
    value = transform_values(normal_radiance, units.RADIANCE_UNIT, unit)
    return to_log(value, scale)


def radiance_cmap(star, scale, unit):
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
