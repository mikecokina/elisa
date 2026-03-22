"""Module for generating colormaps for surface visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, settings, units, utils
from elisa.ld import limb_darkening_factor
from elisa.pulse import container_ops
from elisa.pulse import utils as putils
from elisa.utils import transform_values

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.const import Position
    from elisa.types import AstropyUnit as Unit
    from elisa.types import Float


def add_colormap_to_plt_kwargs(*args, **kwargs) -> NDArray | None:
    """Return a colormap that can be passed to surface plot kwargs.

    This function dispatches to specific colormap functions based on the
    requested colormap type and returns computed surface colormaps for
    visualization purposes.

    :param args: Positional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **colormap** (:class:`str`) - Surface colormap identifier based on physical parameter.
      See available options below.
    - **star** (:class:`elisa.base.container.StarContainer`) - Star container with surface data.
    - **phase** (:class:`float`) - Photometric phase.
    - **com_x** (:class:`float`) - Center of mass x-coordinate.
    - **system_scale** (:class:`float`) - Scaling factor of the system.
    - **inclination** (:class:`float`) - Inclination of the system.
    - **position** (:class:`elisa.const.Position`) - Position object.

    :param kwargs: Keyword arguments for colormap computation
    :type kwargs: dict

    **kwargs options:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``.
      Default: ``'linear'``.
    - **unit** (:class:`astropy.units.Unit`) - Unit for the colormap values.
      Default: ``'default'``.
    - **subtract_equilibrium** (:class:`bool`) - If ``True``, return only perturbation
      from equilibrium state. Default: ``False``.

    **available colormap options:**

    - ``'gravity_acceleration'`` - Surface distribution of gravity acceleration
    - ``'temperature'`` - Surface distribution of effective temperature
    - ``'velocity'`` - Absolute values of surface element velocities relative to observer
    - ``'radial_velocity'`` - Radial component of surface velocities relative to observer
    - ``'normal_radiance'`` - Surface element radiance perpendicular to surface
    - ``'radiance'`` - Surface element radiance in direction towards observer
    - ``'radius'`` - Distance of surface elements from center of mass
    - ``'horizontal_displacement'`` - Distribution of horizontal component of surface displacement
    - ``'horizontal_acceleration'`` - Distribution of horizontal surface acceleration component
    - ``'v_r_perturbed'`` - Radial component of pulsation velocity
    - ``'v_horizontal_perturbed'`` - Horizontal component of pulsation velocity

    :return: Colormap array for the requested physical parameter, or None
    :rtype: NDArray | None
    :raises KeyError: If colormap name is not recognized
    :raises ValueError: If ``subtract_equilibrium=True`` but star has no pulsations
    """
    star: StarContainer
    colormap: Colormap
    model_scale: Float
    inclination: Float
    position: Position

    colorbar_fn: dict = {
        "radius": r_cmap,
        "horizontal_displacement": horizonatal_displacement_cmap,
        "gravity_acceleration": g_cmap,
        "temperature": t_cmap,
        "velocity": v_cmap,
        "horizontal_acceleration": horizontal_g_pert_cmap,
        "v_r_perturbed": v_rad_pert_cmap,
        "v_horizontal_perturbed": v_horizontal_pert_cmap,
        "radial_velocity": v_rad_cmap,
        "normal_radiance": norm_radiance_cmap,
        "radiance": radiance_cmap,
    }

    colormap, star, *_, model_scale, inclination, position = args

    scale: str = kwargs.get("scale", "linear")
    unit = kwargs.get("unit", "default")
    subtract_equilibrium: bool = kwargs.get("subtract_equilibrium", False)

    if colormap is None:
        return None

    if colormap not in colorbar_fn:
        error_msg: str = f"Unknown `colormap` argument {colormap}. Options: {list(colorbar_fn.keys())}"
        raise KeyError(error_msg)

    if subtract_equilibrium and not star.has_pulsations():
        error_msg = (
            "You are trying to display surface colormap with "
            "`subtract_equilibrium=True` but the surface of the star "
            "does not oscillate."
        )
        raise ValueError(error_msg)

    args_colormap: tuple = (
        scale,
        unit,
        subtract_equilibrium,
        model_scale,
        inclination,
        position,
    )
    retval: NDArray | None = colorbar_fn[colormap](star, *args_colormap)

    return retval


def r_cmap(star: StarContainer, *args) -> NDArray:
    """Return the radius of surface points as a colormap.

    Computes the radial distance of surface elements from the center of mass,
    optionally including pulsation perturbations, and returns values suitable
    for visualization as a colormap.

    :param star: Star container with surface data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Include pulsation perturbations
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Radius values at surface face centers
    :rtype: NDArray
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    if not subtract_equilibrium:
        points: NDArray = star.points - star.com[None, :]
        value: NDArray = utils.cartesian_to_spherical(points)[:, 0]
    else:
        pargs: tuple = (star, 0.0)
        kwargs = {
            "update_container": True,
            "return_perturbation": False,
            "spherical_perturbation": False,
        }
        perturbation: NDArray = container_ops.position_perturbation(*pargs, **kwargs)
        value = perturbation[:, 0]

    value = value[star.faces].mean(axis=1) * model_scale
    unt = units.DISTANCE_UNIT if unit == "default" else unit
    value = transform_values(value, units.DISTANCE_UNIT, unt)

    return to_log(value, scale)


def horizonatal_displacement_cmap(star: StarContainer, *args) -> NDArray:
    """Return the horizontal component of surface element displacement as a colormap.

    Computes the horizontal component of pulsation-induced surface displacement
    and returns values suitable for visualization as a colormap.

    :param star: Star container with pulsation data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Not used for this colormap
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Horizontal displacement values at surface face centers
    :rtype: NDArray
    :raises ValueError: If star has no pulsations
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    if not subtract_equilibrium and not star.has_pulsations():
        error_msg: str = "Horizontal displacement colormap is relevant only for stars with pulsations."
        raise ValueError(error_msg)

    pargs: tuple = (star, 0.0)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
        "spherical_perturbation": True,
    }

    perturbation: NDArray = container_ops.position_perturbation(*pargs, **pkwargs)

    value: NDArray = putils.horizontal_component(perturbation, star.points_spherical)
    value = value[star.faces].mean(axis=1) * model_scale
    unt = units.DISTANCE_UNIT if unit == "default" else unit
    value = transform_values(value, units.DISTANCE_UNIT, unt)

    return to_log(value, scale)


def v_cmap(star: StarContainer, *args) -> NDArray:
    """Return surface element speed as a colormap.

    Computes the absolute speed of surface elements in the reference frame,
    optionally including pulsation perturbations, and returns values suitable
    for visualization as a colormap.

    :param star: Star container with velocity data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Include pulsation perturbations
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Speed values at surface points
    :rtype: NDArray
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    pargs: tuple = (star, model_scale)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
        "spherical_perturbation": False,
    }
    velocities: NDArray = (
        container_ops.velocity_perturbation(*pargs, **pkwargs)
        if subtract_equilibrium
        else star.velocities
    )
    velocities = np.linalg.norm(velocities, axis=1)
    unt = units.m / units.s if unit == "default" else unit
    value: NDArray = transform_values(velocities, units.VELOCITY_UNIT, unt)

    return to_log(value, scale)


def v_rad_cmap(
    star: StarContainer,
    *args,
) -> NDArray:
    """Return radial velocity as a colormap (with respect to the observer).

    Computes the radial component of surface element velocities relative to
    the observer, optionally including pulsation perturbations, and returns
    values suitable for visualization as a colormap.

    :param star: Star container with velocity data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Include pulsation perturbations
    - **model_scale** (:class:`float`) - Scale factor for the system
    - **inclination** (:class:`float`) - Inclination angle
    - **position** (:class:`elisa.const.Position`) - Position object

    :return: Radial velocity values at surface points
    :rtype: NDArray
    :note: Logarithmic scale is not supported for radial velocity
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float
    inclination: float
    position: const.Position

    scale, unit, subtract_equilibrium, model_scale, inclination, position = args

    pargs: tuple = (star, model_scale)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
        "spherical_perturbation": False,
    }
    velocities: NDArray = (
        utils.rotate_item(
            container_ops.velocity_perturbation(*pargs, **pkwargs),
            position,
            inclination,
        )
        if subtract_equilibrium
        else star.velocities
    )
    velocities = velocities[:, 0]
    unt = units.m / units.s if unit == "default" else unit
    value: NDArray = transform_values(velocities, units.VELOCITY_UNIT, unt)

    if scale in ["log", "logarithmic"]:
        warn_msg: str = "`log` scale is not allowed for radial velocity colormap."
        raise Warning(warn_msg)

    return value


def v_rad_pert_cmap(star: StarContainer, *args) -> NDArray:
    """Return radial component of the velocity perturbation as a colormap.

    Computes the radial component of pulsation-induced velocity perturbations
    and returns values suitable for visualization as a colormap.

    :param star: Star container with pulsation velocity data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Not used for this colormap
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Radial velocity perturbation values at surface points
    :rtype: NDArray
    :raises ValueError: If star has no pulsations
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    if not subtract_equilibrium and not star.has_pulsations():
        error_msg: str = "`v_r_perturbed` is relevant only for stars with pulsations."
        raise ValueError(error_msg)

    pargs: tuple = (star, model_scale)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
        "spherical_perturbation": True,
    }
    velocities: NDArray = container_ops.velocity_perturbation(*pargs, **pkwargs)[:, 0]
    unt = units.m / units.s if unit == "default" else unit
    value: NDArray = transform_values(velocities, units.VELOCITY_UNIT, unt)

    return to_log(value, scale)


def v_horizontal_pert_cmap(star: StarContainer, *args) -> NDArray:
    """Return horizontal component of the velocity perturbation as a colormap.

    Computes the horizontal component of pulsation-induced velocity perturbations
    and returns values suitable for visualization as a colormap.

    :param star: Star container with pulsation velocity data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Not used for this colormap
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Horizontal velocity perturbation values at surface face centers
    :rtype: NDArray
    :raises ValueError: If star has no pulsations
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    if not subtract_equilibrium and not star.has_pulsations():
        error_msg: str = "`v_horizontal_perturbed` colormap is relevant only for stars with pulsations."
        raise ValueError(error_msg)

    pargs: tuple = (star, model_scale)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
        "spherical_perturbation": True,
    }
    velocities: NDArray = container_ops.velocity_perturbation(*pargs, **pkwargs)
    face_centres_sph: NDArray = star.points_spherical[star.faces].mean(axis=1)
    velocities = (
        putils.horizontal_component(velocities, face_centres_sph, treat_poles=True)
        * model_scale
    )
    unt = units.m / units.s if unit == "default" else unit
    value: NDArray = transform_values(velocities, units.VELOCITY_UNIT, unt)

    return to_log(value, scale)


def g_cmap(star: StarContainer, *args) -> NDArray:
    """Return gravity acceleration colormap.

    Computes the surface distribution of gravitational acceleration magnitude,
    optionally including pulsation perturbations, and returns values suitable
    for visualization as a colormap.

    :param star: Star container with gravity acceleration data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Include pulsation perturbations
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Gravity acceleration values at surface points
    :rtype: NDArray
    :raises ValueError: If logarithmic scale is used with ``subtract_equilibrium=True``
    """
    scale: str
    unit: Unit
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    if subtract_equilibrium:
        if scale in ["log", "logarithmic"]:
            error_msg: str = "Logarithmic scale is not permitted with the `subtract_equilibrium=True`."
            raise ValueError(error_msg)
        pargs: tuple = (star, model_scale)
        pkwargs: dict = {
            "update_container": False,
            "return_perturbation": True,
            "spherical_perturbation": True,
        }
        g: NDArray = container_ops.gravity_acc_perturbation(*pargs, **pkwargs)[:, 0]
    else:
        log_g: NDArray = star.log_g
        g = np.power(10, log_g)

    value: NDArray = transform_values(g, units.ACCELERATION_UNIT, unit)

    return to_log(value, scale)


def horizontal_g_pert_cmap(star: StarContainer, *args) -> NDArray:
    """Return horizontal component of the acceleration perturbation as a colormap.

    Computes the horizontal component of pulsation-induced gravitational
    acceleration perturbations and returns values suitable for visualization
    as a colormap.

    :param star: Star container with gravity acceleration perturbation data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Not used for this colormap
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Horizontal gravity acceleration perturbation values at surface points
    :rtype: NDArray
    :raises ValueError: If star has no pulsations
    """
    scale: str
    unit: str
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    if not subtract_equilibrium and not star.has_pulsations():
        error_msg: str = "`horizontal_acceleration` colormap is relevant only for stars with pulsations."
        raise ValueError(error_msg)

    pargs: tuple = (star, model_scale)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
        "spherical_perturbation": True,
    }
    acceleration: NDArray = container_ops.gravity_acc_perturbation(*pargs, **pkwargs)
    face_centres_sph: NDArray = star.points_spherical[star.faces].mean(axis=1)
    acceleration = (
        putils.horizontal_component(acceleration, face_centres_sph, treat_poles=True)
        * model_scale
    )
    unt = units.ACCELERATION_UNIT if unit == "default" else unit
    value: NDArray = transform_values(acceleration, units.ACCELERATION_UNIT, unt)

    return to_log(value, scale)


def t_cmap(star: StarContainer, *args) -> NDArray:
    """Return temperature colormap.

    Computes the surface distribution of effective temperature, optionally
    including pulsation perturbations, and returns values suitable for
    visualization as a colormap.

    :param star: Star container with temperature data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Include pulsation perturbations
    - **model_scale** (:class:`float`) - Scale factor for the system

    :return: Temperature values at surface points
    :rtype: NDArray
    """
    scale: str
    unit: Unit
    subtract_equilibrium: bool
    model_scale: float

    scale, unit, subtract_equilibrium, model_scale = args[:4]

    pargs: tuple = (star, model_scale)
    pkwargs: dict = {
        "update_container": False,
        "return_perturbation": True,
    }
    temperatures: NDArray = (
        container_ops.temp_perturbation(*pargs, **pkwargs)
        if subtract_equilibrium
        else star.temperatures
    )
    value: NDArray = transform_values(temperatures, units.DefaultStarInputUnits.t_eff, unit)

    return to_log(value, scale)


def norm_radiance_cmap(star: StarContainer, *args) -> NDArray:
    """Return radiance in the direction of surface normal vector as a colormap.

    Computes the radiance of surface elements perpendicular to the surface
    and returns values suitable for visualization as a colormap.

    :param star: Star container with radiance data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Not used for this colormap

    :return: Radiance values at surface face centers
    :rtype: NDArray
    """
    scale: str
    unit: Unit

    scale, unit = args[:2]

    normal_radiance: NDArray = star.normal_radiance["bolometric"]
    value: NDArray = transform_values(normal_radiance, units.RADIANCE_UNIT, unit)

    return to_log(value, scale)


def radiance_cmap(star: StarContainer, *args) -> NDArray:
    """Return radiance in the direction of the observer as a colormap.

    Computes the radiance of surface elements in the direction towards the
    observer, accounting for limb darkening effects, and returns values suitable
    for visualization as a colormap.

    :param star: Star container with radiance data
    :type star: StarContainer
    :param args: Additional arguments for colormap computation
    :type args: tuple

    **args contents:**

    - **scale** (:class:`str`) - Scale type: ``'log'``, ``'logarithmic'``, or ``'linear'``
    - **unit** - Unit for the colormap values
    - **subtract_equilibrium** (:class:`bool`) - Not used for this colormap

    :return: Radiance values at surface face centers with limb darkening applied
    :rtype: NDArray
    :note: Logarithmic scale is not supported for radiance
    """
    scale: str
    unit: Unit

    scale, unit = args[:2]

    normal_radiance: NDArray = star.normal_radiance["bolometric"]
    los_cosines: NDArray = star.los_cosines
    indices: NDArray = star.indices
    ld_cfs: NDArray = star.ld_cfs["bolometric"][indices]

    ld_cors: NDArray = limb_darkening_factor(
        coefficients=ld_cfs,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=los_cosines[indices],
    )

    retval: NDArray = np.zeros(normal_radiance.shape)
    retval[indices] = normal_radiance[indices] * los_cosines[indices] * ld_cors

    value: NDArray = transform_values(retval, units.RADIANCE_UNIT, unit)

    if scale in ["log", "logarithmic"]:
        warn_msg: str = "`log` scale is not allowed for radiance colormap."
        raise Warning(warn_msg)

    return value


def to_log(value: NDArray | float, scale: str) -> NDArray | float:
    """Transform values to logarithmic scale when requested.

    Applies a base-10 logarithmic transformation to the input values if the
    scale type is logarithmic, otherwise returns the values unchanged.

    :param value: Input values to potentially transform
    :type value: NDArray | float
    :param scale: Scale type for transformation
    :type scale: str

    **scale options:**

    - ``'log'`` or ``'logarithmic'`` - Apply log10 transformation
    - ``'linear'`` or any other string - Return values unchanged

    :return: Transformed or original values depending on scale
    :rtype: NDArray | float
    """
    return np.log10(value) if scale in ["log", "logarithmic"] else value
