from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from elisa import const
from elisa import umpy as up
from elisa.base.surface import gravity as bgravity
from elisa.binary_system import utils as bsutils
from elisa.logger import getLogger
from elisa.utils import is_empty

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import Float

logger = getLogger("binary_system.surface.gravity")

ComponentSelection: TypeAlias = Literal["primary", "secondary", "all", "both"]
SurfaceComponent: TypeAlias = Literal["primary", "secondary"]


def calculate_potential_gradient(
    components_distance: Float,
    component: SurfaceComponent,
    points: NDArray[Float],
    synchronicity: Float,
    mass_ratio: Float,
) -> NDArray[Float]:
    """Return outer potential gradients at the given surface points.

    :param components_distance: Component separation in SMA units.
    :type components_distance: Float
    :param component: Target component for the calculation.
    :type component: Literal["primary", "secondary"]
    :param points: Surface points at which the gradient is evaluated.
    :type points: NDArray[Float]
    :param synchronicity: Component synchronicity factor.
    :type synchronicity: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :return: Potential gradient vectors at the supplied points.
    :rtype: NDArray[Float]
    """
    r3 = up.power(np.linalg.norm(points, axis=1), 3)
    r_hat3 = up.power(
        np.linalg.norm(
            points - np.array([components_distance, 0.0, 0.0]),
            axis=1,
        ),
        3,
    )

    f2 = up.power(synchronicity, 2)

    if component == "primary":
        domega_dx = (
            -points[:, 0] / r3
            + mass_ratio * (components_distance - points[:, 0]) / r_hat3
            + f2 * (mass_ratio + 1) * points[:, 0]
            - mass_ratio / up.power(components_distance, 2)
        )
    elif component == "secondary":
        domega_dx = (
            -points[:, 0] / r3
            + mass_ratio * (components_distance - points[:, 0]) / r_hat3
            - f2 * (mass_ratio + 1) * (components_distance - points[:, 0])
            + 1 / up.power(components_distance, 2)
        )
    else:
        msg = f"Invalid value `{component}` of argument `component`.\nUse `primary` or `secondary`."
        raise ValueError(msg)

    domega_dy = -points[:, 1] * (1 / r3 + mass_ratio / r_hat3 - f2 * (mass_ratio + 1))
    domega_dz = -points[:, 2] * (1 / r3 + mass_ratio / r_hat3)

    return -np.column_stack((domega_dx, domega_dy, domega_dz))


def calculate_polar_potential_gradient_magnitude(
    components_distance: Float,
    mass_ratio: Float,
    polar_radius: Float,
    component: SurfaceComponent,
    synchronicity: Float,
) -> Float:
    """Calculate the magnitude of the polar potential gradient.

    :param components_distance: Component separation in SMA units.
    :type components_distance: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param polar_radius: Polar radius of the component.
    :type polar_radius: Float
    :param component: Target component for the calculation.
    :type component: Literal["primary", "secondary"]
    :param synchronicity: Component synchronicity factor.
    :type synchronicity: Float
    :return: Magnitude of the polar potential gradient.
    :rtype: Float
    """
    points = [0.0, 0.0, polar_radius] if component == "primary" else [components_distance, 0.0, polar_radius]
    points = np.array(points)

    r3 = up.power(np.linalg.norm(points), 3)
    r_hat3 = up.power(
        np.linalg.norm(points - np.array([components_distance, 0.0, 0.0])),
        3,
    )

    if component == "primary":
        domega_dx = mass_ratio * components_distance / r_hat3 - mass_ratio / up.power(components_distance, 2)
    elif component == "secondary":
        domega_dx = (
            -points[0] / r3
            + mass_ratio * (components_distance - points[0]) / r_hat3
            - np.power(synchronicity, 2) * (mass_ratio + 1) * (1 - points[0])
            + 1.0 / up.power(components_distance, 2)
        )
    else:
        msg = f"Invalid value `{component}` of argument `component`.\nUse `primary` or `secondary`."
        raise ValueError(msg)

    domega_dz = -points[2] * (1.0 / r3 + mass_ratio / r_hat3)
    return up.power(
        up.power(domega_dx, 2) + up.power(domega_dz, 2),
        0.5,
    )


def calculate_polar_gravity_acceleration(
    star: StarContainer,
    components_distance: Float,
    mass_ratio: Float,
    component: SurfaceComponent,
    semi_major_axis: Float,
    synchronicity: Float,
    *,
    logg: bool = False,
) -> Float:
    """Calculate polar gravity acceleration for a binary-system component.

    This is derived from the gradient of the Roche potential::

        d_Omega / dr

    using the transformation::

        g = d_Psi / dr = (G M_component / semi_major_axis**2) * d_Omega / dr

    with an additional ``1 / q`` factor for the secondary component.

    :param star: Stellar container.
    :type star: StarContainer
    :param components_distance: Component separation in SMA units.
    :type components_distance: Float
    :param mass_ratio: Binary mass ratio.
    :type mass_ratio: Float
    :param component: Target component for the calculation.
    :type component: Literal["primary", "secondary"]
    :param semi_major_axis: Semi-major axis.
    :type semi_major_axis: Float
    :param synchronicity: Component synchronicity factor.
    :type synchronicity: Float
    :param logg: If ``True``, return ``log10(g)`` instead of ``g``.
    :type logg: bool
    :return: Polar gravity acceleration or its base-10 logarithm.
    :rtype: Float
    """
    pgm = calculate_polar_potential_gradient_magnitude(
        components_distance,
        mass_ratio,
        star.polar_radius,
        component,
        synchronicity,
    )
    gradient = const.G * star.mass * pgm / up.power(semi_major_axis, 2)
    gradient = gradient / mass_ratio if component == "secondary" else gradient
    return up.log10(gradient) if logg else gradient


def build_surface_gravity(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer | None:
    """Calculate surface gravity for each face.

    The value assigned to each face is the mean of the surface gravity values
    evaluated at the corners of that face.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Component separation in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container, or ``None`` when no component
        is selected.
    :rtype: OrbitalPositionContainer | None
    """
    if is_empty(component):
        logger.debug("no component set to build surface gravity")
        return None

    if is_empty(components_distance):
        msg = "Component distance value was not supplied or is invalid."
        raise ValueError(msg)

    components = bsutils.component_to_list(component)
    mass_ratio = system.mass_ratio

    for component_name in components:
        star = getattr(system, component_name)
        synchronicity = star.synchronicity

        pgm = calculate_polar_potential_gradient_magnitude(
            components_distance,
            mass_ratio,
            star.polar_radius,
            component_name,
            star.synchronicity,
        )
        star.polar_potential_gradient_magnitude = pgm

        logger.debug(
            "computing potential gradient magnitudes distribution of %s component",
            component_name,
        )

        points, faces = bgravity.eval_args_for_magnitude_gradient(star)

        scaling_factor = const.G * system.primary.mass / system.semi_major_axis**2
        p_grad = calculate_potential_gradient(
            components_distance,
            component_name,
            points=points,
            synchronicity=synchronicity,
            mass_ratio=mass_ratio,
        )
        g_acc_vector = scaling_factor * p_grad

        gravity = (
            np.mean(np.linalg.norm(g_acc_vector, axis=1)[faces], axis=1)
            if star.symmetry_test
            else np.mean(np.linalg.norm(g_acc_vector, axis=1), axis=1)
        )

        if star.symmetry_test():
            star.potential_gradient_magnitudes = star.mirror_face_values(gravity)
        else:
            star.potential_gradient_magnitudes = gravity

        star.log_g = np.log10(star.potential_gradient_magnitudes)

        if star.has_spots():
            g_acc_vector_spot = {}
            for spot_index, spot in star.spots.items():
                logger.debug(
                    "calculating surface SI unit gravity of %s component / %s spot",
                    component_name,
                    spot_index,
                )
                logger.debug(
                    "calculating distribution of potential gradient magnitudes of spot index: %s / %s component",
                    spot_index,
                    component_name,
                )

                p_grad = calculate_potential_gradient(
                    components_distance,
                    component_name,
                    points=spot.points,
                    synchronicity=synchronicity,
                    mass_ratio=mass_ratio,
                )
                g_acc_vector_spot.update({spot_index: scaling_factor * p_grad})

                spot.potential_gradient_magnitudes = np.mean(
                    np.linalg.norm(g_acc_vector_spot[spot_index], axis=1)[spot.faces], axis=1,
                )
                spot.log_g = np.log10(spot.potential_gradient_magnitudes)

    return system
