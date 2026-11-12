"""Apply intrinsic Cartesian pulsations to flattened component surfaces.

Harmonic mathematics belongs in pulsations, physical vector normalization in
kinematics, and frame transformation/mesh application here. Stored displacement
vectors use component axes and mesh length units; observable synthesis later
rotates positions and velocities to observer axes without re-evaluating modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.base.error import raise_elisa_unexpected_error
from elisa.base.surface.faces import calculate_normals, set_all_surface_centres
from elisa.pulse import pulsations
from elisa.pulse import utils as putils
from elisa.pulse.surface import kinematics

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.container import StarContainer


def generate_harmonics(
    star_container: StarContainer,
    com_x: float,
    phase: float,
    time: float,
) -> StarContainer:
    """Generate spherical harmonics and derivatives for pulsation modes.

    Calculates spherical harmonics and Cartesian tangent gradients at
    equilibrium surface points in the shared mode-0 pulsation-axis frame.
    These harmonics are used for calculating perturbed surface properties.
    The caller supplies an equilibrium snapshot, flattened with any spot
    replacement mesh. Subtract the component center before evaluating radial
    directions. All modes share the axis and locking settings of mode 0, while
    each keeps its own temporal frequency and phase.

    :param star_container: Star container with surface mesh.
    :type star_container: elisa.base.container.StarContainer
    :param com_x: Component center x coordinate in mesh length units.
    :type com_x: float
    :param phase: Caller-supplied rotational/orbital phase for axis orientation.
    :type phase: float
    :param time: Harmonic time in seconds, including the emission-time correction
                 already supplied by the binary wrapper when applicable.
    :type time: float
    :returns: Star container with updated harmonics on all pulsation modes.
    :rtype: elisa.base.container.StarContainer
    :raises ElisaUnexpectedError: If the caller has not flattened the star container.
    """
    # All perturbations must reference the same flattened surface, including
    # replacement spot vertices; flattening is the caller's preparation step.
    if not star_container.is_flat():
        error_msg = ("Cannot generate pulsation harmonics: the star container "
                     "has not been flattened.")
        raise_elisa_unexpected_error(error_msg)

    # Transform surface points to spherical coordinates
    coords_kwargs = {"kind": "points", "com_x": com_x}
    star_container.points_spherical = star_container.transform_points_to_spherical_coordinates(**coords_kwargs)

    # Calculate tilt angles for pulsation mode axis
    tilt_phi, tilt_theta = putils.generate_tilt_coordinates(star_container, phase)
    star_container.pulsations[0].tilt_phi = tilt_phi
    star_container.pulsations[0].tilt_theta = tilt_theta

    # Rotate Cartesian equilibrium directions rather than converting tilted
    # angles through arccos, which can round near-pole points to exact poles.
    equilibrium = star_container.points - np.array([com_x, 0., 0.])
    radii = np.linalg.norm(equilibrium, axis=1)
    directions = equilibrium/radii[:, None]
    rotation = putils.mode_rotation(tilt_phi, tilt_theta)
    # Q maps column vectors from mode axes to component axes; row-vector
    # component directions therefore enter mode axes as n_component @ Q.
    tilted_directions = directions @ rotation
    star_container.pulsation_rotation = rotation
    star_container.pulsation_directions = tilted_directions
    star_container.pulsation_equilibrium_points = equilibrium
    star_container.points_spherical[:, 0] = radii
    for mode in star_container.pulsations.values():
        exponential = putils.generate_time_exponential(mode, time)
        mode.point_harmonics, mode.point_harmonics_gradient = pulsations.cartesian_harmonics(
            mode, tilted_directions, exponential,
        )

    return star_container


def incorporate_pulsations_to_model(
    star_container: StarContainer,
    com_x: float,
    scale: float = 1.0,
) -> StarContainer:
    """Apply pulsation perturbations to surface mesh and properties.

    Calculates and incorporates perturbations to surface positions,
    velocities, accelerations, and temperatures due to pulsation modes.
    Position uses a linear Cartesian vector sum. Intrinsic velocity is added
    to the existing orbital/spin velocity, and pulsation acceleration is used
    in the existing effective-gravity prescription. Temperature uses the radial
    coefficient. Background normals and thermodynamic fields remain available
    while these perturbations are applied; centers, normals and areas are then
    recomputed from displaced vertices with the existing face connectivity.

    :param star_container: Star container with surface mesh and pulsation modes.
    :type star_container: elisa.base.container.StarContainer
    :param com_x: X-coordinate of component centre of mass.
    :type com_x: float
    :param scale: Scaling factor (semi-major axis for binary systems). Defaults to 1.0.
    :type scale: float
    :returns: Star container with updated surface properties.
    :rtype: elisa.base.container.StarContainer
    """
    # Calculate complex displacement amplitudes for all modes
    complex_displacement(star_container, scale=scale)

    # Apply perturbations to surface properties
    position_perturbation(
        star_container,
        com_x=com_x,
        update_container=True,
        return_perturbation=False,
    )
    velocity_perturbation(
        star_container,
        scale=scale,
        update_container=True,
        return_perturbation=False,
    )
    gravity_acc_perturbation(
        star_container,
        scale=scale,
        update_container=True,
        return_perturbation=False,
    )
    temp_perturbation(
        star_container,
        scale=scale,
        update_container=True,
        return_perturbation=False,
    )

    # Only now replace the equilibrium geometry: visibility, coverage and
    # surface-element integration need geometry of the displaced triangles.
    set_all_surface_centres(star_container)
    normals_args = (
        star_container.points,
        star_container.faces,
        star_container.face_centres,
        com_x,
    )
    star_container.normals = calculate_normals(*normals_args)
    star_container.calculate_all_areas()

    return star_container


def _pulsation_reference(star: StarContainer) -> tuple[NDArray, NDArray, NDArray]:
    """Require the reference frame and surface prepared by generate_harmonics."""
    rotation = star.pulsation_rotation
    directions = star.pulsation_directions
    equilibrium = star.pulsation_equilibrium_points
    if rotation is None or directions is None or equilibrium is None:
        raise_elisa_unexpected_error(
            "Cannot calculate pulsation perturbations: generate_harmonics must prepare the reference surface first.")
    return rotation, directions, equilibrium


def complex_displacement(star: StarContainer, scale: float) -> None:
    """Calculate complex displacement vector for surface points.

    Constructs vectors in pulsation axes and stores them in component Cartesian
    axes, in mesh length units. These displacements are subsequently used to
    compute kinematic quantities (position, velocity, acceleration). The radial
    scalar is stored separately for temperature perturbations; a Cartesian
    column cannot stand in for it. Norms are fixed for this snapshot.

    :param star: Star container with flattened surface mesh.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor for amplitudes.
    :type scale: float
    :raises ElisaUnexpectedError: If the caller has not flattened the star container.
    """
    # Displacements must use the same flattened vertex ordering as harmonics.
    if not star.is_flat():
        error_msg = ("Cannot calculate pulsation displacements: the star "
                     "container has not been flattened.")
        raise_elisa_unexpected_error(error_msg)

    rotation, directions, _ = _pulsation_reference(star)
    for mode in star.pulsations.values():
        displacement = kinematics.calculate_cartesian_displacement(
            mode, directions, mode.point_harmonics,
            mode.point_harmonics_gradient, star.points_spherical[:, 0], scale,
        )
        # Store all modes in the component Cartesian frame once, so position,
        # velocity and acceleration share exactly the same vector rotation.
        # For row vectors the mode-to-component transformation is D_mode @ Q.T;
        # no finite angular increments enter this transport.
        mode.complex_displacement = displacement @ rotation.T
        mode.complex_radial_displacement = kinematics.calculate_radial_displacement(
            mode, mode.point_harmonics)/scale


def position_perturbation(
    star: StarContainer,
    com_x: float = 0.0,
    *,
    update_container: bool = True,
    return_perturbation: bool = False,
) -> NDArray | None:
    """Calculate surface mesh deformation due to pulsations.

    Each stored D includes exp(-i*(omega*t+start_phase)); its real part is
    the physical displacement at this snapshot. Add Re(sum(D)) to the frozen
    component-relative equilibrium vertices and restore the component center
    only when writing absolute mesh points. Lengths remain in mesh units.

    The returned vertex vectors always have Cartesian component order
    (delta_x, delta_y, delta_z), in metres for singles or SMA units for binaries.
    Coordinate differences are not part of this physical-vector API. Diagnostic
    projections, when needed, are explicit calls to spherical_vector_components.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param com_x: X-coordinate of component center of mass. Defaults to 0.0.
    :type com_x: float
    :param update_container: If True, update star.points with perturbed positions.
                             Defaults to True.
    :type update_container: bool
    :param return_perturbation: If True, return displacement perturbation. Defaults to False.
    :type return_perturbation: bool
    :returns: Displacement perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    _, _, equilibrium = _pulsation_reference(star)
    displacement = np.zeros_like(equilibrium)
    for mode in star.pulsations.values():
        displacement += mode.complex_displacement.real
    points = equilibrium + displacement
    if update_container:
        star.points = points + np.array([com_x, 0., 0.])
    if return_perturbation:
        return displacement
    return None


def velocity_perturbation(
    star: StarContainer,
    scale: float,
    *,
    update_container: bool = False,
    return_perturbation: bool = False,
    point_perturbations: bool = False,
) -> NDArray | None:
    """Calculate velocity perturbation on pulsating star surface.

    Return sum(omega*Im(D)) in m/s along component Cartesian axes, with
    equilibrium geometry and mode orientation frozen during differentiation.
    Updating the container adds face-averaged vectors to the existing equilibrium
    velocity; no spin, orbital or moving-axis transport contribution is derived
    here. Observer rotation re-expresses this vector but does not make it the
    inertial derivative of phase-dependent geometry.

    Returned vectors always have Cartesian component order (v_x, v_y, v_z),
    in m/s, either at vertices or averaged onto faces. Diagnostic projections
    are explicit calls to spherical_vector_components; face vectors must be
    projected at the equilibrium face centers after Cartesian averaging.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor (semi-major axis for binary systems).
    :type scale: float
    :param update_container: If True, add perturbation to star.velocities.
                             Defaults to False.
    :type update_container: bool
    :param return_perturbation: If True, return velocity perturbation. Defaults to False.
    :type return_perturbation: bool
    :param point_perturbations: If True, return point values (not face-averaged).
                                Defaults to False.
    :type point_perturbations: bool
    :returns: Velocity perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    # Harmonic derivative at frozen equilibrium coordinates and frozen mode
    # orientation. This is the existing intrinsic pulsational velocity, expressed
    # along component axes. Phase transport and observer rotations reorient it;
    # they do not add rotational transport, Coriolis or orbital terms.
    _, _, equilibrium = _pulsation_reference(star)
    velocity = np.zeros_like(equilibrium)
    for mode in star.pulsations.values():
        velocity += kinematics.calculate_mode_derivatives(
            mode.complex_displacement, mode.angular_frequency)
    velocity *= scale
    if update_container:
        star.velocities += velocity[star.faces].mean(axis=1)
    if return_perturbation:
        if not point_perturbations:
            velocity = velocity[star.faces].mean(axis=1)
        return velocity
    return None


def gravity_acc_perturbation(
    star: StarContainer,
    scale: float,
    *,
    update_container: bool = False,
    return_perturbation: bool = False,
    point_perturbations: bool = False,
) -> NDArray | None:
    """Calculate gravitational acceleration perturbation on pulsating star.

    Return -sum(omega**2*Re(D)) in m/s**2 along component axes, using the
    same frozen-frame derivative as velocity_perturbation. The optional update
    adds the face-averaged acceleration to the inward equilibrium effective
    gravity vector -10**log_g*normal, then stores log10 of its magnitude.
    This retains the existing effective-gravity prescription; it does not
    recompute a gravitational potential or add equilibrium tidal-flow dynamics.

    Returned vectors always have Cartesian component order (a_x, a_y, a_z),
    in m/s**2, either at vertices or averaged onto faces. Spherical-basis
    diagnostics require an explicit spherical_vector_components projection.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor (semi-major axis for binary systems).
    :type scale: float
    :param update_container: If True, update star.log_g. Defaults to False.
    :type update_container: bool
    :param return_perturbation: If True, return acceleration perturbation. Defaults to False.
    :type return_perturbation: bool
    :param point_perturbations: If True, return point values (not face-averaged).
                                Defaults to False.
    :type point_perturbations: bool
    :returns: Acceleration perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    # Intrinsic second harmonic derivative, under the same frozen-frame
    # convention as velocity_perturbation; equilibrium gravity is unchanged.
    _, _, equilibrium = _pulsation_reference(star)
    acceleration = np.zeros_like(equilibrium)
    for mode in star.pulsations.values():
        acceleration += kinematics.calculate_mode_second_derivatives(
            mode.complex_displacement, mode.angular_frequency)
    acceleration *= scale
    if update_container:
        # Incorporation rebuilds normals afterward, so this update uses the
        # equilibrium normal together with the existing equilibrium gravity.
        g_eq = -np.power(10, star.log_g)[:, None]*star.normals
        star.log_g = np.log10(np.linalg.norm(g_eq + acceleration[star.faces].mean(axis=1), axis=1))
    if return_perturbation:
        if not point_perturbations:
            acceleration = acceleration[star.faces].mean(axis=1)
        return acceleration
    return None


def temp_perturbation(
    star: StarContainer,
    scale: float = 1.0,
    *,
    update_container: bool = False,
    return_perturbation: bool = False,
    point_perturbations: bool = False,
) -> NDArray | None:
    """Calculate temperature perturbation on pulsating star surface.

    Sum the dimensionless temperature factors from each mode's radial
    coefficient, including its configured thermal phase shift. Face perturbations
    use the mean vertex factor times the existing local face temperature. Point diagnostics
    instead use the reference stellar t_eff; they are not a new pointwise
    temperature field used by atmosphere synthesis.

    :param star: Star container with surface mesh and pulsations.
    :type star: elisa.base.container.StarContainer
    :param scale: Scaling factor (semi-major axis for binary systems). Defaults to 1.0.
    :type scale: float
    :param update_container: If True, add perturbation to star.temperatures.
                             Defaults to False.
    :type update_container: bool
    :param return_perturbation: If True, return temperature perturbation. Defaults to False.
    :type return_perturbation: bool
    :param point_perturbations: If True, return point values (not face-averaged).
                                Defaults to False.
    :type point_perturbations: bool
    :returns: Temperature perturbation array if return_perturbation is True, else None.
    :rtype: np.ndarray | None
    """
    # Sum temperature perturbations from all modes
    temp_pert = np.sum(
        [kinematics.calculate_temperature_pert_factor(mode, scale) for mode in star.pulsations.values()],
        axis=0,
    )

    # Average to face values and scale by stellar temperature
    temp_pert_face = temp_pert[star.faces].mean(axis=1) * star.temperatures

    if update_container:
        star.temperatures += temp_pert_face

    if return_perturbation:
        if point_perturbations:
            return temp_pert * star.t_eff
        return temp_pert_face

    return None


