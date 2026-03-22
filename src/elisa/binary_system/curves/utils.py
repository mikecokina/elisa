from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import atm, const, settings
from elisa.base.curves.utils import get_component_limbdarkening_cfs
from elisa.binary_system import radius as bsradius
from elisa.binary_system import utils as butils
from elisa.observer.passband import init_bolometric_passband

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.binary_system.system import BinarySystem
    from elisa.types import NP_BOOL_, ComponentSelection, Float


def get_limbdarkening_cfs(
    system: OrbitalPositionContainer,
    component: ComponentSelection = "all",
    **kwargs: Any,
) -> dict[str, dict[str, NDArray[Float]]]:
    """Return limb-darkening coefficients for each face of each component.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: ComponentSelection
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Limb-darkening coefficients for each requested component.
    :rtype: dict[str, dict[str, NDArray[Float]]]
    """
    components = butils.component_to_list(component)
    symmetry_test = not system.has_spots() and not system.has_pulsations()

    return {
        component_name: get_component_limbdarkening_cfs(
            getattr(system, component_name),
            passbands=kwargs["passband"],
            symmetry_test=symmetry_test,
        )
        for component_name in components
    }


def _get_normal_radiance(
    system: OrbitalPositionContainer,
    component: ComponentSelection = "all",
    **kwargs: Any,
) -> dict[str, dict[str, NDArray[Float]]]:
    """Compute normal radiance for all faces of selected components.

    This function evaluates normal radiance for all faces in
    ``elisa.binary_system.container.OrbitalPositionContainer``.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param component: Component selector.
    :type component: ComponentSelection
    :param kwargs: Arguments passed into light-curve generator functions.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Normal radiance values for selected components.
    :rtype: dict[str, dict[str, NDArray[Float]]]
    """
    components = butils.component_to_list(component)
    symmetry_test: dict[str, bool] = {}
    for component_name in components:
        comp = getattr(system, component_name)
        symmetry_test[component_name] = not comp.has_spots() and not comp.has_pulsations()

    temperatures: dict[str, NDArray[Float]] = {}
    log_g: dict[str, NDArray[Float]] = {}

    # utilizing surface symmetry in case of a clear surface
    for component_name in components:
        component_instance = getattr(system, component_name)
        if symmetry_test[component_name]:
            temperatures[component_name] = component_instance.symmetry_faces(
                component_instance.temperatures,
            )
            log_g[component_name] = component_instance.symmetry_faces(
                component_instance.log_g,
            )
        else:
            temperatures[component_name] = component_instance.temperatures
            log_g[component_name] = component_instance.log_g

    retval: dict[str, dict[str, NDArray[Float]]] = {}
    for component_name in components:
        comp = getattr(system, component_name)
        retval[component_name] = atm.NaiveInterpolatedAtm.radiance(
            **dict(
                temperature=temperatures[component_name],
                log_g=log_g[component_name],
                metallicity=comp.metallicity,
                atlas=(comp.atmosphere or settings.ATM_ATLAS),
                **kwargs,
            ),
        )

    # mirroring symmetrical part back to the rest of the surface
    for component_name in components:
        if symmetry_test[component_name]:
            retval[component_name] = {
                fltr: getattr(system, component_name).mirror_face_values(vals)
                for fltr, vals in retval[component_name].items()
            }

    return retval


def prep_surface_params(
    system: OrbitalPositionContainer,
    *,
    return_values: bool = True,
    write_to_containers: bool = False,
    **kwargs: Any,
) -> (
    tuple[
        dict[str, dict[str, NDArray[Float]]],
        dict[str, dict[str, NDArray[Float]]],
    ]
    | None
):
    """Prepare normal radiances and limb-darkening coefficients.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param return_values: Whether normal radiances and limb-darkening
        coefficients should be returned.
    :type return_values: bool
    :param write_to_containers: Whether calculated values should be assigned to
        the ``system`` container.
    :type write_to_containers: bool
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Tuple of normal radiances and limb-darkening coefficients, or
        ``None``.
    :rtype: tuple[dict[str, dict[str, NDArray[Float]]], dict[str, dict[str, NDArray[Float]]]] | None
    """
    # obtain limb darkening factor for each face
    ld_cfs = get_limbdarkening_cfs(system, **kwargs)

    # compute normal radiance for each face and each component
    normal_radiance = _get_normal_radiance(system, **kwargs)

    # checking if `bolometric` filter is already used
    if "bolometric" in ld_cfs["primary"]:
        bol_ld_cfs = {
            component_name: {"bolometric": ld_cfs[component_name]["bolometric"]}
            for component_name in settings.BINARY_COUNTERPARTS
        }
    else:
        passband, left_bandwidth, right_bandwidth = init_bolometric_passband()
        bol_kwargs = {
            "passband": {"bolometric": passband},
            "left_bandwidth": left_bandwidth,
            "right_bandwith": right_bandwidth,
            "atlas": "whatever",
        }
        bol_ld_cfs = get_limbdarkening_cfs(system, **bol_kwargs)

    normal_radiance = atm.correct_normal_radiance_to_optical_depth(
        normal_radiance,
        bol_ld_cfs,
    )

    if write_to_containers:
        for component_name in settings.BINARY_COUNTERPARTS:
            star = getattr(system, component_name)
            star.normal_radiance = normal_radiance[component_name]
            star.ld_cfs = ld_cfs[component_name]

    if return_values:
        return normal_radiance, ld_cfs
    return None


def update_surface_params(
    container: OrbitalPositionContainer,
    normal_radiance: dict[str, dict[str, NDArray[Float]]],
    ld_cfs: dict[str, dict[str, NDArray[Float]]],
    *,
    require_rebuild: bool,
    **kwargs: Any,
) -> tuple[
    OrbitalPositionContainer,
    dict[str, dict[str, NDArray[Float]]],
    dict[str, dict[str, NDArray[Float]]],
]:
    """Update surface radiances and limb-darkening coefficients.

    The function either recalculates normal radiances and limb-darkening
    coefficients or assigns old values to the container according to the
    ``require_rebuild`` condition.

    :param require_rebuild: Testing condition for recalculation of surface
        parameters.
    :type require_rebuild: bool
    :param container: Orbital position container.
    :type container: OrbitalPositionContainer
    :param normal_radiance: Previous normal-radiance values.
    :type normal_radiance: dict[str, dict[str, NDArray[Float]]]
    :param ld_cfs: Previous limb-darkening coefficients.
    :type ld_cfs: dict[str, dict[str, NDArray[Float]]]
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    :return: Updated container, normal radiances, and limb-darkening
        coefficients.
    :rtype: tuple[OrbitalPositionContainer, dict[str, dict[str, NDArray[Float]]], dict[str, dict[str, NDArray[Float]]]]
    """
    if require_rebuild:
        prepared = prep_surface_params(
            container,
            return_values=True,
            write_to_containers=True,
            **kwargs,
        )
        if prepared is None:
            message = "Surface parameter preparation unexpectedly returned None."
            raise RuntimeError(message)
        normal_radiance, ld_cfs = prepared
    else:
        for component_name in settings.BINARY_COUNTERPARTS:
            star = getattr(container, component_name)
            star.normal_radiance = normal_radiance[component_name]
            star.ld_cfs = ld_cfs[component_name]

    return container, normal_radiance, ld_cfs


def split_orbit_by_apse_line(
    orbital_motion: NDArray[Float],
    orbital_mask: NDArray[NP_BOOL_],
) -> tuple[NDArray[Float], NDArray[Float]]:
    """Split orbital positions into two groups separated by the line of apsides.

    Separation is defined by ``orbital_mask``.

    :param orbital_motion: Array representing orbital positions.
    :type orbital_motion: NDArray[Float]
    :param orbital_mask: Boolean mask which defines the separation. ``True`` is
        one side and ``False`` is the other side.
    :type orbital_mask: NDArray[bool]
    :return: Reduced orbit array and its supplement.
    :rtype: tuple[NDArray[Float], NDArray[Float]]
    """
    reduced_orbit_arr = orbital_motion[orbital_mask]
    supplement_to_reduced_arr = orbital_motion[~orbital_mask]
    return reduced_orbit_arr, supplement_to_reduced_arr


def forward_radii_from_distances(
    binary: BinarySystem,
    distances: NDArray,
    potentials: dict[str, NDArray] | None = None,
) -> NDArray[Float]:
    """Return forward radii for each component distance.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param distances: Orbital distances.
    :type distances: NDArray
    :param potentials: Corrected potentials. If ``None``, they are calculated.
    :type potentials: dict[str, NDArray] | None
    :return: Array of forward radii for each component with shape ``(2, N)``.
    :rtype: NDArray[Float]
    """
    distance_array = np.asarray(distances, dtype=np.float64)

    corrected_potentials = (
        binary.correct_potentials(
            distances=distance_array,
            component="all",
            iterations=2,
        )
        if potentials is None
        else potentials
    )

    pargs = (
        distance_array,
        corrected_potentials["primary"],
        binary.mass_ratio,
        binary.primary.synchronicity,
        "primary",
    )
    sargs = (
        distance_array,
        corrected_potentials["secondary"],
        binary.mass_ratio,
        binary.secondary.synchronicity,
        "secondary",
    )

    return np.vstack(
        (
            bsradius.calculate_forward_radii(*pargs),
            bsradius.calculate_forward_radii(*sargs),
        ),
    )


def compute_rel_d_geometry(
    binary: BinarySystem,
    radii: NDArray,
    radii_counterpart: NDArray,
) -> NDArray[Float]:
    """Estimate the maximum flux change due to a geometry change.

    The geometry change is estimated from the change in forward radius.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param radii: Forward radii.
    :type radii: NDArray
    :param radii_counterpart: Counterpart forward radii.
    :type radii_counterpart: NDArray
    :return: Relative flux change estimate.
    :rtype: NDArray[Float]
    """
    radii_array = np.asarray(radii, dtype=np.float64)
    radii_counterpart_array = np.asarray(radii_counterpart, dtype=np.float64)

    eq_radii = np.array(
        [binary.primary.equivalent_radius, binary.secondary.equivalent_radius],
        dtype=np.float64,
    )
    fwd_r_diff = np.abs(radii_counterpart_array - radii_array)

    d_flux = 2 * eq_radii[:, np.newaxis] * fwd_r_diff + fwd_r_diff**2
    total_flux = eq_radii**2
    return d_flux / np.sum(total_flux)


def relative_irradiation(
    binary: BinarySystem,
    distances: NDArray,
) -> NDArray[Float]:
    """Return an estimate of reflected-light contribution from the companion.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param distances: Orbital distances.
    :type distances: NDArray
    :return: Relative irradiation estimate.
    :rtype: NDArray[Float]
    """
    distance_array = np.asarray(distances, dtype=np.float64)

    temp_ratio4 = (binary.primary.t_eff / binary.secondary.t_eff) ** 4
    r_ratio2 = (binary.primary.equivalent_radius / binary.secondary.equivalent_radius) ** 2
    coeff = r_ratio2 * temp_ratio4
    irrad1 = np.power(binary.primary.equivalent_radius / distance_array, 2) / (1 + coeff)
    irrad2 = np.power(binary.secondary.equivalent_radius / distance_array, 2) / (1 + 1 / coeff)
    return np.vstack((irrad1, irrad2))


def compute_counterparts_rel_d_irrad(
    binary: BinarySystem,
    distances: NDArray,
    distances_counterpart: NDArray,
) -> NDArray[Float]:
    """Estimate a relative change in received irradiation from a companion.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param distances: Orbital distances.
    :type distances: NDArray
    :param distances_counterpart: Counterpart orbital distances.
    :type distances_counterpart: NDArray
    :return: Relative irradiation change.
    :rtype: NDArray[Float]
    """
    irrad = relative_irradiation(binary, distances)
    irrad_counterpart = relative_irradiation(binary, distances_counterpart)
    return np.abs(irrad - irrad_counterpart)


def compute_rel_d_irradiation(
    binary: BinarySystem,
    distances: NDArray,
) -> NDArray[Float]:
    """Estimate relative irradiation change between nearby orbital positions.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param distances: Sorted orbital distances.
    :type distances: NDArray
    :return: Relative irradiation change.
    :rtype: NDArray[Float]
    """
    irrad = relative_irradiation(binary, distances)
    return np.abs(irrad[:, 1:] - irrad[:, :-1])


def compute_rel_d_radii_from_counterparts(
    radii: NDArray,
    base_positions: NDArray[np.int_],
    mirrors: NDArray[np.int_],
) -> NDArray[Float]:
    """Return relative forward-radius differences between orbital counterparts.

    :param radii: Forward radii.
    :type radii: NDArray
    :param base_positions: Base orbital position array.
    :type base_positions: NDArray[int]
    :param mirrors: Orbital counterpart position array.
    :type mirrors: NDArray[int]
    :return: Relative changes in relative distances.
    :rtype: NDArray[Float]
    """
    radii_array = np.asarray(radii, dtype=np.float64)
    fwd_radii_base = radii_array[base_positions[:, 0]]
    fwd_radii_counterpart = radii_array[mirrors[:, 0]]
    return np.abs(fwd_radii_base - fwd_radii_counterpart) / fwd_radii_base.mean(axis=1)[:, np.newaxis]


def prepare_apsidaly_symmetric_orbit(
    binary: BinarySystem,
    azimuths: NDArray,
    phases: NDArray,
) -> tuple[NDArray[np.int_], NDArray[Float], NDArray[NP_BOOL_]]:
    """Prepare orbital positions symmetrical in terms of surface geometry.

    For each pair, the orbital position is mirrored using the apsidal line in
    order to reduce light-curve generation time.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param azimuths: Orbital azimuths of positions in which the light curve
        will be calculated.
    :type azimuths: NDArray
    :param phases: Orbital phases of positions in which the light curve will be
        calculated.
    :type phases: NDArray
    :return: Tuple containing unique phase indices, mirrored counterpart orbital
        positions, and the geometry-selection mask.
    :rtype: tuple[NDArray[int], NDArray[Float], NDArray[bool]]

    Shape::

        (numpy.array, list, numpy.array)

        - unique_phase_indices - indices that point to orbital positions from
          one half of the orbital motion divided by the apsidal line
        - orbital_motion_counterpart - positions produced by mirroring orbital
          positions given by ``unique_phase_indices``
        - orbital_motion_array_counterpart - same as
          ``orbital_motion_counterpart`` but in ``numpy.array`` form
    """
    azimuth_array = np.asarray(azimuths, dtype=np.float64)
    phase_array = np.asarray(phases, dtype=np.float64)

    azimuth_boundaries = [
        binary.argument_of_periastron,
        (binary.argument_of_periastron + const.PI) % const.FULL_ARC,
    ]

    if azimuth_boundaries[0] < azimuth_boundaries[1]:
        unique_geometry = np.logical_and(
            azimuth_array >= azimuth_boundaries[0],
            azimuth_array < azimuth_boundaries[1],
        )
    else:
        unique_geometry = np.logical_xor(
            azimuth_array <= azimuth_boundaries[0],
            azimuth_array > azimuth_boundaries[1],
        )

    unique_phase_indices = np.arange(phase_array.shape[0])[unique_geometry]
    unique_geometry_azimuths = azimuth_array[unique_geometry]
    unique_geometry_counterazimuths = (2 * binary.argument_of_periastron - unique_geometry_azimuths) % const.FULL_ARC

    kwargs = {
        "input_argument": unique_geometry_counterazimuths,
        "return_nparray": True,
        "calculate_from": "azimuth",
    }
    orbital_motion_array_counterpart = binary.calculate_orbital_motion(**kwargs)

    return unique_phase_indices, orbital_motion_array_counterpart, unique_geometry


def adjust_eclipse_width(
    true_anomalies: NDArray,
    true_anomaly_of_eclipse: Float,
) -> Float:
    """Extend eclipse angular width to smooth transitions around eclipse.

    The width is extended by the separation of true anomalies near the eclipse
    to smooth out the transition before and after eclipse.

    :param true_anomalies: True anomalies of the orbital positions.
    :type true_anomalies: NDArray
    :param true_anomaly_of_eclipse: True anomaly of the eclipse.
    :type true_anomaly_of_eclipse: Float
    :return: Adjusted eclipse width.
    :rtype: Float
    """
    true_anomaly_array = np.asarray(true_anomalies, dtype=np.float64)
    distances = np.abs(true_anomaly_array - true_anomaly_of_eclipse)
    inverse_points_mask = distances > const.PI
    distances[inverse_points_mask] = const.FULL_ARC - distances[inverse_points_mask]
    indices = np.argsort(distances)[:2]
    return 1.5 * np.abs(
        true_anomaly_array[indices[1]] - true_anomaly_array[indices[0]],
    )
