from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from elisa import ld, settings, utils
from elisa import umpy as up
from elisa.base.surface import temperature as btemperature
from elisa.base.surface.temperature import renormalize_temperatures
from elisa.base.types import FLOAT
from elisa.binary_system import utils as bsutils
from elisa.binary_system.surface import faces as bsfaces
from elisa.logger import getLogger
from elisa.numba_functions import reflection_effect as re_numba
from elisa.types import Float
from elisa.utils import is_empty

if TYPE_CHECKING:

    from elisa.base.container import StarContainer
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.types import ComponentSelection, Int

logger = getLogger("binary_system.surface.temperature")

SurfaceDataDict: TypeAlias = dict[str, NDArray[Float]]
VisibilityDict: TypeAlias = dict[str, NDArray[np.bool]]
GammaDict: TypeAlias = dict[str, NDArray[Float]]


def redistribute_temperatures(
    in_system: OrbitalPositionContainer,
    temperatures: SurfaceDataDict,
) -> OrbitalPositionContainer:
    """Redistribute flattened temperature arrays back to stars and spots.

    The temperature arrays are expected to be ordered as stellar surface first,
    then spot surfaces in iteration order.

    :param in_system: Orbital position container to update.
    :type in_system: OrbitalPositionContainer
    :param temperatures: Temperatures from the whole surface, ordered as
        surface, spot1, spot2, ... for each component.
    :type temperatures: dict[str, NDArray[Float]]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    for component in ["primary", "secondary"]:
        star = getattr(in_system, component)
        counter = len(star.temperatures)
        star.temperatures = temperatures[component][:counter]

        if star.has_spots():
            for spot in star.spots.values():
                next_counter = counter + len(spot.temperatures)
                spot.temperatures = temperatures[component][counter:next_counter]
                counter = next_counter

    return in_system



def apply_reflection_effect(  # noqa: C901, PLR0912, PLR0915
    system: OrbitalPositionContainer,
    components_distance: Float,
    iterations: int,
) -> OrbitalPositionContainer:
    """Alter component temperatures to include the reflection effect.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param iterations: Number of reflection-effect iterations.
    :type iterations: int
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    if not settings.REFLECTION_EFFECT:
        logger.debug("reflection effect is switched off")
        return system

    if iterations <= 0:
        logger.debug(
            "number of reflections in reflection effect was set to zero or "
            "negative; reflection effect will not be calculated",
        )
        return system

    components = bsutils.component_to_list(component="all")

    xlim = bsfaces.faces_visibility_x_limits(
        system.primary.polar_radius,
        system.secondary.polar_radius,
        components_distance,
    )

    # this tests if you can use surface symmetries
    use_quarter_star_test = not system.has_pulsations() and not system.has_spots()
    vis_test_symmetry: dict[str, NDArray[np.bool] | None] = {}

    # declaring variables
    centres: SurfaceDataDict = {}
    vis_test: VisibilityDict = {}
    normals: SurfaceDataDict = {}
    faces: dict[str, NDArray[Int]] = {}
    points: SurfaceDataDict = {}
    temperatures: SurfaceDataDict = {}
    areas: SurfaceDataDict = {}
    log_g: SurfaceDataDict = {}
    # centres - dict with all centres concatenated (star and spot) into one matrix for convenience
    # vis_test - dict with bool map for centres to select only faces visible from any face on companion
    # gamma is of dimensions num_of_visible_faces_primary x num_of_visible_faces_secondary

    # selecting faces that have a chance to be visible from other component
    for component in components:
        star = getattr(system, component)

        (
            points[component],
            faces[component],
            centres[component],
            normals[component],
            temperatures[component],
            areas[component],
            log_g[component],
        ) = init_surface_variables(star)

        # test for visibility of star faces
        vis_test[component], vis_test_symmetry[component] = bsfaces.get_visibility_tests(
            centres[component],
            q_test=use_quarter_star_test,
            xlim=xlim[component],
            component=component,
            morphology=system.morphology,
        )

        if star.has_spots():
            # including spots into overall surface
            for spot in star.spots.values():
                vis_test_spot = bsfaces.visibility_test(
                    spot.face_centres,
                    xlim[component],
                    component,
                )

                # merge surface and spot face parameters into one variable
                (
                    centres[component],
                    normals[component],
                    temperatures[component],
                    areas[component],
                    vis_test[component],
                    log_g[component],
                ) = include_spot_to_surface_variables(
                    centres[component],
                    spot.face_centres,
                    normals[component],
                    spot.normals,
                    temperatures[component],
                    spot.temperatures,
                    areas[component],
                    spot.areas,
                    log_g[component],
                    spot.log_g,
                    vis_test[component],
                    vis_test_spot,
                )

    # limb darkening coefficients for each face of each component
    ldc = {
        cmp: ld.get_bolometric_ld_coefficients(
            temperatures[cmp],
            log_g[cmp],
            getattr(system, cmp).metallicity,
            getattr(system, cmp).limb_darkening_coefficients,
        )
        for cmp in components
    }

    # calculating C_A = (albedo_A / D_intB) - scalar
    # D_intB - bolometric limb darkening factor
    d_int = {
        cmp: ld.calculate_integrated_limb_darkening_factor(
            settings.LIMB_DARKENING_LAW,
            ldc[cmp],
        )
        for cmp in components
    }
    _c = {
        "primary": system.primary.albedo / d_int["primary"],
        "secondary": system.secondary.albedo / d_int["secondary"],
    }

    # setting reflection factor R = 1 + F_irradiated / F_original, initially equal to one everywhere - vector
    reflection_factor = {
        cmp: np.ones(np.sum(vis_test[cmp]), dtype=FLOAT)
        for cmp in components
    }
    counterpart = settings.BINARY_COUNTERPARTS

    # for faster convergence, reflection effect is calculated first on cooler component
    components = (
        ["primary", "secondary"]
        if system.primary.t_eff <= system.secondary.t_eff
        else ["secondary", "primary"]
    )

    # pre-calculating 4th power of visible teff for visible triangles
    teff4 = {component: np.empty(temperatures[component].shape) for component in components}
    for cmp in components:
        teff4[cmp][vis_test[cmp]] = up.power(temperatures[cmp][vis_test[cmp]], 4)

    if use_quarter_star_test:
        # calculating distances and distance vectors between, join vector is already normalized
        shp, shp_reduced = get_distance_matrix_shape(system, vis_test)
        shp_xy: tuple[Int, Int] = cast("tuple[Int, Int]", shp[:2])
        distance, join_vector = get_symmetrical_distance_matrix(
            shp,
            shp_reduced,
            centres,
            vis_test,
            vis_test_symmetry,
        )

        # calculating cos of angle gamma between face normal and join vector
        # initialising gammma matrices
        gamma = get_symmetrical_gammma(
            shp_xy,
            shp_reduced,
            normals,
            join_vector,
            vis_test,
            vis_test_symmetry,
        )

        # testing mutual visibility of faces by assigning 0 to non-visible face combination
        check_symmetric_gamma_for_negative_num(gamma, shp_reduced)

        # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
        q_ab = get_symmetrical_q_ab(shp_xy, shp_reduced, gamma, distance)

        # calculating limb darkening factor for each combination of surface faces
        d_gamma = get_symmetrical_d_gamma(shp_xy, shp_reduced, ldc, gamma)

        # calculating limb darkening factors for each combination of faces shape
        # (N_faces_primary * N_faces_secondary)
        # precalculating matrix part of reflection effect correction
        matrix_to_sum2 = {
            "primary": q_ab[:shp_reduced[0], :] * d_gamma["secondary"][:shp_reduced[0], :],
            "secondary": q_ab[:, :shp_reduced[1]] * d_gamma["primary"][:, :shp_reduced[1]],
        }
        symmetry_to_use = {"primary": shp_reduced[0], "secondary": shp_reduced[1]}

        for _ in range(iterations):
            for component in components:
                star = getattr(system, component)
                counterpart_component = settings.BINARY_COUNTERPARTS[component]
                vis_sym = vis_test_symmetry[component]
                if vis_sym is None:
                    msg = "Symmetry visibility mask is required in symmetry branch."
                    raise ValueError(msg)

                # calculation of reflection effect correction as
                # 1 + (c / t_effi) * sum_j(r_j * Q_ab * t_effj^4 * D(gamma_j) * areas_j)
                # calculating vector part of reflection effect correction
                vector_to_sum1 = (
                    reflection_factor[counterpart_component]
                    * teff4[counterpart_component][vis_test[counterpart_component]]
                    * areas[counterpart_component][vis_test[counterpart_component]]
                )
                counterpart_to_sum = (
                    up.matmul(vector_to_sum1, matrix_to_sum2["secondary"])
                    if component == "secondary"
                    else up.matmul(matrix_to_sum2["primary"], vector_to_sum1)
                )
                reflection_factor[component][:symmetry_to_use[component]] = (
                    1
                    + (_c[component][vis_sym] / teff4[component][vis_sym])
                    * counterpart_to_sum
                )

                # using symmetry to redistribute reflection factor R
                refl_fact_aux = np.empty(shape=np.shape(temperatures[component]))
                refl_fact_aux[vis_sym] = reflection_factor[component][:symmetry_to_use[component]]
                refl_fact_aux = star.mirror_face_values(refl_fact_aux)
                reflection_factor[component] = refl_fact_aux[vis_test[component]]

        for component in components:
            star = getattr(system, component)
            vis_sym = vis_test_symmetry[component]
            if vis_sym is None:
                msg = "Symmetry visibility mask is required in symmetry branch."
                raise ValueError(msg)

            # assigning new temperatures according to last iteration as
            # teff_new = teff_old * reflection_factor^0.25
            temperatures[component][vis_sym] = (
                temperatures[component][vis_sym]
                * up.power(
                    reflection_factor[component][:symmetry_to_use[component]],
                    0.25,
                )
            )
            temperatures[component] = star.mirror_face_values(temperatures[component])
    else:
        # calculating distances and distance vectors between, join vector is already normalized
        distance, join_vector = utils.calculate_distance_matrix(
            points1=centres["primary"][vis_test["primary"]],
            points2=centres["secondary"][vis_test["secondary"]],
            return_join_vector_matrix=True,
        )

        # calculating cos of angle gamma between face normal and join vector
        gamma = {
            "primary": re_numba.gamma_primary(
                normals["primary"][vis_test["primary"]],
                join_vector,
            ),
            "secondary": re_numba.gamma_secondary(
                normals["secondary"][vis_test["secondary"]],
                join_vector,
            ),
        }
        # negative sign is there because of reversed distance vector used for secondary component

        # testing mutual visibility of faces by assigning 0 to non-visible face combination
        gamma["primary"][gamma["primary"] < 0] = 0.0
        gamma["secondary"][gamma["secondary"] < 0] = 0.0

        # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
        q_ab = up.divide(
            up.multiply(gamma["primary"], gamma["secondary"]),
            up.power(distance, 2),
        )

        # calculating limb darkening factors for each combination of faces shape
        # (N_faces_primary * N_faces_secondary)
        d_gamma = {
            "primary": ld.limb_darkening_factor(
                coefficients=ldc["primary"][:, vis_test["primary"]].T,
                limb_darkening_law=settings.LIMB_DARKENING_LAW,
                cos_theta=gamma["primary"],
            ),
            "secondary": ld.limb_darkening_factor(
                coefficients=ldc["secondary"][:, vis_test["secondary"]].T,
                limb_darkening_law=settings.LIMB_DARKENING_LAW,
                cos_theta=gamma["secondary"].T,
            ).T,
        }

        # precalculating matrix part of reflection effect correction
        matrix_to_sum2 = {cmp: q_ab * d_gamma[counterpart[cmp]] for cmp in components}

        for _ in range(iterations):
            for component in components:
                counterpart_component = settings.BINARY_COUNTERPARTS[component]

                # calculation of reflection effect correction as
                # 1 + (c / t_effi) * sum_j(r_j * Q_ab * t_effj^4 * D(gamma_j) * areas_j)
                # calculating vector part of reflection effect correction
                vector_to_sum1 = (
                    reflection_factor[counterpart_component]
                    * teff4[counterpart_component][vis_test[counterpart_component]]
                    * areas[counterpart_component][vis_test[counterpart_component]]
                )
                counterpart_to_sum = (
                    up.matmul(vector_to_sum1, matrix_to_sum2["secondary"])
                    if component == "secondary"
                    else up.matmul(matrix_to_sum2["primary"], vector_to_sum1)
                )
                reflection_factor[component] = (
                    1
                    + (_c[component][vis_test[component]] / teff4[component][vis_test[component]])
                    * counterpart_to_sum
                )

        for component in components:
            # assigning new temperatures according to last iteration as
            # teff_new = teff_old * reflection_factor^0.25
            temperatures[component][vis_test[component]] = (
                temperatures[component][vis_test[component]]
                * up.power(reflection_factor[component], 0.25)
            )

    # redistributing temperatures back to the parent objects
    redistribute_temperatures(system, temperatures)
    return system



def build_temperature_distribution(
    system: OrbitalPositionContainer,
    components_distance: Float,
    component: ComponentSelection = "all",
) -> OrbitalPositionContainer:
    """Calculate temperature distribution across all surface faces.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param components_distance: Distance between components in SMA units.
    :type components_distance: Float
    :param component: Component selector.
    :type component: Literal["primary", "secondary", "all", "both"]
    :return: Updated orbital position container.
    :rtype: OrbitalPositionContainer
    """
    if is_empty(component):
        logger.debug("no component set to build temperature distribution")
        return system

    components = bsutils.component_to_list(component)

    for component_name in components:
        star = getattr(system, component_name)

        logger.debug(
            "computing effective temperature distribution on %s component name: %s",
            component_name,
            star.name,
        )

        temperatures = btemperature.calculate_effective_temperatures(
            star,
            star.potential_gradient_magnitudes,
        )
        star.temperatures = temperatures

        if star.has_spots():
            for spot_index, spot in star.spots.items():
                logger.debug(
                    "computing temperature distribution of spot %s / %s component",
                    spot_index,
                    component_name,
                )

                pgms = spot.potential_gradient_magnitudes
                spot_temperatures = (
                    spot.temperature_factor
                    * btemperature.calculate_effective_temperatures(star, pgms)
                )
                spot.temperatures = spot_temperatures

        logger.debug(
            "renormalizing temperature of components due to presence of spots "
            "in case of component %s",
            component_name,
        )
        renormalize_temperatures(star)

    if "primary" in components and "secondary" in components:
        logger.debug(
            "calculating reflection effect with %s iterations.",
            settings.REFLECTION_EFFECT_ITERATIONS,
        )
        apply_reflection_effect(
            system,
            components_distance,
            settings.REFLECTION_EFFECT_ITERATIONS,
        )

    return system



def init_surface_variables(
    star: StarContainer,
) -> tuple[
    NDArray[Float],
    NDArray[Int],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
]:
    """Copy basic surface parameters of the given star into new arrays.

    These arrays are used during reflection-effect calculations.

    :param star: Stellar container.
    :type star: StarContainer
    :return: Tuple ``(points, faces, centres, normals, temperatures, areas,
        log_g)``.
    :rtype: tuple[NDArray[Float], NDArray[Int], NDArray[Float],
        NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float]]
    """
    points, faces = star.surface_serializer()
    centres = star.face_centres.copy()
    normals = star.normals.copy()
    temperatures = star.temperatures.copy()
    log_g = star.log_g.copy()
    areas = star.areas.copy()
    return points, faces, centres, normals, temperatures, areas, log_g



def include_spot_to_surface_variables(
    centres: NDArray[Float],
    spot_centres: NDArray[Float],
    normals: NDArray[Float],
    spot_normals: NDArray[Float],
    temperatures: NDArray[Float],
    spot_temperatures: NDArray[Float],
    areas: NDArray[Float],
    spot_areas: NDArray[Float],
    log_g: NDArray[Float],
    spot_log_g: NDArray[Float],
    vis_test: NDArray[np.bool],
    vis_test_spot: NDArray[np.bool],
) -> tuple[
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[Float],
    NDArray[np.bool],
    NDArray[Float],
]:
    """Include spot-face parameters into global reflection-effect arrays.

    :param centres: Surface face centres.
    :type centres: NDArray[Float]
    :param spot_centres: Spot face centres to append to ``centres``.
    :type spot_centres: NDArray[Float]
    :param normals: Surface face normals.
    :type normals: NDArray[Float]
    :param spot_normals: Spot face normals to append to ``normals``.
    :type spot_normals: NDArray[Float]
    :param temperatures: Surface temperatures.
    :type temperatures: NDArray[Float]
    :param spot_temperatures: Spot temperatures to append to ``temperatures``.
    :type spot_temperatures: NDArray[Float]
    :param areas: Surface areas.
    :type areas: NDArray[Float]
    :param spot_areas: Spot areas to append to ``areas``.
    :type spot_areas: NDArray[Float]
    :param log_g: Surface log-g values.
    :type log_g: NDArray[Float]
    :param spot_log_g: Spot log-g values to append to ``log_g``.
    :type spot_log_g: NDArray[Float]
    :param vis_test: Surface visibility mask.
    :type vis_test: NDArray[bool]
    :param vis_test_spot: Spot visibility mask to append to ``vis_test``.
    :type vis_test_spot: NDArray[bool]
    :return: Tuple ``(centres, normals, temperatures, areas, vis_test, log_g)``.
    :rtype: tuple[NDArray[Float], NDArray[Float], NDArray[Float], NDArray[Float], NDArray[bool], NDArray[Float]]
    """
    centres = np.append(centres, spot_centres, axis=0)
    normals = np.append(normals, spot_normals, axis=0)
    temperatures = np.append(temperatures, spot_temperatures, axis=0)
    areas = np.append(areas, spot_areas, axis=0)
    log_g = np.append(log_g, spot_log_g, axis=0)
    vis_test = np.append(vis_test, vis_test_spot, axis=0)

    return centres, normals, temperatures, areas, vis_test, log_g



def get_symmetrical_distance_matrix(
    shape: tuple[int, int, int],
    shape_reduced: tuple[Int, Int],
    centres: SurfaceDataDict,
    vis_test: VisibilityDict,
    vis_test_symmetry: dict[str, NDArray[np.bool] | None],
) -> tuple[NDArray[Float], NDArray[Float]]:
    """Reduce distance-matrix computation by exploiting surface symmetry.

    :param shape: Desired shape of the join-vector matrix.
    :type shape: tuple[int, int, int]
    :param shape_reduced: Shape of the symmetry-reduced subspace.
    :type shape_reduced: tuple[Int, Int]
    :param centres: Face centres for both components.
    :type centres: dict[str, NDArray[Float]]
    :param vis_test: Visibility masks for both components.
    :type vis_test: dict[str, NDArray[np.bool]]
    :param vis_test_symmetry: Symmetry visibility masks for both components.
    :type vis_test_symmetry: dict[str, NDArray[np.bool] | None]
    :return: Tuple ``(distance, join_vector)``.
    :rtype: tuple[NDArray[Float], NDArray[Float]]
    """
    vis_primary_sym = vis_test_symmetry["primary"]
    vis_secondary_sym = vis_test_symmetry["secondary"]
    if vis_primary_sym is None or vis_secondary_sym is None:
        msg = "Symmetry visibility masks are required in symmetry branch."
        raise ValueError(msg)

    # in case of symmetries, you need to calculate only minor part of distance matrix connected with base
    # symmetry part of the both surfaces
    distance = np.empty(shape=shape[:-1])
    join_vector = np.empty(shape=shape)

    (
        distance[:shape_reduced[0], :],
        join_vector[:shape_reduced[0], :, :],
    ) = utils.calculate_distance_matrix(
        points1=centres["primary"][vis_primary_sym],
        points2=centres["secondary"][vis_test["secondary"]],
        return_join_vector_matrix=True,
    )

    aux = centres["primary"][vis_test["primary"]]
    (
        distance[shape_reduced[0]:, :shape_reduced[1]],
        join_vector[shape_reduced[0]:, :shape_reduced[1], :],
    ) = utils.calculate_distance_matrix(
        points1=aux[shape_reduced[0]:],
        points2=centres["secondary"][vis_secondary_sym],
        return_join_vector_matrix=True,
    )

    return distance, join_vector



def get_symmetrical_gammma(
    shape: tuple[Int, Int],
    shape_reduced: tuple[Int, Int],
    normals: SurfaceDataDict,
    join_vector: NDArray[Float],
    vis_test: VisibilityDict,
    vis_test_symmetry: dict[str, NDArray[np.bool] | None],
) -> GammaDict:
    """Use surface symmetries to compute cosine matrices of visibility angles.

    :param shape: Desired shape of gamma matrices.
    :type shape: tuple[Int, Int]
    :param shape_reduced: Shape of the symmetry-reduced subspace.
    :type shape_reduced: tuple[Int, Int]
    :param normals: Surface normals for both components.
    :type normals: dict[str, NDArray[Float]]
    :param join_vector: Join-vector matrix.
    :type join_vector: NDArray[Float]
    :param vis_test: Visibility masks for both components.
    :type vis_test: dict[str, NDArray[np.bool]]
    :param vis_test_symmetry: Symmetry visibility masks for both components.
    :type vis_test_symmetry: dict[str, NDArray[np.bool] | None]
    :return: Cosine matrices ``gamma`` for both components.
    :rtype: dict[str, NDArray[Float]]
    """
    vis_primary_sym = vis_test_symmetry["primary"]
    vis_secondary_sym = vis_test_symmetry["secondary"]
    if vis_primary_sym is None or vis_secondary_sym is None:
        msg = "Symmetry visibility masks are required in symmetry branch."
        raise ValueError(msg)

    gamma = {
        "primary": np.empty(shape=shape, dtype=FLOAT),
        "secondary": np.empty(shape=shape, dtype=FLOAT),
    }

    gamma["primary"][:, :shape_reduced[1]] = re_numba.gamma_primary(
        normals["primary"][vis_test["primary"]],
        join_vector[:, :shape_reduced[1], :],
    )
    gamma["primary"][:shape_reduced[0], shape_reduced[1]:] = re_numba.gamma_primary(
        normals["primary"][vis_primary_sym],
        join_vector[:shape_reduced[0], shape_reduced[1]:, :],
    )

    gamma["secondary"][:shape_reduced[0], :] = re_numba.gamma_secondary(
        normals["secondary"][vis_test["secondary"]],
        join_vector[:shape_reduced[0], :, :],
    )
    gamma["secondary"][shape_reduced[0]:, :shape_reduced[1]] = re_numba.gamma_secondary(
        normals["secondary"][vis_secondary_sym],
        join_vector[shape_reduced[0]:, :shape_reduced[1], :],
    )

    return gamma



def check_symmetric_gamma_for_negative_num(
    gamma: GammaDict,
    shape_reduced: tuple[Int, Int],
) -> None:
    """Replace negative gamma values with zero in-place.

    :param gamma: Cosine matrices for both components.
    :type gamma: dict[str, NDArray[Float]]
    :param shape_reduced: Shape of the symmetry-reduced subspace.
    :type shape_reduced: tuple[Int, Int]
    :return: ``None``.
    :rtype: None
    """
    s = gamma["primary"][:, :shape_reduced[1]]
    np.maximum(s, 0.0, out=s)
    s = gamma["primary"][:shape_reduced[0], shape_reduced[1]:]
    np.maximum(s, 0.0, out=s)
    s = gamma["secondary"][:shape_reduced[0], :]
    np.maximum(s, 0.0, out=s)
    s = gamma["secondary"][shape_reduced[0]:, :shape_reduced[1]]
    np.maximum(s, 0.0, out=s)



def get_symmetrical_d_gamma(
    shape: tuple[Int, Int],
    shape_reduced: tuple[Int, Int],
    ldc: dict[str, NDArray[Float]],
    gamma: GammaDict,
) -> GammaDict:
    """Use symmetries to compute limb-darkening factor matrices.

    :param shape: Desired shape of limb-darkening matrices.
    :type shape: tuple[Int, Int]
    :param shape_reduced: Shape of the symmetry-reduced subspace.
    :type shape_reduced: tuple[Int, Int]
    :param ldc: Limb-darkening coefficients for each component.
    :type ldc: dict[str, NDArray[Float]]
    :param gamma: Cosines of angles between the join vector and face normals.
    :type gamma: dict[str, NDArray[Float]]
    :return: Limb-darkening factor matrices.
    :rtype: dict[str, NDArray[Float]]
    """
    d_gamma = {
        "primary": np.empty(shape=shape, dtype=FLOAT),
        "secondary": np.empty(shape=shape, dtype=FLOAT),
    }

    cos_theta = gamma["primary"][:, :shape_reduced[1]]
    d_gamma["primary"][:, :shape_reduced[1]] = ld.limb_darkening_factor(
        coefficients=ldc["primary"][:, :shape[0]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta,
    )

    cos_theta = gamma["primary"][:shape_reduced[0], shape_reduced[1]:]
    d_gamma["primary"][:shape_reduced[0], shape_reduced[1]:] = ld.limb_darkening_factor(
        coefficients=ldc["primary"][:, :shape_reduced[0]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta,
    )

    cos_theta = gamma["secondary"][:shape_reduced[0], :]
    d_gamma["secondary"][:shape_reduced[0], :] = ld.limb_darkening_factor(
        coefficients=ldc["secondary"][:, :shape[1]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta.T,
    ).T

    cos_theta = gamma["secondary"][shape_reduced[0]:, :shape_reduced[1]]
    d_gamma["secondary"][shape_reduced[0]:, :shape_reduced[1]] = ld.limb_darkening_factor(
        coefficients=ldc["secondary"][:, :shape_reduced[1]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta.T,
    ).T

    return d_gamma



def get_symmetrical_q_ab(
    shape: tuple[Int, Int],
    shape_reduced: tuple[Int, Int],
    gamma: GammaDict,
    distance: NDArray[Float],
) -> NDArray[Float]:
    """Use symmetries to compute the reflection-effect quantity ``QAB``.

    The quantity is defined as::

        QAB = (cos gamma_a) * cos(gamma_b) / d**2

    :param shape: Desired shape of ``q_ab``.
    :type shape: tuple[Int, Int]
    :param shape_reduced: Shape of the symmetry-reduced subspace.
    :type shape_reduced: tuple[Int, Int]
    :param gamma: Cosine matrices for both components.
    :type gamma: dict[str, NDArray[Float]]
    :param distance: Distance matrix.
    :type distance: NDArray[Float]
    :return: ``QAB`` matrix.
    :rtype: NDArray[Float]
    """
    q_ab = np.empty(shape=shape, dtype=FLOAT)
    q_ab[:, :shape_reduced[1]] = up.divide(
        up.multiply(
            gamma["primary"][:, :shape_reduced[1]],
            gamma["secondary"][:, :shape_reduced[1]],
        ),
        up.power(distance[:, :shape_reduced[1]], 2),
    )
    q_ab[:shape_reduced[0], shape_reduced[1]:] = up.divide(
        up.multiply(
            gamma["primary"][:shape_reduced[0], shape_reduced[1]:],
            gamma["secondary"][:shape_reduced[0], shape_reduced[1]:],
        ),
        up.power(distance[:shape_reduced[0], shape_reduced[1]:], 2),
    )
    return q_ab



def get_distance_matrix_shape(
    system: OrbitalPositionContainer,
    vis_test: VisibilityDict,
) -> tuple[tuple[Int, Int, Int], tuple[Int, Int]]:
    """Calculate full and symmetry-reduced matrix shapes for reflection.

    :param system: Orbital position container.
    :type system: OrbitalPositionContainer
    :param vis_test: Visibility masks for both components.
    :type vis_test: dict[str, NDArray[np.bool]]
    :return: Tuple ``(shape, shape_reduced)``.
    :rtype: tuple[tuple[Int, Int, Int], tuple[Int, Int]]
    """
    shape = (np.sum(vis_test["primary"]), np.sum(vis_test["secondary"]), 3)
    shape_reduced = (
        np.sum(system.primary.symmetry_faces(vis_test["primary"])),
        np.sum(system.secondary.symmetry_faces(vis_test["secondary"])),
    )
    return shape, shape_reduced



@functools.cache
def _load_albedo_data() -> tuple[list, list]:
    """Load and cache albedo interpolation tables from disk.

    :return: Tuple ``(temperatures, albedos)`` of interpolation data loaded
        from ``settings.PATH_TO_ALBEDOS``.
    :rtype: tuple[list, list]
    """
    with Path(settings.PATH_TO_ALBEDOS).open("r") as f:
        data = json.load(f)
    return data["x"], data["y"]


def interpolate_albedo(temperature: Float) -> Float:
    """Interpolate the default albedo value from tabulated data.

    The interpolation data are taken from Figure 6 in Claret (2001), MNRAS
    327, 989-994.

    :param temperature: Stellar effective temperature.
    :type temperature: Float
    :return: Interpolated albedo in the interval ``(0, 1)``.
    :rtype: Float
    """
    if temperature <= 0:
        msg = "Negative temperature of the star encountered."
        raise ValueError(msg)

    interp_temps, interp_a = _load_albedo_data()
    return np.interp(np.log10(temperature), interp_temps, interp_a)
