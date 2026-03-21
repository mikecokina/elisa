from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa import atm, settings
from elisa.base.curves.utils import get_component_limbdarkening_cfs
from elisa.observer.passband import init_bolometric_passband

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.container import SinglePositionContainer  # pragma: no cover
    from elisa.types import Float  # pragma: no cover


def prep_surface_params(
    system: SinglePositionContainer,
    *,
    return_values: bool = True,
    write_to_containers: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, dict[str, NDArray[Float]]], dict[str, dict[str, NDArray[Float]]]] | None:
    """Prepare normal radiance and limb darkening coefficient variables.

    Compute limb darkening coefficients and normal radiance for the supplied
    single-system position container. Optionally write calculated values back
    into the provided container.

    :param system: elisa.single_system.container.SystemContainer
    :type system: elisa.single_system.container.SinglePositionContainer
    :param return_values: If True return computed values, otherwise return None
    :type return_values: bool
    :param write_to_containers: If True assign computed values into the container
    :type write_to_containers: bool
    :param kwargs: Forwarded keyword arguments. Typical keys accepted by the
        underlying functions are documented below but any other keys are
        forwarded unchanged: passband, left_bandwidth, right_bandwidth, atlas
    :type kwargs: dict

    :return: Tuple of normal radiance and limb-darkening coefficients or None
    :rtype: tuple[dict[str, dict[str, numpy.ndarray]], dict[str, dict[str, numpy.ndarray]]] | None
    """
    # obtain limb darkening factor for each face
    ld_cfs = get_limbdarkening_cfs(system, **kwargs)

    # compute normal radiance for each face and each component
    normal_radiance = get_normal_radiance(system, **kwargs)

    # checking if "bolometric" filter is already present
    if "bolometric" in ld_cfs["star"]:
        bol_ld_cfs = {"star": {"bolometric": ld_cfs["star"]["bolometric"]}}
    else:
        # build minimal kwargs for bolometric computation; do not fabricate extra keys
        passband_obj, left_bw, right_bw = init_bolometric_passband()
        bol_kwargs = {
            "passband": {"bolometric": passband_obj},
            "left_bandwidth": left_bw,
            "right_bandwidth": right_bw,
        }
        bol_ld_cfs = get_limbdarkening_cfs(system, **bol_kwargs)

    normal_radiance = atm.correct_normal_radiance_to_optical_depth(normal_radiance, bol_ld_cfs)

    if write_to_containers:
        # write results into the container's star sub-object
        star = system.star
        star.normal_radiance = normal_radiance["star"]
        star.ld_cfs = ld_cfs["star"]

    return (normal_radiance, ld_cfs) if return_values else None


def get_normal_radiance(system: SinglePositionContainer, **kwargs: Any) -> dict[str, dict[str, NDArray[Float]]]:
    """Compute normal radiance for all faces in a single position container.

    The function forwards all additional keyword arguments to the atmosphere
    interpolation backend. Do not expand or assume specific keys in ``kwargs``
    — they are forwarded unchanged.

    :param system: elisa.single_system.container.SystemContainer
    :type system: elisa.single_system.container.SinglePositionContainer
    :param kwargs: Keyword arguments forwarded to the atmosphere radiance
        interpolator (e.g. passband, left_bandwidth, right_bandwidth, atlas)
    :type kwargs: dict
    :return: Mapping of component name to a mapping of passband name to radiance arrays
    :rtype: dict[str, dict[str, numpy.ndarray]]
    """
    star = system.star
    symmetry_test = not system.has_spots() and not system.has_pulsations()

    # utilize surface symmetry for clear surfaces
    if symmetry_test:
        temperatures = star.symmetry_faces(star.temperatures)
        log_g = star.symmetry_faces(star.log_g)
    else:
        temperatures = star.temperatures
        log_g = star.log_g

    retval = {
        "star": atm.NaiveInterpolatedAtm.radiance(
            temperature=temperatures,
            log_g=log_g,
            metallicity=star.metallicity,
            atlas=(star.atmosphere or settings.ATM_ATLAS),
            **kwargs,
        ),
    }

    if symmetry_test:
        retval["star"] = {band: star.mirror_face_values(vals) for band, vals in retval["star"].items()}

    return retval


def get_limbdarkening_cfs(system: SinglePositionContainer, **kwargs: Any) -> dict[str, dict[str, NDArray[Float]]]:
    """Return limb darkening coefficients for each face.

    All ``kwargs`` are forwarded; the function expects a ``passband`` key when
    per-passband coefficients are requested but will accept ``None`` and
    construct an empty passband list in that case. Do not expand ``kwargs``;
    pass them through or index them as needed.

    :param system: elisa.single_system.container.SystemContainer
    :type system: elisa.single_system.container.SinglePositionContainer
    :param kwargs: Forwarded keyword arguments. Expected key: 'passband'
    :type kwargs: dict
    :return: Mapping from component name to limb-darkening coefficient arrays
    :rtype: dict[str, dict[str, numpy.ndarray]]
    """
    star_container = system.star
    symmetry_test = not system.has_spots() and not system.has_pulsations()

    # extract passband mapping without changing other kwargs
    passband = kwargs.get("passband")
    passband_names = list(passband.keys()) if passband is not None else []

    return {
        "star": get_component_limbdarkening_cfs(
            star_container,
            passbands=passband_names,
            symmetry_test=symmetry_test,
        ),
    }
