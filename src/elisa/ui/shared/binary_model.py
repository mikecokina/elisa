"""Shared builders for UI binary-model objects."""

from __future__ import annotations

from elisa import BinarySystem, Star
from elisa import units as u
from elisa.ui.shared.utils import opt_float
from elisa.ui.shared.validators import validated_float, validated_positive_int


def build_star(
    params: dict[str, object],
    *,
    label: str,
    pulsations: list[dict[str, object]] | None = None,
    spots: list[dict[str, object]] | None = None,
) -> Star:
    """Construct a :class:`~elisa.base.star.Star` from a flat parameter dict.

    :param params: Flat parameter dictionary with keys ``mass``,
        ``t_eff``, ``surface_potential``, ``synchronicity``, and
        optional keys ``gravity_darkening``, ``albedo``,
        ``metallicity``, ``discretization_factor``, ``atmosphere``.
    :type params: dict[str, object]
    :param label: Human-readable label used in error messages
        (for example ``"primary"`` or ``"secondary"``).
    :type label: str
    :param pulsations: Optional list of pulsation mode parameter dicts.
    :type pulsations: list[dict[str, object]] | None
    :param spots: Optional list of spot parameter dicts.
    :type spots: list[dict[str, object]] | None
    :returns: Initialised :class:`~elisa.base.star.Star` instance.
    :rtype: Star
    :raises ValueError: If any mandatory parameter is ``None`` or
        missing from *params*.
    """
    mass_raw = opt_float(params.get("mass"))  # type: ignore[arg-type]
    t_eff_raw = opt_float(params.get("t_eff"))  # type: ignore[arg-type]
    surface_potential_raw = opt_float(params.get("surface_potential"))  # type: ignore[arg-type]
    synchronicity_raw = opt_float(params.get("synchronicity"))  # type: ignore[arg-type]

    for name, val in [
        ("mass", mass_raw),
        ("t_eff", t_eff_raw),
        ("surface_potential", surface_potential_raw),
        ("synchronicity", synchronicity_raw),
    ]:
        if val is None:
            msg = f"{label} star: mandatory parameter '{name}' is missing or empty."
            raise ValueError(msg)

    kwargs: dict[str, object] = {
        "mass": mass_raw * u.solMass,
        "t_eff": t_eff_raw * u.K,
        "surface_potential": surface_potential_raw,
        "synchronicity": synchronicity_raw,
    }

    gravity_darkening = validated_float(
        params.get("gravity_darkening"),  # type: ignore[arg-type]
        name=f"{label}.gravity_darkening",
        lo=0.0,
        hi=1.0,
    )
    albedo = validated_float(
        params.get("albedo"),  # type: ignore[arg-type]
        name=f"{label}.albedo",
        lo=0.0,
        hi=1.0,
    )
    metallicity = opt_float(params.get("metallicity"))  # type: ignore[arg-type]
    discretization_factor = validated_positive_int(
        params.get("discretization_factor"),  # type: ignore[arg-type]
        name=f"{label}.discretization_factor",
    )
    atmosphere: str | None = params.get("atmosphere") or None  # type: ignore[assignment]

    if gravity_darkening is not None:
        kwargs["gravity_darkening"] = gravity_darkening
    if albedo is not None:
        kwargs["albedo"] = albedo
    if metallicity is not None:
        kwargs["metallicity"] = metallicity
    if discretization_factor is not None:
        kwargs["discretization_factor"] = discretization_factor
    if atmosphere:
        kwargs["atmosphere"] = atmosphere
    if pulsations:
        kwargs["pulsations"] = pulsations
    if spots:
        kwargs["spots"] = spots

    return Star(**kwargs)


def build_system(
    primary: Star,
    secondary: Star,
    params: dict[str, object],
) -> BinarySystem:
    """Construct a :class:`~elisa.binary_system.system.BinarySystem`.

    :param primary: Primary stellar component.
    :type primary: Star
    :param secondary: Secondary stellar component.
    :type secondary: Star
    :param params: Flat parameter dict with keys ``inclination``,
        ``period``, ``eccentricity``, ``argument_of_periastron``, and
        optional keys ``gamma``, ``phase_shift``,
        ``additional_light``, ``primary_minimum_time``, ``distance``.
    :type params: dict[str, object]
    :returns: Initialised :class:`~elisa.binary_system.system.BinarySystem`.
    :rtype: BinarySystem
    :raises ValueError: If any mandatory parameter is ``None`` or missing.
    """
    inclination_raw = opt_float(params.get("inclination"))  # type: ignore[arg-type]
    period_raw = opt_float(params.get("period"))  # type: ignore[arg-type]
    eccentricity_raw = opt_float(params.get("eccentricity"))  # type: ignore[arg-type]
    aop_raw = opt_float(params.get("argument_of_periastron"))  # type: ignore[arg-type]

    for name, val in [
        ("inclination", inclination_raw),
        ("period", period_raw),
        ("eccentricity", eccentricity_raw),
        ("argument_of_periastron", aop_raw),
    ]:
        if val is None:
            msg = f"Binary system: mandatory parameter '{name}' is missing or empty."
            raise ValueError(msg)

    bs_kwargs: dict[str, object] = {
        "inclination": inclination_raw * u.deg,
        "period": period_raw * u.d,
        "eccentricity": eccentricity_raw,
        "argument_of_periastron": aop_raw * u.deg,
    }

    gamma_raw = opt_float(params.get("gamma"))  # type: ignore[arg-type]
    phase_shift_raw = opt_float(params.get("phase_shift"))  # type: ignore[arg-type]
    additional_light_raw = validated_float(
        params.get("additional_light"),  # type: ignore[arg-type]
        name="additional_light",
        lo=0.0,
        hi=1.0,
    )
    pmt_raw = opt_float(params.get("primary_minimum_time"))  # type: ignore[arg-type]
    distance_raw = validated_float(
        params.get("distance"),  # type: ignore[arg-type]
        name="distance",
        lo=0.0,
    )

    if gamma_raw is not None:
        bs_kwargs["gamma"] = gamma_raw * u.km / u.s
    if phase_shift_raw is not None:
        bs_kwargs["phase_shift"] = phase_shift_raw
    if additional_light_raw is not None:
        bs_kwargs["additional_light"] = additional_light_raw
    if pmt_raw is not None:
        bs_kwargs["primary_minimum_time"] = pmt_raw * u.d
    if distance_raw is not None:
        bs_kwargs["distance"] = distance_raw * u.pc

    return BinarySystem(primary=primary, secondary=secondary, **bs_kwargs)
