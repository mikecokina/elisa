"""Check installed scientific test resources without parsing or synthesizing data.

Availability checks inspect configured directories and readable, nonempty files.
The tests retain responsibility for parsing, interpolation and scientific validity.
ELISA uses the Van Hamme LD filename convention; it has no provenance setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elisa import ld, settings
from elisa.atm import const as atm_const

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _configured_directory(value: str | Path | None, label: str) -> tuple[Path | None, str | None]:
    """Return the configured directory or a reason why it cannot be used."""
    if value is None or not str(value).strip():
        return None, f"{label} is not configured"

    directory = Path(value)
    try:
        if not directory.is_dir():
            return None, f"{label}={str(directory)!r} is not a directory"
    except OSError as error:
        return None, f"{label}={str(directory)!r} cannot be accessed: {error}"

    return directory, None


def _file_unavailability_reason(directory: Path, filenames: Iterable[str], label: str) -> str | None:
    """Check each distinct file without interpreting its contents."""
    for filename in dict.fromkeys(filenames):
        path = directory / filename
        try:
            if not path.is_file() or path.stat().st_size == 0:
                return f"{label}: missing or empty required file {path}"
            with path.open("rb") as stream:
                stream.read(1)
        except OSError as error:
            return f"{label}: cannot read {path}: {error}"

    return None


def ld_unavailability_reason(
    table_directory: str | Path | None,
    *,
    passbands: Sequence[str] = ("bolometric",),
    laws: Sequence[str] = ("linear",),
    metallicities: Sequence[float] = (0.0,),
    source: str = "van_hamme",
) -> str | None:
    """Check every LD table required by the selected bands, laws and metallicities."""
    if source != "van_hamme":
        message = f"Unsupported LD source: {source!r}"
        raise ValueError(message)

    label = "Van Hamme LD tables (LD_TABLES)"
    directory, reason = _configured_directory(table_directory, label)
    if directory is None:
        return reason

    required_filenames = []
    for law in laws:
        for passband in passbands:
            for metallicity in metallicities:
                required_filenames.extend(ld.get_relevant_ld_tables(passband, metallicity, law))

    return _file_unavailability_reason(directory, required_filenames, label)


def atmosphere_unavailability_reason(
    table_directory: str | Path | None,
    *,
    atlas: str = "ck04",
    files: Sequence[str] | None = None,
) -> str | None:
    """Check an atlas installation or the exact model files required by a case."""
    prefix = atm_const.ATM_ATLAS_NORMALIZER.get(atlas)
    if prefix is None:
        return f"Atmosphere atlas {atlas!r} is not supported"
    if prefix == "bb":
        return None

    setting_name = "CK04_ATM_TABLES" if prefix == "ck" else "K93_ATM_TABLES"
    label = f"{atlas} atmosphere models ({setting_name})"
    directory, reason = _configured_directory(table_directory, label)
    if directory is None:
        return reason

    if files is None:
        # Preflight establishes that an atlas is installed, not that it covers
        # every model a test may request. Later reader failures must propagate.
        model_pattern = f"{prefix}[mp]*/{prefix}[mp]*_g*.csv"
        sample_model = next(directory.glob(model_pattern), None)
        if sample_model is None:
            return f"{label}: no model CSV files in {directory}"
        files = (str(sample_model.relative_to(directory)),)

    return _file_unavailability_reason(directory, files, label)


def require_scientific_data(item: pytest.Item) -> None:
    """Skip unavailable requirements before fixtures and unittest setup execute."""
    parameterization = getattr(item, "callspec", None)
    parameters = parameterization.params if parameterization is not None else {}
    unavailable_reasons = []

    for marker in item.iter_markers("requires_ld"):
        options = dict(marker.kwargs)
        if "law" in parameters:
            options.setdefault("laws", (parameters["law"],))
        if "metal" in parameters:
            options.setdefault("metallicities", (parameters["metal"],))
        if "band" in parameters:
            options.setdefault("passbands", (parameters["band"],))
        if "bands" in parameters:
            options["passbands"] = (*options.get("passbands", ()), *parameters["bands"])
        options.setdefault("laws", (settings.LIMB_DARKENING_LAW,))

        reason = ld_unavailability_reason(settings.LD_TABLES, **options)
        if reason is not None:
            unavailable_reasons.append(reason)

    for marker in item.iter_markers("requires_atmosphere"):
        options = dict(marker.kwargs)
        atlas = options.pop("atlas", None) or settings.ATM_ATLAS
        directory = settings.ATLAS_TO_BASE_DIR.get(atlas)
        reason = atmosphere_unavailability_reason(directory, atlas=atlas, **options)
        if reason is not None:
            unavailable_reasons.append(reason)

    if unavailable_reasons:
        pytest.skip("; ".join(dict.fromkeys(unavailable_reasons)))
