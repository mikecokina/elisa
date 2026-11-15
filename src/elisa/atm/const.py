"""Atlas constants and input normalization, independent of runtime settings.

Kept separate from spectrum calculations so Star and settings initialization can
use the same input contract without importing the atmosphere synthesis pipeline.
"""

from enum import StrEnum

from elisa.base.error import raise_elisa_initial_parameters_error


class Atmosphere(StrEnum):
    """Canonical atmosphere identifiers used throughout ELISa."""

    CK04 = "ck04"
    K93 = "k93"
    BB = "bb"


DEFAULT_ATM = Atmosphere.CK04


# Atlas file name prefixes are an implementation detail of atmosphere file I/O.
ATLAS_TO_ATM_FILE_PREFIX = {
    "castelli": "ck",
    "castelli-kurucz": "ck",
    "ck": "ck",
    "ck04": "ck",
    "kurucz": "k",
    "k": "k",
    "k93": "k",
}

_PREFIX_TO_ATMOSPHERE = {"ck": Atmosphere.CK04, "k": Atmosphere.K93}

ATM_ATLAS_NORMALIZER = {
    **{alias: _PREFIX_TO_ATMOSPHERE[prefix] for alias, prefix in ATLAS_TO_ATM_FILE_PREFIX.items()},
    "bb": Atmosphere.BB,
    "black_body": Atmosphere.BB,
    "black-body": Atmosphere.BB,
}

ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX = {
    "temperature": "TEMPERATURE_LIST_ATM",
    "gravity": "GRAVITY_LIST_ATM",
    "metallicity": "METALLICITY_LIST_ATM",
}


def normalize_atlas(atlas: object) -> Atmosphere:
    """Validate a fixed atlas choice and return its canonical internal name."""
    try:
        # noinspection PyTypeChecker
        return ATM_ATLAS_NORMALIZER[atlas]
    except (KeyError, TypeError) as exc:
        allowed = ", ".join(ATM_ATLAS_NORMALIZER)
        msg = f"Incorrect atlas: {atlas}. Following are allowed: {allowed}"
        return raise_elisa_initial_parameters_error(msg, exc)


def atm_file_prefix(atlas: object) -> str:
    """Return the atmosphere file name prefix of a tabulated atlas.

    This is the only place where canonical identifiers are translated to file name prefixes.

    :param atlas: Canonical atmosphere identifier or a supported alias.
    :returns: ``"ck"`` for CK04 or ``"k"`` for K93.
    :raises ElisaInitialParametersError: If the atlas is invalid or has no atlas files (blackbody).
    """
    canonical = normalize_atlas(atlas)
    if canonical == Atmosphere.BB:
        msg = "Blackbody atmosphere has no atlas files."
        raise_elisa_initial_parameters_error(msg)
    return ATLAS_TO_ATM_FILE_PREFIX[canonical]
