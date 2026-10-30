"""Atlas constants and input normalization, independent of runtime settings.

Kept separate from spectrum calculations so Star and settings initialization can
use the same input contract without importing the atmosphere synthesis pipeline.
"""

from elisa.base.error import raise_elisa_initial_parameters_error

ATLAS_TO_ATM_FILE_PREFIX = {
    "castelli": "ck",
    "castelli-kurucz": "ck",
    "ck": "ck",
    "ck04": "ck",
    "kurucz": "k",
    "k": "k",
    "k93": "k",
}

ATM_ATLAS_NORMALIZER = {
    **ATLAS_TO_ATM_FILE_PREFIX,
    "bb": "bb",
    "black_body": "bb",
    "black-body": "bb",
}

ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX = {
    "temperature": "TEMPERATURE_LIST_ATM",
    "gravity": "GRAVITY_LIST_ATM",
    "metallicity": "METALLICITY_LIST_ATM",
}


def normalize_atlas(atlas: object) -> str:
    """Validate a fixed atlas choice and return its canonical internal name."""
    try:
        # noinspection PyTypeChecker
        return ATM_ATLAS_NORMALIZER[atlas]
    except (KeyError, TypeError) as exc:
        allowed = ", ".join(ATM_ATLAS_NORMALIZER)
        msg = f"Incorrect atlas: {atlas}. Following are allowed: {allowed}"
        return raise_elisa_initial_parameters_error(msg, exc)
