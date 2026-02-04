import textwrap
from pathlib import Path

import pytest

import sys


def import_fresh_elisa_settings():
    # Remove ALL elisa-related modules, not just elisa + elisa.settings
    for name in list(sys.modules.keys()):
        if name == "elisa" or name.startswith("elisa."):
            sys.modules.pop(name, None)

    # Now import fresh and return the Settings instance
    from elisa import settings as settings_instance
    return settings_instance


@pytest.fixture
def elisa_ini(tmp_path: Path, monkeypatch):
    home = tmp_path / "elisa_home"
    home.mkdir(parents=True, exist_ok=True)

    expected = {
        # general
        "LOG_CONFIG": "default",
        "SUPPRESS_WARNINGS": True,
        "SUPPRESS_LOGGER": False,
        "HOME": home.as_posix(),

        # physics
        "REFLECTION_EFFECT": True,
        "REFLECTION_EFFECT_ITERATIONS": 2,
        "LIMB_DARKENING_LAW": "cosine",
        "PULSATION_MODEL": "uniform",
        "DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT": 1.0471975512,
        "SURFACE_DISPLACEMENT_TOL": 0.01,
        "RV_METHOD": "kinematic",
        "RV_LAMBDA_INTERVAL": (5500, 10500),

        # computational
        "MAX_DISCRETIZATION_FACTOR": 20.0,
        "MIN_DISCRETIZATION_FACTOR": 1.0,
        "DEFAULT_DISCRETIZATION_FACTOR": 5,
        "NUMBER_OF_THREADS": 1,
        "NUMBER_OF_PROCESSES": -1,
        "NUMBER_OF_MCMC_PROCESSES": -1,
        "MAX_NU_SEPARATION": 0.045,
        "MAX_D_FLUX": 0.0,
        "MAX_SPOT_D_LONGITUDE": 0.01745329252,
        "MAX_SOLVER_ITERS": 100,
        "MCMC_SAVE_INTERVAL": 1800,
        "MAX_CURVE_DATAPOINTS": 300,
        "MIN_POINTS_IN_ECLIPSE": 10,
        "MESH_GENERATOR": "auto",
        "DEFORMATION_TOL": 0.05,
        "USE_SINGLE_LD_COEFFICIENTS": False,
        "USE_INTERPOLATION_APPROXIMATION": True,
        "USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION": True,
        "USE_SIMILAR_NEIGHBOURS_APPROXIMATION": True,

        # support
        "PASSBAND_TABLES": (tmp_path / "passband_tables").as_posix(),
        "LD_TABLES": (tmp_path / "ld_tables").as_posix(),
        "CASTELLI_KURUCZ_04_ATM_TABLES": (tmp_path / "ck04_atm").as_posix(),
        "KURUCZ_93_ATM_TABLES": (tmp_path / "k93_atm").as_posix(),
        "ATLAS": "ck04",
        "MAGNITUDE_SYSTEM": "vega",
    }

    ini_text = textwrap.dedent(f"""
        [general]
        log_config = {expected["LOG_CONFIG"]}
        suppress_warnings = {expected["SUPPRESS_WARNINGS"]}
        suppress_logger = {expected["SUPPRESS_LOGGER"]}
        home = {expected["HOME"]}

        [physics]
        reflection_effect = {expected["REFLECTION_EFFECT"]}
        reflection_effect_iterations = {expected["REFLECTION_EFFECT_ITERATIONS"]}
        limb_darkening_law = {expected["LIMB_DARKENING_LAW"]}
        pulsation_model = {expected["PULSATION_MODEL"]}
        default_temperature_perturbation_phase_shift = {expected["DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT"]}
        surface_displacement_tol = {expected["SURFACE_DISPLACEMENT_TOL"]}
        rv_method = {expected["RV_METHOD"]}
        rv_lambda_interval = ({expected["RV_LAMBDA_INTERVAL"][0]}, {expected["RV_LAMBDA_INTERVAL"][1]})

        [computational]
        max_discretization_factor = {expected["MAX_DISCRETIZATION_FACTOR"]}
        min_discretization_factor = {expected["MIN_DISCRETIZATION_FACTOR"]}
        default_discretization_factor = {expected["DEFAULT_DISCRETIZATION_FACTOR"]}
        number_of_threads = {expected["NUMBER_OF_THREADS"]}
        number_of_processes = {expected["NUMBER_OF_PROCESSES"]}
        number_of_mcmc_processes = {expected["NUMBER_OF_MCMC_PROCESSES"]}
        max_nu_separation = {expected["MAX_NU_SEPARATION"]}
        max_d_flux = {expected["MAX_D_FLUX"]}
        max_spot_d_longitude = {expected["MAX_SPOT_D_LONGITUDE"]}
        max_solver_iters = {expected["MAX_SOLVER_ITERS"]}
        mcmc_save_interval = {expected["MCMC_SAVE_INTERVAL"]}
        max_curve_datapoints = {expected["MAX_CURVE_DATAPOINTS"]}
        min_points_in_eclipse = {expected["MIN_POINTS_IN_ECLIPSE"]}
        mesh_generator = {expected["MESH_GENERATOR"]}
        deformation_tol = {expected["DEFORMATION_TOL"]}
        use_single_ld_coefficients = {expected["USE_SINGLE_LD_COEFFICIENTS"]}
        use_interpolation_approximation = {expected["USE_INTERPOLATION_APPROXIMATION"]}
        use_symmetrical_counterparts_approximation = {expected["USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION"]}
        use_similar_neighbours_approximation = {expected["USE_SIMILAR_NEIGHBOURS_APPROXIMATION"]}

        [support]
        passband_tables = {expected["PASSBAND_TABLES"]}
        ld_tables = {expected["LD_TABLES"]}
        castelli_kurucz_04_atm_tables = {expected["CASTELLI_KURUCZ_04_ATM_TABLES"]}
        kurucz_93_atm_tables = {expected["KURUCZ_93_ATM_TABLES"]}
        atlas = {expected["ATLAS"]}
        magnitude_system = {expected["MAGNITUDE_SYSTEM"]}
    """).strip() + "\n"

    ini_path = tmp_path / "elisa_conf.ini"
    ini_path.write_text(ini_text, encoding="utf-8")

    monkeypatch.setenv("ELISA_CONFIG", str(ini_path))

    # Make sure next import uses our env var and rebuilds the singleton
    sys.modules.pop("elisa", None)
    sys.modules.pop("elisa.settings", None)

    return ini_path, expected


def test_elisa_config_ini_types_do_not_raise(elisa_ini):
    ini_path, expected = elisa_ini

    try:
        settings_instance = import_fresh_elisa_settings()
    except Exception as e:
        raise AssertionError(
            "Importing `from elisa import settings` raised with an INI typed per docs.\n\n"
            f"INI used: {ini_path}\n"
            f"Exception: {type(e).__name__}: {e}"
        ) from e

    # sanity: got an instance with some expected attr
    assert hasattr(settings_instance, "HOME")


IGNORE_ATTRS = {
    "LOG_CONFIG",
    "CASTELLI_KURUCZ_04_ATM_TABLES",
    "KURUCZ_93_ATM_TABLES",
    "ATLAS",
}


def test_elisa_settings_values_match_ini(elisa_ini):
    ini_path, expected = elisa_ini
    s = import_fresh_elisa_settings()

    mismatches = []
    for attr, exp in expected.items():
        if attr in IGNORE_ATTRS:
            continue

        if not hasattr(s, attr):
            mismatches.append(f"Missing settings attribute: {attr}")
            continue

        got = getattr(s, attr)

        # floats: tolerant compare, but also ensure type is float
        if isinstance(exp, float):
            if not isinstance(got, float):
                mismatches.append(f"{attr}: expected float, got {type(got).__name__} ({got!r})")
                continue
            if got != pytest.approx(exp):
                mismatches.append(f"{attr}: expected {exp!r}, got {got!r}")
            continue

        # tuples: strict type per element (int vs float), floats approx
        if isinstance(exp, tuple):
            if not isinstance(got, tuple) or len(got) != len(exp):
                mismatches.append(f"{attr}: expected tuple {exp!r}, got {got!r}")
                continue

            for i, (e_i, g_i) in enumerate(zip(exp, got)):
                if isinstance(e_i, float):
                    if not isinstance(g_i, float) or g_i != pytest.approx(e_i):
                        mismatches.append(
                            f"{attr}[{i}]: expected float {e_i!r}, got {type(g_i).__name__} ({g_i!r})"
                        )
                else:
                    if g_i != e_i or type(g_i) is not type(e_i):
                        mismatches.append(
                            f"{attr}[{i}]: expected {e_i!r} ({type(e_i).__name__}), "
                            f"got {g_i!r} ({type(g_i).__name__})"
                        )
            continue

        # everything else: strict value and strict type
        if got != exp or type(got) is not type(exp):
            mismatches.append(
                f"{attr}: expected {exp!r} ({type(exp).__name__}), got {got!r} ({type(got).__name__})"
            )

    assert not mismatches, "Loaded settings do not match INI:\n" + "\n".join(mismatches)
