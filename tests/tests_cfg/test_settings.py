import logging
import os
import subprocess
import sys
import textwrap
import unittest
import warnings
from pathlib import Path
from typing import TypeAlias
from unittest.mock import patch

import pytest

from elisa import settings
from elisa.base.curves.utils import interpolate_on_ld_grid_cached
from elisa.conf.enums import IntersectionBackend, MeanAnomalySolver
from tests.utils import ElisaTestCase

IniFixture: TypeAlias = tuple[Path, dict[str, object]]


@pytest.fixture
def elisa_ini(tmp_path: Path):
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
        "INTERSECTION_BACKEND": "sutherland_hodgman",
        "NUMBER_OF_MCMC_PROCESSES": -1,
        "MAX_NU_SEPARATION": 0.045,
        "MAX_D_FLUX": 0.0,
        "MAX_SPOT_D_LONGITUDE": 0.01745329252,
        "MAX_SOLVER_ITERS": 100,
        "MCMC_SAVE_INTERVAL": 1800,
        "MCMC_AUTOCORR_TOL": 50,
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
        intersection_backend = {expected["INTERSECTION_BACKEND"]}
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
        mcmc_autocorr_tol = {expected["MCMC_AUTOCORR_TOL"]}
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
        magnitude_system = {expected["MAGNITUDE_SYSTEM"]}
    """).strip() + "\n"

    ini_path = tmp_path / "elisa_conf.ini"
    ini_path.write_text(ini_text, encoding="utf-8")

    with settings.override():
        yield ini_path, expected


def test_elisa_config_ini_types_do_not_raise(elisa_ini):
    ini_path, _ = elisa_ini
    result = subprocess.run(
        [sys.executable, "-c", "from elisa import settings\nassert hasattr(settings, 'HOME')\n"],
        env={**os.environ, "ELISA_CONFIG": str(ini_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        "Importing `from elisa import settings` raised with an INI typed per docs.\n\n"
        f"INI used: {ini_path}\n{result.stdout}\n{result.stderr}"
    )


IGNORE_ATTRS = {
    "LOG_CONFIG",
    "CASTELLI_KURUCZ_04_ATM_TABLES",
    "KURUCZ_93_ATM_TABLES",
}


def test_elisa_settings_values_match_ini(elisa_ini):
    ini_path, expected = elisa_ini
    settings.read_and_update_config(conf_path=str(ini_path))
    s = settings

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
                            f"{attr}[{i}]: expected float {e_i!r}, got {type(g_i).__name__} ({g_i!r})",
                        )
                elif g_i != e_i or type(g_i) is not type(e_i):
                    mismatches.append(
                        f"{attr}[{i}]: expected {e_i!r} ({type(e_i).__name__}), "
                        f"got {g_i!r} ({type(g_i).__name__})",
                    )
            continue

        # everything else: strict value and strict type
        if got != exp or type(got) is not type(exp):
            mismatches.append(
                f"{attr}: expected {exp!r} ({type(exp).__name__}), got {got!r} ({type(got).__name__})",
            )

    assert not mismatches, "Loaded settings do not match INI:\n" + "\n".join(mismatches)


def test_invalid_limb_darkening_raises(elisa_ini):
    """Invalid limb-darkening law in INI should raise ValueError when loaded.

    :param elisa_ini: base INI fixture
    :type elisa_ini: tuple[pathlib.Path, dict]
    :returns: None
    :rtype: None
    """
    ini_path, _ = elisa_ini
    text = ini_path.read_text(encoding="utf-8")
    text = text.replace("limb_darkening_law = cosine", "limb_darkening_law = not_a_law")
    ini_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="not valid name of limb darkening law"):
        settings.read_and_update_config(conf_path=str(ini_path))


def test_malformed_rv_lambda_interval_raises(elisa_ini):
    """Malformed rv_lambda_interval should raise ValueError from parser.

    :param elisa_ini: base INI fixture
    :type elisa_ini: tuple[pathlib.Path, dict]
    :returns: None
    :rtype: None
    """
    ini_path, _ = elisa_ini
    text = ini_path.read_text(encoding="utf-8")
    text = text.replace("rv_lambda_interval = (5500, 10500)", "rv_lambda_interval = 5500-10500")
    ini_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid format"):
        settings.read_and_update_config(conf_path=str(ini_path))


def test_number_of_processes_clamped_and_warns(elisa_ini, monkeypatch):
    """If number_of_processes > cpu_count, ensure it's clamped and a warning emitted.

    We set suppress_warnings=False in the INI to allow the warning to be emitted.
    """
    ini_path, _ = elisa_ini
    text = ini_path.read_text(encoding="utf-8")
    text = text.replace("number_of_processes = -1", "number_of_processes = 9999")
    text = text.replace("suppress_warnings = True", "suppress_warnings = False")
    ini_path.write_text(text, encoding="utf-8")

    # Fake a small cpu count
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    with pytest.warns(UserWarning):
        settings.read_and_update_config(conf_path=str(ini_path))
        s = settings

    assert isinstance(s.NUMBER_OF_PROCESSES, int)
    assert s.NUMBER_OF_PROCESSES == 4


def test_missing_tables_warn(elisa_ini):
    """Missing tables paths should emit UserWarning when warnings are not suppressed.

    The fixture points to temp directories that do not exist; enable warnings in the INI
    and ensure a UserWarning is raised when loading settings.
    """
    ini_path, _ = elisa_ini
    text = ini_path.read_text(encoding="utf-8")
    text = text.replace("suppress_warnings = True", "suppress_warnings = False")
    ini_path.write_text(text, encoding="utf-8")

    with pytest.warns(UserWarning):
        settings.read_and_update_config(conf_path=str(ini_path))


@pytest.mark.parametrize("backend", ["pypex", "reference", "sutherland_hodgman"])
def test_intersection_backend_configure_and_serialize(elisa_ini: IniFixture, backend: str) -> None:
    ini_path, _ = elisa_ini
    configuration = ini_path.read_text().replace(
        "intersection_backend = sutherland_hodgman",
        f"intersection_backend = {backend}",
    )
    ini_path.write_text(configuration)
    settings.read_and_update_config(conf_path=str(ini_path))
    loaded_settings = settings
    assert loaded_settings.INTERSECTION_BACKEND == backend

    loaded_settings.configure(INTERSECTION_BACKEND=backend)

    assert loaded_settings.INTERSECTION_BACKEND == backend
    assert loaded_settings.settings_serializer()["INTERSECTION_BACKEND"] == backend


def test_intersection_backend_rejects_invalid_configure(elisa_ini):
    s = settings
    before = s.INTERSECTION_BACKEND
    with pytest.raises(ValueError, match="INTERSECTION_BACKEND"):
        s.configure(INTERSECTION_BACKEND="invalid")
    assert s.INTERSECTION_BACKEND == before


def test_intersection_backend_rejects_invalid_ini(elisa_ini):
    ini_path, _ = elisa_ini
    ini_path.write_text(ini_path.read_text().replace("intersection_backend = sutherland_hodgman", "intersection_backend = invalid"))
    with pytest.raises(ValueError, match="INTERSECTION_BACKEND"):
        settings.read_and_update_config(conf_path=str(ini_path))


def test_intersection_backend_defaults_to_sutherland_hodgman(elisa_ini):
    ini_path, _ = elisa_ini
    ini_path.write_text(ini_path.read_text().replace("intersection_backend = sutherland_hodgman", ""))
    settings.configure(INTERSECTION_BACKEND=settings.DEFAULT_SETTINGS["INTERSECTION_BACKEND"])
    settings.read_and_update_config(conf_path=str(ini_path))
    s = settings
    assert s.INTERSECTION_BACKEND == "sutherland_hodgman"
    assert s.DEFAULT_SETTINGS["INTERSECTION_BACKEND"] == "sutherland_hodgman"


@pytest.mark.parametrize("tol", [0, 50])
def test_mcmc_autocorr_tol_configure_and_serialize(elisa_ini, tol):
    s = settings
    s.configure(MCMC_AUTOCORR_TOL=tol)
    assert s.MCMC_AUTOCORR_TOL == tol
    assert s.settings_serializer()["MCMC_AUTOCORR_TOL"] == tol


def test_mcmc_autocorr_tol_defaults_to_zero(elisa_ini):
    ini_path, _ = elisa_ini
    ini_path.write_text(ini_path.read_text().replace("mcmc_autocorr_tol = 50", ""))
    settings.configure(MCMC_AUTOCORR_TOL=settings.DEFAULT_SETTINGS["MCMC_AUTOCORR_TOL"])
    settings.read_and_update_config(conf_path=str(ini_path))
    s = settings
    assert s.MCMC_AUTOCORR_TOL == 0
    assert s.DEFAULT_SETTINGS["MCMC_AUTOCORR_TOL"] == 0


def test_singleton_effective_values_and_independent_serialization(elisa_ini):
    s = settings
    assert type(s)() is s
    snapshot = s.settings_serializer()
    assert all(getattr(s, name) == value for name, value in snapshot.items())
    snapshot['REFLECTION_EFFECT'] = not s.REFLECTION_EFFECT
    assert snapshot['REFLECTION_EFFECT'] != s.REFLECTION_EFFECT
    # Defaults are startup inputs, never a second live owner.
    from elisa.managers.settings_manager import DefaultSettings
    s.configure(REFLECTION_EFFECT=False)
    assert DefaultSettings.REFLECTION_EFFECT is True
    assert not hasattr(type(s), 'REFLECTION_EFFECT')


def test_direct_assignment_updates_the_single_store(elisa_ini):
    s = settings
    s.REFLECTION_EFFECT_ITERATIONS = 7
    assert s.REFLECTION_EFFECT_ITERATIONS == s.settings_serializer()['REFLECTION_EFFECT_ITERATIONS'] == 7
    before = s.__dict__
    with pytest.raises(ValueError):
        s.REFLECTION_EFFECT_ITERATIONS = 'seven'
    assert s.__dict__ is before
    with pytest.raises(AttributeError):
        del s.REFLECTION_EFFECT_ITERATIONS


@pytest.mark.parametrize('name', [
    'BASE_DIR', 'FIRST_TIME_USER', 'ATLAS_TO_BASE_DIR', 'SCHEMA_REGISTRY',
    'DEFAULT_SETTINGS', 'PATH_TO_SINGLE_CORRECTIONS', 'PASSBANDS', 'configure', '_instance', 'UNKNOWN',
])
def test_constants_services_and_unknown_fields_are_not_writable(elisa_ini, name):
    s = settings
    before = s.__dict__
    with pytest.raises(AttributeError):
        setattr(s, name, None)
    with pytest.raises(ValueError):
        s.configure(**{'REFLECTION_EFFECT': False, name: None})
    assert s.__dict__ is before


@pytest.mark.parametrize('bad', [
    {'UNKNOWN': True}, {'NUMBER_OF_THREADS': 0}, {'REFLECTION_EFFECT': 'false'},
    {'NUMBER_OF_PROCESSES': 1.5}, {'MAX_D_FLUX': 'small'}, {'RV_LAMBDA_INTERVAL': (2, 1)},
    {'RV_LAMBDA_INTERVAL': (1,)}, {'RV_LAMBDA_INTERVAL': ('1', '2')},
    {'LIMB_DARKENING_LAW': 'unsupported'}, {'INTERSECTION_BACKEND': 'unsupported'},
    {'HOME': None}, {'LOG_CONFIG': 'missing-logging-policy.json'},
    {'MCMC_AUTOCORR_TOL': 1.5}, {'MCMC_AUTOCORR_TOL': True},
])
@pytest.mark.parametrize('invalid_first', [True, False])
def test_invalid_first_or_second_argument_never_publishes(elisa_ini, bad, invalid_first):
    s = settings
    good = {'REFLECTION_EFFECT_ITERATIONS': 19}
    changes = {**bad, **good} if invalid_first else {**good, **bad}
    before = s.__dict__
    snapshot = s.settings_serializer()
    with pytest.raises(ValueError):
        s.configure(**changes)
    assert s.__dict__ is before
    assert s.settings_serializer() == snapshot


def test_multifield_transaction_publishes_values_and_derived_map_together(elisa_ini, monkeypatch):
    s = settings
    before = s.__dict__
    import importlib
    module = importlib.import_module('elisa.conf.settings')
    observed = []

    def apply_logging(profile):
        # Even the final external preparation step sees the complete old state.
        observed.append(profile)
        assert s.__dict__ is before

    with monkeypatch.context() as logging_patch:
        logging_patch.setattr(module, 'configure_logging', apply_logging)
        s.configure(REFLECTION_EFFECT_ITERATIONS=11, CK04_ATM_TABLES='new-ck',
                    K93_ATM_TABLES='new-k', LOG_CONFIG='fit')
    assert observed == ['fit']
    assert s.REFLECTION_EFFECT_ITERATIONS == 11
    assert s.ATLAS_TO_BASE_DIR['ck04'] == s.CK04_ATM_TABLES == 'new-ck'
    assert s.ATLAS_TO_BASE_DIR['k93'] == s.K93_ATM_TABLES == 'new-k'
    assert before['ATLAS_TO_BASE_DIR']['ck04'] != 'new-ck'
    with pytest.raises(TypeError):
        s.ATLAS_TO_BASE_DIR['ck04'] = 'shadow'
    assert all(getattr(s, key) == value for key, value in s.settings_serializer().items())


@pytest.mark.parametrize('law', ['linear', 'cosine', 'logarithmic', 'square_root'])
def test_scientific_modes_and_sentinels_remain_mutable(elisa_ini, law):
    s = settings
    for mesh in ('auto', 'trapezoidal', 'improved_trapezoidal'):
        for rv in ('kinematic', 'radiometric'):
            s.configure(LIMB_DARKENING_LAW=law, MESH_GENERATOR=mesh, RV_METHOD=rv,
                        NUMBER_OF_PROCESSES=-1, NUMBER_OF_MCMC_PROCESSES=0,
                        MAX_D_FLUX=0.0, MAX_NU_SEPARATION=-1.0,
                        REFLECTION_EFFECT_ITERATIONS=0, RV_LAMBDA_INTERVAL=(5500, 5500))
            assert s.MESH_GENERATOR == mesh
            assert s.RV_METHOD == rv
            assert s.NUMBER_OF_PROCESSES == -1
            assert s.NUMBER_OF_MCMC_PROCESSES == 0
            assert s.LIMB_DARKENING_LAW == law
    s.configure(RV_LAMBDA_INTERVAL=(1, float('inf')), MAX_D_FLUX=float('inf'),
                SUPPRESS_LOGGER=None, CUDA=True)
    assert s.RV_LAMBDA_INTERVAL[1] == float('inf')
    assert s.CUDA is True  # Backend availability remains outside runtime core validation.


def test_ini_failure_does_not_change_effective_state_or_logging(elisa_ini):
    ini, _ = elisa_ini
    settings.read_and_update_config(conf_path=str(ini))
    s = settings
    before = s.__dict__
    ini.write_text('[general]\nlog_config=fit\n[physics]\nreflection_effect_iterations=29\n'
                   '[computational]\nnumber_of_threads=0\n')
    with pytest.raises(ValueError, match='number_of_threads'):
        s.read_and_update_config(conf_path=str(ini))
    assert s.__dict__ is before
    assert s.LOG_CONFIG == 'default'


def test_each_ini_read_uses_only_its_own_options(elisa_ini, tmp_path):
    ini, _ = elisa_ini
    settings.read_and_update_config(conf_path=str(ini))
    s = settings
    ini.write_text('[physics]\nreflection_effect_iterations=17\n')
    s.read_and_update_config(conf_path=str(ini))
    s.configure(REFLECTION_EFFECT_ITERATIONS=3)
    empty = tmp_path / 'empty.ini'
    empty.write_text('')
    s.read_and_update_config(conf_path=str(empty))
    assert s.REFLECTION_EFFECT_ITERATIONS == 3


def test_home_preparation_failure_retains_all_values(elisa_ini, tmp_path):
    s = settings
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    before = s.__dict__
    with pytest.raises(FileExistsError):
        s.configure(REFLECTION_EFFECT_ITERATIONS=5, HOME=str(blocked))
    assert s.__dict__ is before


def test_logging_application_failure_retains_all_settings(elisa_ini, tmp_path):
    s = settings
    bad = tmp_path / 'invalid.json'
    bad.write_text('{"version": 999}')
    before = s.__dict__
    with pytest.raises(ValueError):
        s.configure(REFLECTION_EFFECT_ITERATIONS=5, LOG_CONFIG=str(bad))
    assert s.__dict__ is before
    # dictConfig owns external side effects; restore logging through its public API.
    s.configure_logging('default')


def test_config_file_selector_and_pathlike_values(elisa_ini, tmp_path):
    s = settings
    selected = tmp_path / 'selected.ini'
    selected.write_text('[physics]\nreflection_effect_iterations=23\n')
    s.CONFIG_FILE = selected
    s.read_and_update_config()
    assert s.settings_serializer()['CONFIG_FILE'] == str(selected)
    assert s.REFLECTION_EFFECT_ITERATIONS == 23
    s.CK04_ATM_TABLES = tmp_path / 'optional-not-installed'
    assert s.ATLAS_TO_BASE_DIR['ck04'] == str(tmp_path / 'optional-not-installed')


def test_fresh_scientific_objects_follow_mutable_defaults(elisa_ini):
    s = settings
    import numpy as np

    from elisa.base.curves.utils import _get_rv_passband
    from elisa.base.star import Star

    s.configure(DEFAULT_DISCRETIZATION_FACTOR=5, RV_LAMBDA_INTERVAL=(5000, 5500))
    first = Star(mass=1., t_eff=6000.)
    bounds_a = _get_rv_passband()[1:]
    s.configure(DEFAULT_DISCRETIZATION_FACTOR=9, RV_LAMBDA_INTERVAL=(6000, 6500))
    second = Star(mass=1., t_eff=6000.)
    assert first.discretization_factor == pytest.approx(np.radians(5))
    assert second.discretization_factor == pytest.approx(np.radians(9))
    assert _get_rv_passband()[1:] != bounds_a
    s.configure(RV_LAMBDA_INTERVAL=(5000, 5500))
    assert _get_rv_passband()[1:] == bounds_a


def test_ini_preserves_numerical_sentinels_and_overflow_interval(elisa_ini):
    ini, _ = elisa_ini
    settings.read_and_update_config(conf_path=str(ini))
    s = settings
    ini.write_text('[physics]\nrv_lambda_interval=(1e309, 1e309)\nreflection_effect_iterations=0\n'
                   '[computational]\nnumber_of_processes=-1\nnumber_of_mcmc_processes=0\n'
                   'max_d_flux=0.0\nmax_nu_separation=-1.0\nmin_discretization_factor=1.5\n')
    s.read_and_update_config(conf_path=str(ini))
    assert s.RV_LAMBDA_INTERVAL == (float('inf'), float('inf'))
    assert s.NUMBER_OF_PROCESSES == -1
    assert s.NUMBER_OF_MCMC_PROCESSES == 0
    assert s.MAX_D_FLUX == 0.0
    assert s.MAX_NU_SEPARATION == -1.0
    assert s.MIN_DISCRETIZATION_FACTOR == 1.5
    # Missing options keep their effective typed values, without stringifying
    # an interval that the numeric INI parser would not accept literally.
    ini.write_text('[physics]\nreflection_effect_iterations=3\n')
    s.read_and_update_config(conf_path=str(ini))
    assert s.RV_LAMBDA_INTERVAL == (float('inf'), float('inf'))


def test_direct_log_assignment_uses_existing_logging_apis(elisa_ini):
    s = settings
    try:
        s.LOG_CONFIG = 'fit'
        assert s.LOG_CONFIG == s.settings_serializer()['LOG_CONFIG'] == 'fit'
        with s.logging_context('default'):
            assert s.LOG_CONFIG == 'default'
        assert s.LOG_CONFIG == 'fit'
        s.configure(LOG_CONFIG=s.DEFAULT_SETTINGS['LOG_CONFIG'])
        assert s.LOG_CONFIG == 'default'
    finally:
        s.configure_logging('default')


class SettingsOverrideTestCase(ElisaTestCase):
    def test_nested_contexts_restore_complete_effective_configuration(self):
        original = settings.settings_serializer()
        identity = id(settings)
        with settings.override(LIMB_DARKENING_LAW="logarithmic"):
            settings.configure(CK04_ATM_TABLES="outer-ck", MAX_CURVE_DATAPOINTS=17)
            outer = settings.settings_serializer()
            with self.assertRaisesRegex(RuntimeError, "inner"):
                with settings.override(LIMB_DARKENING_LAW="square_root", CK04_ATM_TABLES="inner-ck"):
                    self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], "inner-ck")
                    settings.RV_LAMBDA_INTERVAL = (6000, 6500)
                    raise RuntimeError("inner")
            self.assertEqual(settings.settings_serializer(), outer)
            self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], "outer-ck")
        self.assertEqual(id(settings), identity)
        self.assertEqual(settings.settings_serializer(), original)
        self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], original["CK04_ATM_TABLES"])

    def test_exception_restores_additional_settings_and_logging(self):
        original = settings.settings_serializer()
        with self.assertRaisesRegex(RuntimeError, "body"):
            with settings.override(LOG_CONFIG="fit"):
                settings.configure(NUMBER_OF_THREADS=2, K93_ATM_TABLES="temporary-k")
                raise RuntimeError("body")
        self.assertEqual(settings.settings_serializer(), original)
        self.assertEqual(settings.ATLAS_TO_BASE_DIR["k93"], original["K93_ATM_TABLES"])

    def test_invalid_override_does_not_enter_or_publish(self):
        original = settings.__dict__
        with self.assertRaises(ValueError):
            with settings.override(REFLECTION_EFFECT=False, NUMBER_OF_THREADS=0):
                self.fail("invalid context entered")
        self.assertIs(settings.__dict__, original)


class SettingsIsolationTestCase(ElisaTestCase):
    def test_repeated_reordered_tests_and_failures_restore_canonical_defaults(self):
        expected = settings.settings_serializer()
        observed = []

        class Probe(ElisaTestCase):
            def setUp(self):
                super().setUp()
                observed.append(settings.settings_serializer())
                if self._testMethodName == "setup_error":
                    settings.configure(MAX_CURVE_DATAPOINTS=11, CK04_ATM_TABLES="setup-ck")
                    raise RuntimeError("setup error")

            def mutate(self):
                # Includes settings absent from the INI and the derived atlas map.
                settings.configure(LIMB_DARKENING_LAW="logarithmic", MAX_CURVE_DATAPOINTS=13,
                                   MAGNITUDE_SYSTEM="ab", CK04_ATM_TABLES="test-ck")
                Path(self.CONFIG_FILE).write_text("[physics]\nlimb_darkening_law = square_root\n")

            def body_error(self):
                self.mutate()
                raise RuntimeError("body error")

            def assertion_failure(self):
                self.mutate()
                self.fail("expected failure")

            def setup_error(self):
                self.fail("test should not run")

            def clean(self):
                self.assertEqual(settings.settings_serializer(), expected)

        orders = [
            ["mutate", "body_error", "clean", "setup_error", "assertion_failure", "clean"],
            ["clean", "assertion_failure", "setup_error", "clean", "body_error", "mutate"],
        ]
        for order in orders * 2:
            for name in order:
                # Even the first test must discard a contaminated runtime state.
                settings.configure(MAX_CURVE_DATAPOINTS=99, MESH_GENERATOR="auto")
                result = unittest.TestResult()
                Probe(name).run(result)
                self.assertEqual(len(result.errors), int(name in ("body_error", "setup_error")))
                self.assertEqual(len(result.failures), int(name == "assertion_failure"))
                self.assertEqual(settings.settings_serializer(), expected)
                self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], expected["CK04_ATM_TABLES"])
        self.assertTrue(all(state == expected for state in observed))

    def test_cleanup_registered_before_initial_reset_can_fail(self):
        expected = settings.settings_serializer()

        class Probe(ElisaTestCase):
            attempts = 0

            def reset_test_settings(self):
                self.attempts += 1
                if self.attempts == 1:
                    settings.MAX_CURVE_DATAPOINTS = 9
                    raise RuntimeError("initial reset failed")
                super().reset_test_settings()

            def runTest(self):
                self.fail("test should not run")

        probe = Probe()
        result = unittest.TestResult()
        probe.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(probe.attempts, 2)
        self.assertEqual(settings.settings_serializer(), expected)

    def test_reset_loads_test_ini_over_defaults(self):
        # Prove the INI is actually parsed; omitted values still use built-ins.
        settings.configure(MAX_CURVE_DATAPOINTS=123, MESH_GENERATOR="auto")
        with patch.object(self, "touch_default_config"):
            Path(self.CONFIG_FILE).write_text("[computational]\nmesh_generator = improved_trapezoidal\n")
            self.reset_test_settings()
        self.assertEqual(settings.MESH_GENERATOR, "improved_trapezoidal")
        self.assertEqual(settings.MAX_CURVE_DATAPOINTS, settings.DEFAULT_SETTINGS["MAX_CURVE_DATAPOINTS"])

    def test_mocked_scientific_results_do_not_survive_cleanup(self):
        from tests.test_custom_ld_and_atm import TestCurvesUtilsCachingAndBroadcast

        result = unittest.TestResult()
        TestCurvesUtilsCachingAndBroadcast("test_interpolated_ld_cached").run(result)
        self.assertTrue(result.wasSuccessful(), result.errors or result.failures)
        self.assertEqual(interpolate_on_ld_grid_cached.cache_info().currsize, 0)


@pytest.mark.parametrize('previous_disable', [logging.NOTSET, 27])
@pytest.mark.parametrize('failure', [None, 'class_setup', 'body'])
def test_elisa_class_cleanup_restores_logging_threshold(previous_disable, failure):
    original = logging.root.manager.disable
    observed = []

    class Probe(ElisaTestCase):
        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            observed.append(logging.root.manager.disable)
            if failure == 'class_setup':
                raise RuntimeError('class setup failed')

        def runTest(self):
            self.assertEqual(logging.root.manager.disable, logging.CRITICAL)
            if failure == 'body':
                raise RuntimeError('test body failed')

    class FollowingClass(unittest.TestCase):
        def runTest(self):
            observed.append(logging.root.manager.disable)
            self.assertEqual(logging.root.manager.disable, previous_disable)

    try:
        logging.disable(previous_disable)
        result = unittest.TestResult()
        unittest.TestSuite([Probe(), FollowingClass()]).run(result)
        assert len(result.errors) == int(failure is not None), result.errors
        assert not result.failures, result.failures
        assert observed == [logging.CRITICAL, previous_disable]
        assert logging.root.manager.disable == previous_disable
    finally:
        logging.disable(original)


def test_mean_anomaly_solver_defaults_for_existing_ini(elisa_ini: IniFixture) -> None:
    ini_path, _ = elisa_ini
    settings.configure(MEAN_ANOMALY_SOLVER=settings.DEFAULT_SETTINGS["MEAN_ANOMALY_SOLVER"])
    settings.read_and_update_config(conf_path=str(ini_path))
    loaded_settings = settings

    assert loaded_settings.MEAN_ANOMALY_SOLVER == "vectorized"
    assert loaded_settings.DEFAULT_SETTINGS["MEAN_ANOMALY_SOLVER"] == "vectorized"
    assert loaded_settings.settings_serializer()["MEAN_ANOMALY_SOLVER"] == "vectorized"


@pytest.mark.parametrize("solver", ["scalar", "vectorized"])
def test_mean_anomaly_solver_ini_and_updates(elisa_ini: IniFixture, solver: str) -> None:
    ini_path, _ = elisa_ini
    ini_path.write_text(f"[computational]\nMEAN_ANOMALY_SOLVER = {solver}\n")
    settings.read_and_update_config(conf_path=str(ini_path))
    loaded_settings = settings
    assert loaded_settings.MEAN_ANOMALY_SOLVER == solver

    replacement_solver = "scalar" if solver == "vectorized" else "vectorized"
    with loaded_settings.override(MEAN_ANOMALY_SOLVER=replacement_solver):
        assert loaded_settings.settings_serializer()["MEAN_ANOMALY_SOLVER"] == replacement_solver
    assert loaded_settings.MEAN_ANOMALY_SOLVER == solver

    loaded_settings.MEAN_ANOMALY_SOLVER = replacement_solver
    assert loaded_settings.settings_serializer()["MEAN_ANOMALY_SOLVER"] == replacement_solver


def test_mean_anomaly_solver_invalid_ini_is_atomic(elisa_ini: IniFixture) -> None:
    ini_path, _ = elisa_ini
    settings.read_and_update_config(conf_path=str(ini_path))
    loaded_settings = settings
    original_settings = loaded_settings.settings_serializer()
    invalid_configuration = textwrap.dedent("""
        [physics]
        reflection_effect_iterations = 19

        [computational]
        mean_anomaly_solver = invalid
    """)
    ini_path.write_text(invalid_configuration)

    with pytest.raises(ValueError, match="MEAN_ANOMALY_SOLVER"):
        loaded_settings.read_and_update_config(conf_path=str(ini_path))
    assert loaded_settings.settings_serializer() == original_settings

    code = textwrap.dedent("""
        import pytest

        with pytest.raises(ValueError, match="MEAN_ANOMALY_SOLVER"):
            from elisa import settings
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "ELISA_CONFIG": str(ini_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    "invalid_solver",
    [
        pytest.param("invalid", id="unsupported-name"),
        pytest.param("Vectorized", id="wrong-case"),
        pytest.param("", id="empty-name"),
        pytest.param(None, id="none"),
        pytest.param(1, id="integer"),
    ],
)
def test_mean_anomaly_solver_invalid_update_is_atomic(elisa_ini: IniFixture, invalid_solver: object) -> None:
    loaded_settings = settings
    original_settings = loaded_settings.settings_serializer()

    with pytest.raises(ValueError, match="MEAN_ANOMALY_SOLVER"):
        loaded_settings.configure(REFLECTION_EFFECT_ITERATIONS=19, MEAN_ANOMALY_SOLVER=invalid_solver)

    assert loaded_settings.settings_serializer() == original_settings


@pytest.mark.parametrize("solver", list(MeanAnomalySolver))
def test_mean_anomaly_solver_accepts_enum_members(elisa_ini: IniFixture, solver: MeanAnomalySolver) -> None:
    loaded_settings = settings

    with loaded_settings.override(MEAN_ANOMALY_SOLVER=solver):
        assert loaded_settings.MEAN_ANOMALY_SOLVER == solver.value
        assert loaded_settings.settings_serializer()["MEAN_ANOMALY_SOLVER"] == solver.value
        assert str(solver) == solver.value


@pytest.mark.parametrize("backend", list(IntersectionBackend))
def test_intersection_backend_accepts_enum_members(elisa_ini: IniFixture, backend: IntersectionBackend) -> None:
    loaded_settings = settings
    original_settings = loaded_settings.settings_serializer()

    with loaded_settings.override(INTERSECTION_BACKEND=backend):
        assert loaded_settings.INTERSECTION_BACKEND == backend.value
        assert loaded_settings.settings_serializer()["INTERSECTION_BACKEND"] == backend.value
        assert str(backend) == backend.value

    assert loaded_settings.settings_serializer() == original_settings


@pytest.mark.parametrize("legacy_atlas", ["k93", "bb"])
def test_legacy_ini_atlas_is_ignored_without_deprecation(elisa_ini: IniFixture, legacy_atlas: str) -> None:
    ini_path, expected = elisa_ini
    for setting_name in ("PASSBAND_TABLES", "LD_TABLES", "CASTELLI_KURUCZ_04_ATM_TABLES", "KURUCZ_93_ATM_TABLES"):
        Path(str(expected[setting_name])).mkdir(parents=True, exist_ok=True)

    configuration = ini_path.read_text().replace("suppress_warnings = True", "suppress_warnings = False")
    configuration = configuration.replace("[support]", f"[support]\natlas = {legacy_atlas}")
    ini_path.write_text(configuration)

    settings.configure(ATM_ATLAS=settings.DEFAULT_SETTINGS["ATM_ATLAS"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        settings.read_and_update_config(conf_path=str(ini_path))
        loaded_settings = settings

    assert loaded_settings.ATM_ATLAS == loaded_settings.DEFAULT_SETTINGS["ATM_ATLAS"]
