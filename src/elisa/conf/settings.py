from __future__ import annotations

import os
import warnings
from configparser import ConfigParser
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from elisa.base.error import raise_attribute_error, raise_value_error
from elisa.conf.enums import IntersectionBackend, LoggingProfile, MeanAnomalySolver
from elisa.conf.logconfig import configure_logging
from elisa.conf.parsers import parse_tuple_interval
from elisa.managers.settings_manager import DefaultSettings, SettingsManager
from elisa.schema_registry import registry

if TYPE_CHECKING:
    from collections.abc import Iterator

first_time_user = False
c_parse = ConfigParser()
env_variable_config = Path(os.environ.get("ELISA_CONFIG", ""))
venv_config = Path(os.environ.get("VIRTUAL_ENV", "")) / "conf" / "elisa_conf.ini"
default_config = Path(DefaultSettings.HOME) / "config.ini"

# read configuration file
if env_variable_config.is_file():
    config_file = str(env_variable_config)
elif venv_config.is_file():
    config_file = str(venv_config)
elif default_config.is_file():
    config_file = str(default_config)
else:
    # provide a minimal configuration for first time running users
    config_file = str(default_config)
    first_time_user = True
    SettingsManager.run()


class _Const:
    # PASSBAND RELATED *************************************************************************************************
    PASSBANDS: ClassVar = [
        "bolometric",
        "Generic.Bessell.U",
        "Generic.Bessell.B",
        "Generic.Bessell.V",
        "Generic.Bessell.R",
        "Generic.Bessell.I",
        "SLOAN.SDSS.u",
        "SLOAN.SDSS.g",
        "SLOAN.SDSS.r",
        "SLOAN.SDSS.i",
        "SLOAN.SDSS.z",
        "Generic.Stromgren.u",
        "Generic.Stromgren.v",
        "Generic.Stromgren.b",
        "Generic.Stromgren.y",
        "Kepler",
        # 'GaiaDR2',
        "TESS",
        "Gaia.2010.G",
        "Gaia.2010.BP",
        "Gaia.2010.RP",
    ]

    PASSBAND_DATAFRAME_THROUGHPUT = "throughput"
    PASSBAND_DATAFRAME_WAVE = "wavelength"
    # ******************************************************************************************************************

    # ATM RELATED ******************************************************************************************************
    ATM_MODEL_DATAFRAME_FLUX = "flux"
    ATM_MODEL_DATAFRAME_WAVE = "wave"

    ATM_MODEL_DATAFARME_DTYPES: ClassVar = {
        ATM_MODEL_DATAFRAME_FLUX: float,
        ATM_MODEL_DATAFRAME_WAVE: float,
    }
    # ******************************************************************************************************************

    # LIMB DARKENING RELATED *******************************************************************************************
    LD_LAW_TO_FILE_PREFIX: ClassVar = {
        "linear": "lin",
        "cosine": "lin",
        "logarithmic": "log",
        "square_root": "sqrt",
    }

    AVAILABLE_LD_LAWS: ClassVar = list(LD_LAW_TO_FILE_PREFIX.keys())

    LD_LAW_CFS_COLUMNS: ClassVar = {
        "linear": ["xlin"],
        "cosine": ["xlin"],
        "logarithmic": ["xlog", "ylog"],
        "square_root": ["xsqrt", "ysqrt"],
    }

    BASIC_COLS: ClassVar = ["temperature", "gravity"]
    LD_LAW_COLS_ORDER: ClassVar = {
        "linear": BASIC_COLS + LD_LAW_CFS_COLUMNS["linear"],
        "cosine": BASIC_COLS + LD_LAW_CFS_COLUMNS["cosine"],
        "logarithmic": BASIC_COLS + LD_LAW_CFS_COLUMNS["logarithmic"],
        "square_root": BASIC_COLS + LD_LAW_CFS_COLUMNS["square_root"],
    }

    LD_DOMAIN_COLS: ClassVar = ["temperature", "gravity"]
    # ******************************************************************************************************************

    BINARY_COUNTERPARTS: ClassVar = {"primary": "secondary", "secondary": "primary"}

    ATLAS_TO_ATM_FILE_PREFIX: ClassVar = {
        "castelli": "ck",
        "castelli-kurucz": "ck",
        "ck": "ck",
        "ck04": "ck",
        "kurucz": "k",
        "k": "k",
        "k93": "k",
    }

    ATM_ATLAS_NORMALIZER: ClassVar = {
        **ATLAS_TO_ATM_FILE_PREFIX,
        "bb": "bb",
        "black_body": "bb",
    }

    ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX: ClassVar = {
        "temperature": "TEMPERATURE_LIST_ATM",
        "gravity": "GRAVITY_LIST_ATM",
        "metallicity": "METALLICITY_LIST_ATM",
    }

    DATETIME_MASK = "%Y-%m-%dT%H.%M.%S"
    DATE_MASK = "%Y-%m-%d"

    DATASET_MANDATORY_KWARGS: ClassVar = ("x_data", "y_data", "x_unit", "y_unit")
    DATASET_OPTIONAL_KWARGS: ClassVar = ("y_err",)

    DELIM_WHITESPACE = r"\s+|\t+|\s+\t+|\t+\s+"
    DATA_PATH = Path(__file__).resolve().parent.parent / "data"

    # paths to mesh correction factors
    PATH_TO_SINGLE_CORRECTIONS = DATA_PATH / "mesh_corrections" / "correction_factors_single.npy"
    PATH_TO_DETACHED_CORRECTIONS = DATA_PATH / "mesh_corrections" / "correction_factors_detached.npy"
    PATH_TO_OVER_CONTACT_CORRECTIONS = DATA_PATH / "mesh_corrections" / "correction_factors_over-contact.npy"
    PATH_TO_ALBEDOS = DATA_PATH / "albedos.json"
    PATH_TO_BETA = DATA_PATH / "gravity_darkening.json"


class Settings(_Const):
    """Mutable process settings, changed between calculations.

    Effective values live only on the singleton instance. Transactions replace
    its attribute dictionary after validation; numerical reads stay ordinary
    Python attribute reads. Existing scientific objects retain their inputs.
    """

    # Type declarations only: effective values are initialized on the instance.
    CONFIG_FILE: str
    LOG_CONFIG: str
    SUPPRESS_WARNINGS: bool
    SUPPRESS_LOGGER: bool | None
    HOME: str
    REFLECTION_EFFECT: bool
    REFLECTION_EFFECT_ITERATIONS: int
    LIMB_DARKENING_LAW: str
    DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT: float
    SURFACE_DISPLACEMENT_TOL: float
    RV_METHOD: str
    RV_LAMBDA_INTERVAL: tuple[float, float]
    MAX_DISCRETIZATION_FACTOR: float
    MIN_DISCRETIZATION_FACTOR: float
    DEFAULT_DISCRETIZATION_FACTOR: int
    NUMBER_OF_THREADS: int
    NUMBER_OF_PROCESSES: int
    INTERSECTION_BACKEND: str
    MEAN_ANOMALY_SOLVER: str
    NUMBER_OF_MCMC_PROCESSES: int
    MAX_NU_SEPARATION: float
    MAX_D_FLUX: float
    MAX_SPOT_D_LONGITUDE: float
    MAX_SOLVER_ITERS: int
    MAX_CURVE_DATAPOINTS: int
    MIN_POINTS_IN_ECLIPSE: int
    TIMER: float
    PASSBAND_TABLES: str
    LD_TABLES: str
    CK04_ATM_TABLES: str
    K93_ATM_TABLES: str
    ATM_ATLAS: str
    MESH_GENERATOR: str
    DEFORMATION_TOL: float
    PULSATION_MODEL: str
    MCMC_SAVE_INTERVAL: int
    MCMC_AUTOCORR_TOL: int
    CUDA: bool
    USE_SINGLE_LD_COEFFICIENTS: bool
    USE_INTERPOLATION_APPROXIMATION: bool
    USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION: bool
    USE_SIMILAR_NEIGHBOURS_APPROXIMATION: bool
    MAGNITUDE_SYSTEM: str
    ATLAS_TO_BASE_DIR: MappingProxyType[str, str]

    _instance = None
    BASE_DIR = DefaultSettings.BASE_DIR
    SCHEMA_REGISTRY = registry.Registry()
    FIRST_TIME_USER = first_time_user
    _fields = (
        "CONFIG_FILE",
        "LOG_CONFIG",
        "SUPPRESS_WARNINGS",
        "SUPPRESS_LOGGER",
        "HOME",
        "REFLECTION_EFFECT",
        "REFLECTION_EFFECT_ITERATIONS",
        "LIMB_DARKENING_LAW",
        "DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT",
        "SURFACE_DISPLACEMENT_TOL",
        "RV_METHOD",
        "RV_LAMBDA_INTERVAL",
        "MAX_DISCRETIZATION_FACTOR",
        "MIN_DISCRETIZATION_FACTOR",
        "DEFAULT_DISCRETIZATION_FACTOR",
        "NUMBER_OF_THREADS",
        "NUMBER_OF_PROCESSES",
        "INTERSECTION_BACKEND",
        "MEAN_ANOMALY_SOLVER",
        "NUMBER_OF_MCMC_PROCESSES",
        "MAX_NU_SEPARATION",
        "MAX_D_FLUX",
        "MAX_SPOT_D_LONGITUDE",
        "MAX_SOLVER_ITERS",
        "MAX_CURVE_DATAPOINTS",
        "MIN_POINTS_IN_ECLIPSE",
        "TIMER",
        "PASSBAND_TABLES",
        "LD_TABLES",
        "CK04_ATM_TABLES",
        "K93_ATM_TABLES",
        "ATM_ATLAS",
        "MESH_GENERATOR",
        "DEFORMATION_TOL",
        "PULSATION_MODEL",
        "MCMC_SAVE_INTERVAL",
        "MCMC_AUTOCORR_TOL",
        "CUDA",
        "USE_SINGLE_LD_COEFFICIENTS",
        "USE_INTERPOLATION_APPROXIMATION",
        "USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION",
        "USE_SIMILAR_NEIGHBOURS_APPROXIMATION",
        "MAGNITUDE_SYSTEM",
    )
    _defaults: ClassVar = {name: getattr(DefaultSettings, name) for name in _fields if name != "CONFIG_FILE"}
    _defaults["CONFIG_FILE"] = config_file
    DEFAULT_SETTINGS = MappingProxyType(_defaults)

    def __new__(cls) -> Settings:  # noqa: PYI034
        """Create or return the process singleton."""
        if cls._instance is None:
            instance = super().__new__(cls)
            object.__setattr__(instance, "__dict__", dict(cls._defaults))
            instance._update_atlas_to_base_dir()
            instance.read_and_update_config()
            instance._create_home()
            cls._instance = instance
        return cls._instance

    def __setattr__(self, name: str, value: object) -> None:
        """Route public writes through the same transaction as configure()."""
        if name not in self._fields:
            raise_attribute_error(f"{name} is not a mutable setting")
        self.configure(**{name: value})

    def __delattr__(self, name: str) -> None:
        raise_attribute_error(f"Settings attributes cannot be deleted: {name}")

    def settings_serializer(self) -> dict:
        """Return an independent dictionary of the effective user settings."""
        values = self.__dict__
        return {name: values[name] for name in self._fields}

    @staticmethod
    def load_conf(path: str) -> None:
        """Load and parse a JSON logging configuration file.

        :param path: Path to the JSON configuration file.
        :type path: str
        :returns: None
        :rtype: None
        """
        settings.configure_logging(path)

    def set_up_logging(self) -> None:
        """Apply the selected logging policy without resolving LOG_CONFIG in place.

        :returns: None.
        """
        profile = self.LOG_CONFIG
        # The historical default points at a nonexistent managers/logging_schemas
        # file. Retain its startup fallback without hiding missing custom files.
        if profile == DefaultSettings.LOG_CONFIG and not Path(profile).is_file():
            profile = LoggingProfile.DEFAULT.value
        self.configure_logging(profile)

    def configure_logging(self, profile: str) -> None:
        """Configure logging independently of other Settings mutations.

        :param profile: ``default``, ``fit`` or a trusted custom JSON file path.
        :raises ValueError: If the profile or logging policy is invalid.
        :returns: None.
        """
        self.configure(LOG_CONFIG=profile)

    @contextmanager
    def logging_context(self, profile: str) -> Iterator[None]:
        """Temporarily change logging and reload the previously selected JSON policy.

        :param profile: Built-in profile name or trusted custom JSON file path.
        :yields: Control with the temporary policy applied.
        """
        previous_selector = self.LOG_CONFIG
        self.configure_logging(profile)
        try:
            yield
        finally:
            self.configure_logging(previous_selector)

    def read_and_update_config(self, *, conf_path: str | None = None) -> None:
        """Read configuration from file and update settings.

        :param conf_path: Path to the configuration file. If None, uses CONFIG_FILE.
        :type conf_path: str | None
        :returns: None
        :rtype: None
        """
        if not conf_path:
            conf_path = self.CONFIG_FILE

        if not Path(conf_path).is_file():
            msg = (
                "Couldn't find configuration file. Using default settings.\n"
                "   To customize configuration using file either\n"
                "    - specify config with environment variable ELISA_CONFIG\n"
                "    - add conf/elisa_conf.ini under your virtualenv root \n"
            )
            warnings.warn(msg, Warning, stacklevel=2)
            return

        parser = ConfigParser()
        parser.read(conf_path)
        self.update_config(parser)

    @contextmanager
    def override(self, **kwargs: object) -> Iterator[None]:
        """Temporarily configure settings, restoring all effective values on exit.

        Nested contexts restore their own entry state, including settings changed
        in the body that were not named in the override.
        """
        previous = self.settings_serializer()
        self.configure(**kwargs)
        try:
            yield
        finally:
            self.configure(**previous)

    def configure(self, **kwargs: object) -> None:
        """Validate a full candidate and publish it once, or retain all old values.

        Constants, services and derived values are not configuration fields.
        Logging is applied last before publication; arbitrary custom dictConfig
        side effects cannot be rolled back, but failed logging never publishes
        candidate Settings values.
        """
        unknown = kwargs.keys() - self._fields
        if unknown:
            raise_value_error(f"Unknown or read-only setting(s): {', '.join(sorted(unknown))}")
        candidate = self.settings_serializer()
        candidate.update(kwargs)
        self._validate_candidate(candidate)
        candidate["ATLAS_TO_BASE_DIR"] = self._atlas_to_base_dir(candidate)
        if "HOME" in kwargs:
            Path(candidate["HOME"]).mkdir(parents=True, exist_ok=True)
        if "LOG_CONFIG" in kwargs:
            profile = candidate["LOG_CONFIG"]
            if profile == DefaultSettings.LOG_CONFIG and not Path(profile).is_file():
                profile = LoggingProfile.DEFAULT.value
            configure_logging(profile)
            candidate["LOG_CONFIG"] = str(profile)
        object.__setattr__(self, "__dict__", candidate)

    @staticmethod
    def _validate_candidate(values: dict) -> None:  # noqa: C901, PLR0912
        """Use INI types and constraints without restricting scientific modes."""
        float_fields = {
            "MAX_DISCRETIZATION_FACTOR",
            "MIN_DISCRETIZATION_FACTOR",
        }
        for name, value in values.items():
            default = Settings._defaults[name]
            valid = True
            if name == "RV_LAMBDA_INTERVAL":
                valid = (
                    isinstance(value, tuple)
                    and len(value) == 2  # noqa: PLR2004
                    and all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in value)
                    and value[0] <= value[1]
                )
            elif name == "SUPPRESS_LOGGER":
                valid = value is None or isinstance(value, (bool, np.bool_))
            elif isinstance(default, bool):
                valid = isinstance(value, (bool, np.bool_))
            elif name in float_fields or isinstance(default, float):
                valid = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            elif isinstance(default, int):
                valid = isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            elif name in {
                "CONFIG_FILE",
                "LOG_CONFIG",
                "HOME",
                "PASSBAND_TABLES",
                "LD_TABLES",
                "CK04_ATM_TABLES",
                "K93_ATM_TABLES",
            }:
                valid = isinstance(value, (str, os.PathLike))
                if valid:
                    values[name] = str(value)
            elif isinstance(default, str):
                valid = isinstance(value, str)
            if not valid:
                raise_value_error(f"Invalid value for {name}: {value!r}")
        if values["LIMB_DARKENING_LAW"] not in _Const.AVAILABLE_LD_LAWS:
            raise_value_error(
                f"{values['LIMB_DARKENING_LAW']} is not valid name of limb darkening law. "
                f"Available limb darkening laws are: {_Const.AVAILABLE_LD_LAWS}"
            )
        backends = tuple(backend.value for backend in IntersectionBackend)
        if values["INTERSECTION_BACKEND"] not in backends:
            raise_value_error(f"INTERSECTION_BACKEND must be one of: {', '.join(backends)}")
        solvers = tuple(solver.value for solver in MeanAnomalySolver)
        if values["MEAN_ANOMALY_SOLVER"] not in solvers:
            raise_value_error(f"MEAN_ANOMALY_SOLVER must be one of: {', '.join(solvers)}")
        if values["NUMBER_OF_THREADS"] <= 0:
            raise_value_error("Invalid value for `number_of_threads`, allowed >= 1")
        for name in ("NUMBER_OF_PROCESSES", "NUMBER_OF_MCMC_PROCESSES"):
            cores = os.cpu_count() or 1
            if values[name] > cores:
                if not values["SUPPRESS_WARNINGS"]:
                    warnings.warn(
                        f"argument {name.lower()} is too big, fallback to number of machine cores", stacklevel=3,
                    )
                values[name] = cores

    def _create_home(self) -> None:
        """Create the home directory if it does not exist.

        :returns: None
        :rtype: None
        """
        Path(self.HOME).mkdir(parents=True, exist_ok=True)

    def update_config(self, parser: ConfigParser | None = None) -> None:  # noqa: C901, PLR0912, PLR0915
        """Update settings from the parsed configuration file.

        Reads all configuration sections (general, physics, computational, support)
        and publishes the validated candidate in one transaction.

        :returns: None
        :rtype: None
        """
        parser = c_parse if parser is None else parser
        candidate = SimpleNamespace(**self.settings_serializer())
        candidate.CUDA = False
        # **************************************************************************************************************
        if parser.has_section("general"):
            candidate.SUPPRESS_WARNINGS = parser.getboolean(
                "general", "suppress_warnings", fallback=candidate.SUPPRESS_WARNINGS,
            )
            log_config = parser.get("general", "log_config", fallback=None)
            if log_config is None:
                candidate.LOG_CONFIG = self.LOG_CONFIG
            else:
                candidate.LOG_CONFIG = log_config

            candidate.SUPPRESS_LOGGER = parser.getboolean(
                "general", "suppress_logger", fallback=candidate.SUPPRESS_LOGGER,
            )
            candidate.HOME = parser.get("general", "home", fallback=candidate.HOME)
        # **************************************************************************************************************
        if parser.has_section("physics"):
            candidate.REFLECTION_EFFECT = parser.getboolean(
                "physics", "reflection_effect", fallback=candidate.REFLECTION_EFFECT,
            )
            candidate.REFLECTION_EFFECT_ITERATIONS = parser.getint(
                "physics", "reflection_effect_iterations", fallback=candidate.REFLECTION_EFFECT_ITERATIONS,
            )
            candidate.LIMB_DARKENING_LAW = parser.get(
                "physics", "limb_darkening_law", fallback=candidate.LIMB_DARKENING_LAW,
            )
            candidate.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT = parser.getfloat(
                "physics",
                "default_temperature_perturbation_phase_shift",
                fallback=candidate.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT,
            )
            candidate.SURFACE_DISPLACEMENT_TOL = parser.getfloat(
                "physics", "surface_displacement_tol", fallback=candidate.SURFACE_DISPLACEMENT_TOL,
            )
            candidate.RV_METHOD = parser.get("physics", "rv_method", fallback=candidate.RV_METHOD)

            if parser.has_option("physics", "rv_lambda_interval"):
                candidate.RV_LAMBDA_INTERVAL = parse_tuple_interval(
                    parser.get("physics", "rv_lambda_interval"),
                    name="rv_lambda_interval",
                )

            candidate.PULSATION_MODEL = parser.get("physics", "pulsation_model", fallback=candidate.PULSATION_MODEL)
        # **************************************************************************************************************
        if parser.has_section("computational"):
            candidate.MEAN_ANOMALY_SOLVER = parser.get(
                "computational", "mean_anomaly_solver", fallback=candidate.MEAN_ANOMALY_SOLVER,
            )
            backend = parser.get("computational", "intersection_backend", fallback=candidate.INTERSECTION_BACKEND)
            candidate.INTERSECTION_BACKEND = backend
            candidate.MAX_DISCRETIZATION_FACTOR = parser.getfloat(
                "computational", "max_discretization_factor", fallback=candidate.MAX_DISCRETIZATION_FACTOR,
            )
            candidate.MIN_DISCRETIZATION_FACTOR = parser.getfloat(
                "computational", "min_discretization_factor", fallback=candidate.MIN_DISCRETIZATION_FACTOR,
            )
            candidate.DEFAULT_DISCRETIZATION_FACTOR = parser.getint(
                "computational",
                "default_discretization_factor",
                fallback=candidate.DEFAULT_DISCRETIZATION_FACTOR,
            )
            candidate.NUMBER_OF_THREADS = parser.getint(
                "computational", "number_of_threads", fallback=candidate.NUMBER_OF_THREADS,
            )
            candidate.NUMBER_OF_PROCESSES = parser.getint(
                "computational", "number_of_processes", fallback=candidate.NUMBER_OF_PROCESSES,
            )
            candidate.NUMBER_OF_MCMC_PROCESSES = parser.getint(
                "computational", "number_of_mcmc_processes", fallback=candidate.NUMBER_OF_MCMC_PROCESSES,
            )

            candidate.MAX_NU_SEPARATION = parser.getfloat(
                "computational", "max_nu_separation", fallback=candidate.MAX_NU_SEPARATION,
            )
            candidate.MAX_D_FLUX = parser.getfloat("computational", "max_d_flux", fallback=candidate.MAX_D_FLUX)
            candidate.MAX_SPOT_D_LONGITUDE = parser.getfloat(
                "computational", "max_spot_d_longitude", fallback=candidate.MAX_SPOT_D_LONGITUDE,
            )
            candidate.MAX_SOLVER_ITERS = parser.getint(
                "computational", "max_solver_iters", fallback=candidate.MAX_SOLVER_ITERS,
            )
            candidate.MAX_CURVE_DATAPOINTS = parser.getint(
                "computational", "max_curve_datapoints", fallback=candidate.MAX_CURVE_DATAPOINTS,
            )
            candidate.MIN_POINTS_IN_ECLIPSE = parser.getint(
                "computational", "min_points_in_eclipse", fallback=candidate.MIN_POINTS_IN_ECLIPSE,
            )
            candidate.MESH_GENERATOR = parser.get("computational", "mesh_generator", fallback=candidate.MESH_GENERATOR)
            candidate.DEFORMATION_TOL = parser.getfloat(
                "computational", "deformation_tol", fallback=candidate.DEFORMATION_TOL,
            )
            candidate.MCMC_SAVE_INTERVAL = parser.getint(
                "computational", "mcmc_save_interval", fallback=candidate.MCMC_SAVE_INTERVAL,
            )
            candidate.MCMC_AUTOCORR_TOL = parser.getint(
                "computational", "mcmc_autocorr_tol", fallback=candidate.MCMC_AUTOCORR_TOL,
            )

            candidate.CUDA = parser.getboolean("computational", "cuda", fallback=candidate.CUDA)

            if candidate.CUDA:
                try:
                    # noinspection PyUnresolvedReferences
                    import cumpy  # noqa: PLC0415

                    if not cumpy.cuda.is_available():
                        candidate.CUDA = False
                        warnings.warn(
                            "You have no CUDA enabled/available on your device. Runtime continue with CPU.",
                            UserWarning,
                            stacklevel=2,
                        )

                except ImportError:
                    warnings.warn(
                        "You need to install `pytorch` with cuda to be able to use CUDA features. Fallback to CPU.",
                        UserWarning,
                        stacklevel=2,
                    )
                    candidate.CUDA = False

            candidate.USE_SINGLE_LD_COEFFICIENTS = parser.getboolean(
                "computational", "use_single_ld_coefficients", fallback=candidate.USE_SINGLE_LD_COEFFICIENTS,
            )

            candidate.USE_INTERPOLATION_APPROXIMATION = parser.getboolean(
                "computational",
                "use_interpolation_approximation",
                fallback=candidate.USE_INTERPOLATION_APPROXIMATION,
            )

            candidate.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION = parser.getboolean(
                "computational",
                "use_symmetrical_counterparts_approximation",
                fallback=candidate.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION,
            )

            candidate.USE_SIMILAR_NEIGHBOURS_APPROXIMATION = parser.getboolean(
                "computational",
                "use_similar_neighbours_approximation",
                fallback=candidate.USE_SIMILAR_NEIGHBOURS_APPROXIMATION,
            )

        # **************************************************************************************************************
        if parser.has_section("support"):
            candidate.LD_TABLES = parser.get("support", "ld_tables", fallback=candidate.LD_TABLES)

            if not Path(candidate.LD_TABLES).is_dir() and not candidate.SUPPRESS_WARNINGS:
                warnings.warn(
                    f"path {candidate.LD_TABLES} to limb darkening tables doesn't exists\n"
                    f"Specifiy it in elisa_conf.ini file",
                    stacklevel=2,
                )

            candidate.CK04_ATM_TABLES = parser.get(
                "support", "castelli_kurucz_04_atm_tables", fallback=candidate.CK04_ATM_TABLES,
            )

            if not Path(candidate.CK04_ATM_TABLES).is_dir() and not candidate.SUPPRESS_WARNINGS:
                warnings.warn(
                    f"path {candidate.CK04_ATM_TABLES}\n"
                    f"to castelli-kurucz 2004 atmosphere atlas doesn't exists\n"
                    f"Specifiy it in elisa_conf.ini file",
                    stacklevel=2,
                )

            candidate.K93_ATM_TABLES = parser.get("support", "kurucz_93_atm_tables", fallback=candidate.K93_ATM_TABLES)

            if not Path(candidate.K93_ATM_TABLES).is_dir() and not candidate.SUPPRESS_WARNINGS:
                warnings.warn(
                    f"path {candidate.K93_ATM_TABLES}\n"
                    "to kurucz 1993 atmosphere atlas doesn't exists\n"
                    "Specifiy it in elisa_conf.ini file",
                    UserWarning,
                    stacklevel=2,
                )

            candidate.PASSBAND_TABLES = parser.get("support", "passband_tables", fallback=candidate.PASSBAND_TABLES)

            if not Path(candidate.PASSBAND_TABLES).is_dir() and not candidate.SUPPRESS_WARNINGS:
                warnings.warn(
                    f"path {candidate.PASSBAND_TABLES} to passband tables doesn't exists\n"
                    f"Specifiy it in elisa_conf.ini file",
                    UserWarning,
                    stacklevel=2,
                )

        # **************************************************************************************************************

        changes = vars(candidate)
        if not parser.has_section("general"):
            # Preserve the existing rule: file loading sets up logging only
            # when the general section is present.
            changes.pop("LOG_CONFIG")
        self.configure(**changes)

    @staticmethod
    def _atlas_to_base_dir(values: dict) -> MappingProxyType:
        """Derive a fresh, read-only map as part of the published candidate."""
        return MappingProxyType(
            {
                alias: values["CK04_ATM_TABLES" if prefix == "ck" else "K93_ATM_TABLES"]
                for alias, prefix in _Const.ATLAS_TO_ATM_FILE_PREFIX.items()
            },
        )

    def _update_atlas_to_base_dir(self) -> None:
        """Compatibility helper; normal transactions already synchronize the map."""
        candidate = dict(self.__dict__)
        candidate["ATLAS_TO_BASE_DIR"] = self._atlas_to_base_dir(candidate)
        object.__setattr__(self, "__dict__", candidate)


settings = Settings()
