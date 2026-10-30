"""Edit the uppercase settings and run without arguments; see README.md.

All patches are process-local and restored. No production files are modified.
"""
from __future__ import annotations

import ast
import cProfile
import hashlib
import io
import json
import os
import platform
import sys
import warnings
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import astropy
from astropy import astronomical_constants, physical_constants
import numba
import numpy as np
import pandas as pd
import scipy
from astropy import units as u
from scipy.interpolate import _ndgriddata

from elisa import BinarySystem, Observer, SingleSystem, Star, const, ld, settings, utils
from elisa.base.curves import utils as curve_utils
from elisa.base.error import LimbDarkeningError, LimbDarkeningSourceError
from elisa.binary_system.curves import utils as binary_utils
from elisa.buffer import buffer
from elisa.single_system.curves import utils as single_utils

# Edit these values before running. Six rounds balance every mode in every slot.
MODE = "all"  # "all", "baseline", "direct", "optimized"
OUTPUT_DIR = Path("/tmp/elisa-ld-benchmark/review")
REPETITIONS = 6
SCIENCE_ONLY = False
MICRO_ONLY = False
PROFILE = False  # Separate, instrument-free cProfile runs after wall-clock samples.
COMPARE_DIR: Path | None = None  # Optional versioned baseline/direct reference artifacts.
CASES = ("circular", "eccentric", "multiband", "single")

MODES = ("baseline", "direct", "optimized")
STATES = ("cold_source", "warm_source_new_result", "repeated_curve")
PHASES = dict(from_phase=-.6, to_phase=.6, phase_step=.01)
ERROR_MESSAGE = (
    "Limb darkening interpolation produced numpy.nan/None.\n"
    "Some of the surface parameters (t_eff, log_g, metallicity) are probably out of range.\n"
    "Adjust star parameters or provide custom LD coefficients via Star.limb_darkening_coefficients."
)
FORMAT_VERSION = 3


def model(case: str) -> BinarySystem | SingleSystem:
    if case == "single":
        # Rapid enough rotation to exercise non-uniform surface T/g, below breakup.
        star = Star(mass=2.15*u.solMass, t_eff=10000*u.K, gravity_darkening=1.,
                    polar_log_g=4.2*u.dex(u.cm/u.s**2), discretization_factor=5,
                    metallicity=0., atmosphere="bb")
        return SingleSystem(star=star, rotation_period=.7*u.d, inclination=65*u.deg,
                            gamma=0*u.km/u.s, distance=155*u.pc, reference_time=0*u.d,
                            phase_shift=0.)
    check(case in ("circular", "eccentric", "multiband"), f"Unknown case: {case}")
    eccentricity = .2 if case == "eccentric" else 0.
    primary = Star(mass=2.15*u.solMass, surface_potential=3.6, synchronicity=1., t_eff=10000*u.K,
                   gravity_darkening=1., discretization_factor=5, albedo=.6, metallicity=0., atmosphere='bb')
    secondary = Star(mass=1.2*u.solMass, surface_potential=4., synchronicity=1., t_eff=7000*u.K,
                     gravity_darkening=1., albedo=.6, metallicity=0., atmosphere='bb')
    return BinarySystem(primary=primary, secondary=secondary, argument_of_periastron=58*u.deg,
                        gamma=-30.7*u.km/u.s, period=2.5*u.d, eccentricity=eccentricity,
                        inclination=85*u.deg, distance=155*u.pc, primary_minimum_time=2440000*u.d,
                        phase_shift=0.)


def get_ld_table(passband: str, metallicity: float, law: str | None = None) -> pd.DataFrame:
    """Load a Van Hamme limb darkening table from the configured tables' directory.

    :param passband: Passband identifier.
    :param metallicity: Metallicity value.
    :param law: Limb darkening law name. If not provided, the configured default is used.
    :returns: Table content as a :class:`pandas.DataFrame`.
    :raises FileNotFoundError: If the expected CSV file is not present.
    """
    resolved_law = law if not utils.is_empty(law) else settings.LIMB_DARKENING_LAW
    filename = ld.get_ld_table_filename(passband, metallicity, law=resolved_law)
    path = Path(settings.LD_TABLES) / filename
    if not path.is_file():
        msg = f"There is no file like {path}."
        raise FileNotFoundError(msg)
    # noinspection PyArgumentList
    return pd.read_csv(path)


def get_ld_table_by_name(fname: str) -> pd.DataFrame:
    """Load a Van Hamme limb darkening table by filename from the configured tables' directory.

    :param fname: Filename of the CSV table (no directory).
    :returns: Table content as a :class:`pandas.DataFrame`.
    :raises FileNotFoundError: If the file is not present in the configured directory.
    """
    ld.logger.debug("accessing limb darkening file %s", fname)
    path = Path(settings.LD_TABLES) / fname
    if not path.is_file():
        msg = f"There is no file like {path}."
        raise FileNotFoundError(msg)
    # noinspection PyArgumentList
    return pd.read_csv(path)


def legacy_interpolate(temperature, log_g, metallicity, passband, author=None):
    """Original ordered preparation/griddata, including frame keys and errors.

    Ported from the pre-optimization ld.interpolate_on_ld_grid. No path/stat
    lookup or new cache-key/ownership machinery is added to this path.
    """
    del author
    bands = list(passband.keys()) if isinstance(passband, Mapping) else list(passband)
    t_eff = np.asarray(temperature, dtype=float)
    gravity = utils.convert_gravity_acceleration_array(np.asarray(log_g, dtype=float), units="log_cgs")
    results = {}
    ld.logger.debug("interpolating limb darkening coefficients")
    for band in bands:
        names = ld.get_relevant_ld_tables("bolometric" if band == "rv_band" else band,
                                         metallicity, settings.LIMB_DARKENING_LAW)
        columns = settings.LD_LAW_COLS_ORDER[settings.LIMB_DARKENING_LAW]
        frames = []
        for name in names:
            if name not in buffer.LD_CFS_TABLES:
                buffer.LD_CFS_TABLES[name] = get_ld_table_by_name(name)[columns]
            frames.append(buffer.LD_CFS_TABLES[name])
        buffer.reduce_buffer(buffer.LD_CFS_TABLES)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        frame = frame.drop_duplicates()
        points = frame[settings.LD_DOMAIN_COLS].to_numpy()
        values = frame[settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW]].to_numpy()
        result = np.asarray(ld.interpolate.griddata(points, values, np.column_stack((t_eff, gravity)), method="linear"))
        if np.isnan(result).any():
            raise LimbDarkeningError(ERROR_MESSAGE)
        results[band] = result
    ld.logger.debug("limb darkening coefficients interpolation finished")
    return results


def legacy_array_key(array):
    """Original ravel/round/astype/list/tuple sequence, not the new byte encoder."""
    values = np.asarray(array).ravel()
    return tuple(np.round(values, 9).astype(float).tolist()) if values.size else ()


@lru_cache(maxsize=128)
def legacy_evaluated(temps_key, logg_key, metallicity, passbands_key):
    return ld.interpolate_on_ld_grid(np.asarray(temps_key, dtype=float), np.asarray(logg_key, dtype=float),
                                     metallicity, list(passbands_key))


def legacy_component(component_instance, passbands, *, symmetry_test):
    """Original result LRU and mirroring for the benchmark's non-custom models."""
    if component_instance.limb_darkening_coefficients is not None:
        raise ValueError("This baseline reproduction only supports the benchmark's interpolated coefficients.")
    temperatures, gravity = curve_utils.generate_teff_logg_for_ld_cfs(component_instance, symmetry_test=symmetry_test)
    result = legacy_evaluated(legacy_array_key(temperatures), legacy_array_key(gravity),
                              float(component_instance.metallicity), tuple(passbands))
    if symmetry_test:
        if settings.USE_SINGLE_LD_COEFFICIENTS:
            indices = np.zeros(component_instance.temperatures.shape, dtype=int)
            result = {band: values[indices] for band, values in result.items()}
        else:
            result = {band: component_instance.mirror_face_values(values) for band, values in result.items()}
    return result


@contextmanager
def implementation(mode):
    """Choose only interpolation/component helpers; leave all physics untouched."""
    with ExitStack() as patches:
        if mode != "optimized":
            patches.enter_context(patch.object(ld, "interpolate_on_ld_grid", legacy_interpolate))
        if mode == "baseline":
            patches.enter_context(patch.object(curve_utils, "get_component_limbdarkening_cfs", legacy_component))
            patches.enter_context(patch.object(binary_utils, "get_component_limbdarkening_cfs", legacy_component))
            patches.enter_context(patch.object(single_utils, "get_component_limbdarkening_cfs", legacy_component))
        try:
            yield
        finally:
            clear()


def clear():
    ld.clear_ld_caches()
    legacy_evaluated.cache_clear()


def stats(mode):
    cache = legacy_evaluated if mode == "baseline" else curve_utils.interpolate_on_ld_grid_cached
    return {"result": cache.cache_info()._asdict(), "source": ld.get_ld_source_interpolator.cache_info()._asdict()}


def delta(before, after):
    return {layer: {name: after[layer][name] - before[layer][name] for name in ("hits", "misses")}
            for layer in before}


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metadata():
    """Fingerprint inputs once per run, not on interpolation cache hits."""
    package = Path(ld.__file__).resolve().parent
    files = sorted(path for path in package.rglob("*") if path.suffix in (".py", ".csv", ".json", ".npy", ".sc"))
    tables = Path(settings.LD_TABLES).resolve()
    for band in ("Generic.Bessell.U", "Generic.Bessell.V", "bolometric"):
        files.extend(tables / name for name in set(ld.get_relevant_ld_tables(band, 0.)))
    passbands = Path(settings.PASSBAND_TABLES).resolve()
    files.extend(sorted(passbands.rglob("*.csv")))
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    # Configuration edits must not make otherwise identical reference runs incompatible.
    syntax = ast.parse(Path(__file__).read_text())
    syntax.body = [node for node in syntax.body if not isinstance(node, (ast.Assign, ast.AnnAssign))]
    config = {name: getattr(settings, name) for name in dir(settings)
              if name.isupper() and name != "DEFAULT_SETTINGS" and not callable(getattr(settings, name))}
    # Registry's default repr contains a process-specific address, not a setting.
    registry_type = type(settings.SCHEMA_REGISTRY)
    config["SCHEMA_REGISTRY"] = f"{registry_type.__module__}.{registry_type.__qualname__}"
    output = io.StringIO()
    with redirect_stdout(output):
        np.show_config()
        scipy.show_config()
    models = {case: model(case) for case in CASES}
    values = canonical(dict(
        format_version=FORMAT_VERSION, implementation_hash=digest(ast.dump(syntax)), files=hashes,
        python=sys.version, executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                      astropy=astropy.__version__, numba=numba.__version__),
        numerical_build=output.getvalue(), numpy_error_policy=np.geterr(), settings=config,
        astropy_constants=dict(physical=physical_constants.get(), astronomical=astronomical_constants.get(),
                               solar_mass_kg=u.solMass.to(u.kg)),
        cpu_model=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else platform.processor(),
        metallicities=const.METALLICITY_LIST_LD,
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        phases=PHASES, models={case: system.to_json() for case, system in models.items()},
        discretization_radians={case: {name: star.discretization_factor for name, star in system.components.items()}
                                for case, system in models.items()},
        bands={case: ["Generic.Bessell.U", "Generic.Bessell.V"] if case == "multiband"
                     else ["Generic.Bessell.U"] for case in CASES},
    ))
    values["fingerprint"] = digest(values)
    return values


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_errors(modes):
    """Reject hull-exterior/NaN queries; preserve baseline and current diagnostics."""
    for mode in modes:
        with implementation(mode):
            for temperature in (1., np.nan):
                try:
                    ld.interpolate_on_ld_grid(np.array([temperature]), np.array([2.]), 0., ["bolometric"])
                except LimbDarkeningError as error:
                    if mode == "optimized":
                        first_line, _, guidance = ERROR_MESSAGE.partition("\n")
                        check(str(error).startswith(first_line + "\n"), "Changed interpolation error summary")
                        check(str(error).endswith(guidance), "Changed interpolation error guidance")
                        check("Passband: bolometric;" in str(error), "Missing interpolation source diagnostic")
                    else:
                        check(str(error) == ERROR_MESSAGE, f"{mode}: changed baseline interpolation error message")
                else:
                    raise AssertionError(f"{mode}: invalid query did not raise LimbDarkeningError")
            try:
                ld.interpolate_on_ld_grid(np.array([9000.]), np.array([2.]), 0., ["missing.benchmark.passband"])
            except (FileNotFoundError, LimbDarkeningSourceError) as error:
                expected_error = LimbDarkeningSourceError if mode == "optimized" else FileNotFoundError
                check(type(error) is expected_error, f"{mode}: changed source lookup error category")
            else:
                raise AssertionError(f"{mode}: missing source did not raise a source lookup error")
    a, b = np.array([5500.1234567891]), np.array([5500.1234567892])
    check(legacy_array_key(a) == legacy_array_key(b), "Legacy keys must merge the original rounding interval")
    check(utils.float64_array_to_cache_key(a) != utils.float64_array_to_cache_key(b), "Exact keys must differ")


def prime_source(bands):
    # Shared, supported query. This initializes lazy transforms too; result LRUs stay empty.
    ld.interpolate_on_ld_grid(np.array([9000.]), np.array([2.]), 0., list(bands) + ["bolometric"])


def measure_curve(observer, bands, mode, state):
    clear()
    if state == "warm_source_new_result":
        prime_source(bands)
    elif state == "repeated_curve":
        observer.observe.lc(**PHASES, normalize=True)
    before = stats(mode)
    start = perf_counter()
    observer.observe.lc(**PHASES, normalize=True)
    seconds = perf_counter() - start
    after = stats(mode)
    changes = delta(before, after)
    if mode == "optimized":
        expected = len(bands) + 1 if state == "cold_source" else 0
        check(changes["source"]["misses"] == expected, f"Unexpected source state: {state}, {changes}")
    if state == "warm_source_new_result":
        check(changes["result"]["hits"] == 0, "New-result sample unexpectedly reused evaluated results")
    if observer.system_cls is SingleSystem:
        expected_result = {"hits": 2, "misses": 0} if state == "repeated_curve" else {"hits": 0, "misses": 2}
        check(changes["result"] == expected_result, f"Single-star timing LRU state: {state}, {changes}")
        if mode == "optimized":
            expected_hits = len(bands) + 1 if state == "warm_source_new_result" else 0
            check(changes["source"]["hits"] == expected_hits, f"Single-star source reuse: {state}, {changes}")
    return dict(seconds=seconds, before=before, after=after, delta=changes)


def balanced_orders(modes):
    return list(permutations(modes))


def capture_science(observer, bands, mode):
    """Capture every scientific stage, including result-cache hits; count SciPy work separately."""
    arrays, counts = {}, dict(constructions=0, evaluations=0, query_points=0, reflection=0, preparations=0,
                             component_groups=0)
    raw_interpolator = ld.interpolate.LinearNDInterpolator
    reflection = ld.get_bolometric_ld_coefficients
    is_single = observer.system_cls is SingleSystem
    surface_utils = single_utils if is_single else binary_utils
    component_names = ("star",) if is_single else ("primary", "secondary")
    preparation = surface_utils.prep_surface_params
    components = surface_utils.get_limbdarkening_cfs

    def construct(*args, **kwargs):
        counts["constructions"] += 1
        interpolator = raw_interpolator(*args, **kwargs)
        def evaluate(query, *args, **kwargs):
            counts["evaluations"] += 1
            counts["query_points"] += len(query)
            return interpolator(query, *args, **kwargs)
        return evaluate

    def capture_reflection(temperature, log_g, metallicity, custom_ld_coefs=None):
        index = counts["reflection"]
        counts["reflection"] += 1
        label = f"reflection_{index:03d}"
        arrays[f"{label}_temperature"] = np.asarray(temperature).copy()
        arrays[f"{label}_logg"] = np.asarray(log_g).copy()
        result = reflection(temperature, log_g, metallicity, custom_ld_coefs)
        arrays[f"{label}_coeff"] = result.copy()  # Native (C,Q), pre-reflection stage.
        return result

    def capture_components(system, *args, **kwargs):
        index = counts["preparations"]
        requested = list(kwargs["passband"])
        stage = "observed" if requested == list(bands) else "bolometric_correction"
        for component in component_names:
            star = getattr(system, component)
            label = f"surface_{index:03d}_{stage}_{component}"
            arrays[f"{label}_temperature"] = star.temperatures.copy()
            arrays[f"{label}_logg"] = star.log_g.copy()
            if is_single:
                check(np.ptp(star.temperatures) > 1., "Single-star gravity darkening is not exercised")
                check(np.ptp(star.log_g) > .001, "Single-star gravity variation is not exercised")
            t, g = curve_utils.generate_teff_logg_for_ld_cfs(star, symmetry_test=True)
            arrays[f"{label}_query_t"] = np.asarray(legacy_array_key(t) if mode == "baseline" else t).copy()
            arrays[f"{label}_query_g"] = np.asarray(legacy_array_key(g) if mode == "baseline" else g).copy()
        result = components(system, *args, **kwargs)
        counts["component_groups"] += len(result)
        for component, values in result.items():
            for band, coefficients in values.items():
                check(coefficients.shape[0] == len(getattr(system, component).temperatures),
                      "Coefficients must cover every surface face")
                if is_single:
                    check(np.ptp(coefficients[:, 0]) > 0., "Single-star per-face LD is not exercised")
                arrays[f"surface_{index:03d}_{stage}_{component}_{band}_coeff"] = coefficients.copy()
        return result

    def capture_preparation(system, *args, **kwargs):
        # Single positions deliberately contain NaN distance/true_anomaly placeholders.
        # Only index, azimuth and phase are meaningful for rotational geometry.
        position = system.position
        geometry = (position.idx, position.azimuth, position.phase) if is_single else position
        arrays[f"geometry_{counts['preparations']:03d}"] = np.asarray(geometry).copy()
        result = preparation(system, *args, **kwargs)
        counts["preparations"] += 1
        return result

    clear()
    before = stats(mode)
    with patch.object(ld.interpolate, "LinearNDInterpolator", construct), \
         patch.object(_ndgriddata, "LinearNDInterpolator", construct), \
         patch.object(ld, "get_bolometric_ld_coefficients", capture_reflection), \
         patch.object(surface_utils, "get_limbdarkening_cfs", capture_components), \
         patch.object(surface_utils, "prep_surface_params", capture_preparation):
        phases, raw = observer.observe.lc(**PHASES, normalize=False, flux_unit=u.W/u.m**2)
        arrays["phases"] = phases.copy()
        for band, values in raw.items():
            arrays[f"raw_flux_{band}"] = values.copy()
        raw_stats, raw_counts = stats(mode), counts.copy()
        # Normalized capture is flux only, but work counts still distinguish its cache hits/misses.
        arrays_raw = {key: value.copy() for key, value in arrays.items()}
        arrays.clear()
        counts.update({key: 0 for key in counts})
        normalized_phases, normalized = observer.observe.lc(**PHASES, normalize=True)
        final_stats = stats(mode)
        check(np.array_equal(phases, normalized_phases), "Raw/normalized phase mismatch")
        intrinsic_keys = {key for key in arrays_raw if key != "phases" and not key.startswith("raw_flux_")}
        check(set(arrays) == intrinsic_keys, "Raw/normalized intrinsic stages differ")
        for key, value in arrays.items():
            check(np.array_equal(arrays_raw[key], value), f"Raw/normalized intrinsic stage mismatch: {key}")
        for band, values in normalized.items():
            arrays_raw[f"normalized_flux_{band}"] = values.copy()
    normalized_counts = counts.copy()
    for label, snapshot, changes in (("raw", raw_counts, delta(before, raw_stats)),
                                      ("normalized", normalized_counts, delta(raw_stats, final_stats))):
        check(snapshot["preparations"] > 0, f"{label}: surface instrumentation missed all preparations")
        reflection_count = 0 if is_single else 2 * snapshot["preparations"]
        check(snapshot["reflection"] == reflection_count, f"{label}: reflection stage count")
        check(snapshot["component_groups"] == 2 * len(component_names) * snapshot["preparations"],
              f"{label}: component stage count")
        if is_single:
            check(snapshot["preparations"] == 1, f"{label}: non-pulsating surface must be prepared once")
            expected_result = {"hits": 0, "misses": 2} if label == "raw" else {"hits": 2, "misses": 0}
            check(changes["result"] == expected_result, f"{label}: single-star LRU reuse: {changes}")
        misses = changes["result"]["misses"]
        # Fixed models request observed bands and one bolometric group per component.
        expected_evaluations = snapshot["reflection"] + misses * (len(bands) + 1) // 2
        check(snapshot["evaluations"] == expected_evaluations, f"{label}: actual SciPy evaluation count")
        expected_builds = changes["source"]["misses"] if mode == "optimized" else expected_evaluations
        check(snapshot["constructions"] == expected_builds, f"{label}: actual SciPy construction count")
        check(changes["result"]["hits"] + misses == snapshot["component_groups"], f"{label}: LRU accounting")
    return arrays_raw, dict(raw=dict(counts=raw_counts, cache_delta=delta(before, raw_stats)),
                            normalized=dict(counts=normalized_counts, cache_delta=delta(raw_stats, final_stats)))


def compare_arrays(reference, optimized, mode):
    check(set(reference) == set(optimized), f"{mode}: stage/array keys differ")
    differences = {}
    for key in reference:
        a, b = reference[key], optimized[key]
        check(a.shape == b.shape and a.dtype == b.dtype, f"{mode}: shape/dtype mismatch at {key}")
        check(np.isfinite(a).all() and np.isfinite(b).all(), f"{mode}: invalid values at {key}")
        if mode == "direct" or key == "phases" or key.startswith("geometry_") or key.endswith(("_temperature", "_logg")):
            check(np.array_equal(a, b), f"{mode}: unexpected scientific difference at {key}")
        category = ("coefficients" if key.endswith("_coeff") else "temperatures" if key.endswith("_temperature")
                    else "surface_gravity" if key.endswith("_logg")
                    else "raw_flux" if key.startswith("raw_flux_") else "normalized_flux"
                    if key.startswith("normalized_flux_") else "coordinates")
        diff = np.abs(a - b)
        entry = differences.setdefault(category, dict(max_absolute=0., max_relative=0.))
        entry["max_absolute"] = max(entry["max_absolute"], float(diff.max(initial=0)))
        entry["max_relative"] = max(entry["max_relative"],
                                     float((diff / np.maximum(np.abs(a), np.finfo(float).tiny)).max(initial=0)))
    return differences


def load_reference(directory, case, mode, expected_metadata):
    summary = json.loads((directory / "benchmark.json").read_text())
    check(summary.get("completed") is True, f"{directory}: reference run did not complete validation")
    path = directory / f"{mode}_{case}.npz"
    with np.load(path, allow_pickle=False) as archive:
        check("metadata_json" in archive, f"{path}: missing versioned metadata; regenerate reference")
        saved = json.loads(str(archive["metadata_json"]))
        check(saved.pop("mode", None) == mode and saved.pop("case", None) == case,
              f"{path}: wrong mode/model reference")
        fingerprint = saved.pop("fingerprint", None)
        check(fingerprint == digest(saved) == expected_metadata["fingerprint"], f"{path}: incompatible metadata")
        return {key: archive[key].copy() for key in archive.files if key != "metadata_json"}


def microbenchmark(modes):
    """Balanced same-call-site component requests: cold, new queries, exact result hits."""
    from types import SimpleNamespace
    orders = balanced_orders(modes)
    records = []
    for round_index in range(REPETITIONS):
        for state in ("cold_source", "warm_source_new_result", "warm_result_cache"):
            for mode in orders[round_index % len(orders)]:
                with implementation(mode):
                    clear()
                    temperatures = np.linspace(6900., 10100., 400)
                    component = SimpleNamespace(temperatures=temperatures, log_g=np.linspace(1.85, 2.25, 400),
                                                metallicity=0., limb_darkening_coefficients=None)
                    helper = curve_utils.get_component_limbdarkening_cfs
                    if state != "cold_source":
                        helper(component, ["Generic.Bessell.U"], symmetry_test=False)
                    if state == "warm_source_new_result":
                        component.temperatures = temperatures + .125
                    before = stats(mode)
                    start = perf_counter()
                    helper(component, ["Generic.Bessell.U"], symmetry_test=False)
                    seconds = perf_counter() - start
                    change = delta(before, stats(mode))
                    expected = {"hits": 1, "misses": 0} if state == "warm_result_cache" else {"hits": 0, "misses": 1}
                    check(change["result"] == expected, f"Micro cache state {state}: {change}")
                    records.append(dict(round=round_index, state=state, mode=mode, seconds=seconds, delta=change))
    return records


def main():
    check(MODE in ("all", *MODES), "Invalid MODE")
    modes = MODES if MODE == "all" else (MODE,)
    check(REPETITIONS > 0, "REPETITIONS must be positive")
    check(settings.NUMBER_OF_PROCESSES == 1 and settings.LIMB_DARKENING_LAW == "cosine", "Use one worker/cosine")
    check(not settings.USE_SINGLE_LD_COEFFICIENTS and settings.REFLECTION_EFFECT
          and settings.REFLECTION_EFFECT_ITERATIONS == 2, "Use per-face LD and two reflection iterations")
    check(COMPARE_DIR is None or ("optimized" in modes and not MICRO_ONLY), "Comparison needs optimized full curves")
    orders = balanced_orders(modes)
    if REPETITIONS % len(orders):
        warnings.warn(f"Use a multiple of {len(orders)} rounds for complete mode-order balance.", stacklevel=2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The engine's first curve can change NumPy's error policy while finding
    # Lagrange points. Stabilize it before recording/validating the environment;
    # all modes then start under the same policy. This is outside every timer.
    if not MICRO_ONLY:
        first_case = CASES[0]
        warmup = Observer(passband=["Generic.Bessell.U"], system=model(first_case))
        with implementation(modes[0]):
            warmup.observe.lc(**PHASES, normalize=True)
    environment = metadata()
    validate_errors(modes)
    if MICRO_ONLY:
        records = microbenchmark(modes)
        check(metadata() == environment, "Inputs changed during benchmark")
        (OUTPUT_DIR / "micro.json").write_text(json.dumps(dict(metadata=environment, completed=True, samples=records), indent=2))
        return
    report = dict(metadata=environment, completed=False, repetitions=REPETITIONS, orders=orders,
                  order_balanced=REPETITIONS % len(orders) == 0, cases={})
    for case in CASES:
        bands = environment["bands"][case]
        # One identical observer per mode; construction is outside all timers.
        observers = {mode: Observer(passband=bands, system=model(case)) for mode in modes}
        # Warm imports/Numba/other model caches equally, before source-state setup.
        for mode in modes:
            with implementation(mode):
                observers[mode].observe.lc(**PHASES, normalize=True)
        samples = []
        if not SCIENCE_ONLY:
            for round_index in range(REPETITIONS):
                states = STATES[round_index % len(STATES):] + STATES[:round_index % len(STATES)]
                for state in states:
                    order = orders[round_index % len(orders)]
                    for position, mode in enumerate(order):
                        with implementation(mode):
                            sample = measure_curve(observers[mode], bands, mode, state)
                        sample.update(round=round_index, state=state, mode=mode, position=position)
                        samples.append(sample)
                        print(case, state, mode, round(sample["seconds"], 6), flush=True)
        captured, counts = {}, {}
        for mode in modes:
            with implementation(mode):
                arrays, counts[mode] = capture_science(observers[mode], bands, mode)
                captured[mode] = arrays
                np.savez(OUTPUT_DIR / f"{mode}_{case}.npz", metadata_json=json.dumps(dict(environment, mode=mode, case=case), sort_keys=True), **arrays)
                if PROFILE:
                    clear()
                    observers[mode].observe.lc(**PHASES, normalize=True)
                    profiler = cProfile.Profile()
                    profiler.runcall(observers[mode].observe.lc, **PHASES, normalize=True)
                    profiler.dump_stats(str(OUTPUT_DIR / f"{mode}_{case}.prof"))
        comparisons = {}
        if "optimized" in captured:
            for mode in ("baseline", "direct"):
                reference = load_reference(COMPARE_DIR, case, mode, environment) if COMPARE_DIR else captured.get(mode)
                if reference is not None:
                    comparisons[mode] = compare_arrays(reference, captured["optimized"], mode)
        report["cases"][case] = dict(samples=samples, work_counts=counts, comparisons=comparisons)
        (OUTPUT_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))
        print(case, "scientific checks passed", counts, flush=True)
    check(metadata() == environment, "Inputs/settings changed during benchmark")
    report["completed"] = True
    (OUTPUT_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))
    print("Benchmark completed; benchmark.json contains raw samples, counts and comparisons.", flush=True)


if __name__ == "__main__":
    main()
