"""Persistent scientific caches follow effective inputs and source revisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from elisa import settings, utils
from elisa import units as u
from elisa.atm import atm
from elisa.base.curves import utils as curves
from elisa.binary_system.surface import mesh as binary_mesh
from elisa.binary_system.surface import temperature
from elisa.observer import observer as observer_module
from elisa.observer.observer import Observer
from elisa.observer.passband import init_rv_passband
from elisa.single_system.surface import mesh as single_mesh

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray


@pytest.fixture(autouse=True)
def isolate_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Local fixture only: no configuration-scope invalidation or runner changes.
    # noinspection PyProtectedMember
    caches = (
        curves._get_rv_passband,
        binary_mesh._load_correction_factors,
        temperature._load_albedo_data,
        single_mesh._load_correction_factors,
    )
    for cache in caches:
        # noinspection PyUnresolvedReferences
        cache.cache_clear()
    monkeypatch.setattr(type(atm.buffer), "ATMOSPHERE_TABLES", {})
    yield
    for cache in caches:
        # noinspection PyUnresolvedReferences
        cache.cache_clear()


def replace_file(path: Path, write: Callable[..., None] | Callable[..., int]) -> None:
    """Replace with equal-sized content and restored mtime to test inode/ctime keys."""
    old = path.stat()
    replacement = path.with_name("replacement-" + path.name)
    write(replacement)
    assert replacement.stat().st_size == old.st_size
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    replacement.replace(path)


def test_rv_interval_round_trip_precision_hits_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    first = (5500.0, 5600.0)
    second = (5500.0, np.nextafter(5600.0, np.inf))
    retained = []
    with patch.object(curves, "init_rv_passband", wraps=init_rv_passband) as load:
        for interval in (first, second, first):
            monkeypatch.setattr(settings, "RV_LAMBDA_INTERVAL", interval)
            result = curves._get_rv_passband()
            retained.append(result)
            assert result is curves._get_rv_passband()
            expected = init_rv_passband()
            assert result[1:] == expected[1:]
            pd.testing.assert_frame_equal(result[0].table, expected[0].table)
        assert load.call_count == 3  # One entry: the first A was evicted by B.
    assert retained[1][1] != retained[0][1]
    assert retained[0][1:] == retained[2][1:]
    info = curves._get_rv_passband.cache_info()
    assert (info.maxsize, info.currsize, info.hits, info.misses) == (1, 1, 3, 3)
    curves._get_rv_passband.cache_clear()
    assert curves._get_rv_passband.cache_info().currsize == 0


def test_rv_passband_schema_participates_before_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = settings.PASSBAND_DATAFRAME_WAVE
    for column in (initial, "alternate_wavelength", initial):
        monkeypatch.setattr(type(settings), "PASSBAND_DATAFRAME_WAVE", column)
        assert column in curves._get_rv_passband()[0].table


@pytest.mark.parametrize(
    ("loader", "field", "maxsize"),
    [
        (temperature._load_albedo_data, "PATH_TO_ALBEDOS", None),
        (binary_mesh._load_correction_factors, "PATH_TO_DETACHED_CORRECTIONS", None),
        (single_mesh._load_correction_factors, "PATH_TO_SINGLE_CORRECTIONS", 1),
    ],
)
def test_resource_round_trip_revisions_hits_and_uncached_equivalence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Callable[..., tuple[list[float], list[float]] | tuple[list[int], list[float]] | None]
    | Callable[..., dict[str, NDArray[np.float64]]]
    | Callable[..., NDArray[np.float64]],
    field: str,
    maxsize: int | None,
) -> None:
    is_albedo = field == "PATH_TO_ALBEDOS"

    def write(path: Path, value: float) -> None:
        if is_albedo:
            path.write_text(json.dumps({"x": [3.0, 4.0], "y": [value, value]}))
        else:
            np.save(path, np.array([[1.0, value], [2.0, value]]))

    suffix = ".json" if is_albedo else ".npy"
    a, b = tmp_path / ("a" + suffix), tmp_path / ("b" + suffix)
    write(a, 0.2)
    write(b, 0.7)
    if field == "PATH_TO_DETACHED_CORRECTIONS":
        monkeypatch.setattr(type(settings), "PATH_TO_OVER_CONTACT_CORRECTIONS", a)
    old = None
    for path in (a, b, a):
        monkeypatch.setattr(type(settings), field, path)
        result = loader()
        assert loader() is result  # Preserve borrowed cached data ownership.
        if is_albedo:
            expected = json.loads(path.read_text())
            assert result == (expected["x"], expected["y"])
            assert temperature.interpolate_albedo(5000.0) == np.interp(np.log10(5000.0), *result)
        elif field == "PATH_TO_DETACHED_CORRECTIONS":
            np.testing.assert_array_equal(result["detached"], np.load(path, allow_pickle=False))
            assert result["detached"] is result["semi-detached"] is result["double-contact"]
            np.testing.assert_array_equal(result["over-contact"], np.load(a, allow_pickle=False))
        else:
            np.testing.assert_array_equal(result, np.load(path, allow_pickle=False))
        if old is None:
            old = result
    info = loader.cache_info()
    assert info.maxsize == maxsize
    assert info.currsize == (2 if maxsize is None else 1)
    if maxsize is None:
        assert result is old
    identity = utils.file_source_identity(a)
    replace_file(a, lambda path: write(path, 0.9))
    assert utils.file_source_identity(a) != identity
    updated = loader()
    assert updated is not result
    if is_albedo:
        assert updated[1] == [0.9, 0.9]
        assert result[1] == [0.2, 0.2]
    else:
        arr = updated["detached"] if isinstance(updated, dict) else updated
        np.testing.assert_array_equal(arr, np.load(a, allow_pickle=False))
    assert loader.cache_parameters()["maxsize"] == maxsize


def test_overcontact_resource_has_its_own_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    detached, other = tmp_path / "detached.npy", tmp_path / "other.npy"
    np.save(detached, np.ones((2, 2)))
    np.save(other, np.full((2, 2), 2.0))
    monkeypatch.setattr(type(settings), "PATH_TO_DETACHED_CORRECTIONS", detached)
    for path, value in ((detached, 1.0), (other, 2.0), (detached, 1.0)):
        monkeypatch.setattr(type(settings), "PATH_TO_OVER_CONTACT_CORRECTIONS", path)
        result = binary_mesh._load_correction_factors()
        np.testing.assert_array_equal(result["over-contact"], np.full((2, 2), value))
        np.testing.assert_array_equal(result["detached"], np.ones((2, 2)))


def test_single_mesh_consumes_selected_resource(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in (("a", 1.0), ("b", 2.0), ("a", 1.0)):
        path = tmp_path / (name + ".npy")
        if not path.exists():
            np.save(path, np.full((2, 2), value))
        monkeypatch.setattr(type(settings), "PATH_TO_SINGLE_CORRECTIONS", path)
        with patch.object(single_mesh, "correct_component_mesh") as correct:
            single_mesh.correct_mesh(SimpleNamespace(star=object()))
        np.testing.assert_array_equal(correct.call_args.kwargs["correction_factors"], np.full((2, 2), value))


def write_atmosphere(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wave": [1000.0, 2000.0, 3000.0], "flux": [value] * 3, "other_flux": [value + 1.0] * 3}).to_csv(
        path,
        index=False,
    )


def test_atmosphere_relative_paths_revisions_aliases_and_numeric_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = "ckp00_5000_g40.csv"
    for name, value in (("a", 1.0), ("b", 2.0)):
        write_atmosphere(tmp_path / name / filename, value)
    models = []
    with patch.object(
        atm,
        "multithread_atm_tables_reader_runner",
        wraps=atm.multithread_atm_tables_reader_runner,
    ) as read:
        for name, _value in (("a", 1.0), ("b", 2.0), ("a", 1.0)):
            monkeypatch.chdir(tmp_path / name)
            loaded, mapping = atm.read_unique_atm_tables([filename, filename])
            model = loaded[0]
            models.append(model)
            assert model.fpath == filename
            assert mapping == {filename: [0, 1]}
            np.testing.assert_array_equal(model.model.flux, pd.read_csv(filename)["flux"].to_numpy())
            assert atm.read_unique_atm_tables([filename])[0][0] is model
        assert read.call_count == 2
        assert models[0] is models[2]
        absolute = str((tmp_path / "a" / filename).resolve())
        alias, mapping = atm.read_unique_atm_tables([absolute])
        assert alias[0].fpath == absolute
        assert alias[0].model is models[0].model
        assert atm.remap_unique_atm_container_to_origin(alias, mapping)[0] is alias[0]
        assert read.call_count == 2
        old_identity = utils.file_source_identity(filename)
        replace_file(Path(filename), lambda path: write_atmosphere(path, 3.0))
        assert utils.file_source_identity(filename) != old_identity
        new = atm.read_unique_atm_tables([filename])[0][0]
        np.testing.assert_array_equal(new.model.flux, [3.0] * 3)
        assert read.call_count == 3
        np.testing.assert_array_equal(models[0].model.flux, [1.0] * 3)


def test_atmosphere_column_interpretation_and_source_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ckp00_5000_g40.csv"
    write_atmosphere(path, 1.0)
    for column, value in (("flux", 1.0), ("other_flux", 2.0), ("flux", 1.0)):
        monkeypatch.setattr(type(settings), "ATM_MODEL_DATAFRAME_FLUX", column)
        model = atm.read_unique_atm_tables([str(path)])[0][0]
        np.testing.assert_array_equal(model.model.flux, [value] * 3)
    path.unlink()
    with pytest.raises(atm.ElisaAtmosphereError, match="doesn't exist"):
        atm.read_unique_atm_tables([str(path)])


@pytest.mark.usefixtures("monkeypatch")
def test_atmosphere_out_of_order_reader_keeps_keys_with_their_models(
    tmp_path: Path,
) -> None:
    paths = [str(tmp_path / f"ckp00_{t}_g40.csv") for t in (5000, 6000)]
    for path, value in zip(paths, (1.0, 2.0), strict=True):
        write_atmosphere(Path(path), value)

    def reversed_reader(submitted: list[str]) -> Queue:
        queue = Queue()
        for index, path in reversed(list(enumerate(submitted))):
            t, g, m = atm.parse_domain_quantities_from_atm_table_filename(Path(path).name)
            queue.put((index, atm.AtmDataContainer(pd.read_csv(path), t, g, m, fpath=path)))
        return queue

    with patch.object(atm, "multithread_atm_tables_reader_runner", side_effect=reversed_reader) as read:
        atm.read_unique_atm_tables(paths)
        for path, value in zip(paths, (1.0, 2.0), strict=True):
            model = atm.read_unique_atm_tables([path])[0][0]
            assert model.fpath == path
            np.testing.assert_array_equal(model.model.flux, [value] * 3)
        assert read.call_count == 1


def test_atmosphere_fifo_capacity_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(type(atm.buffer), "MAX_STORAGE", 2)
    paths = [tmp_path / f"ckp00_{t}_g40.csv" for t in (5000, 6000, 7000)]
    for path in paths:
        write_atmosphere(path, 1.0)
    for path in paths[:2]:
        atm.read_unique_atm_tables([str(path)])
    atm.read_unique_atm_tables([str(paths[0])])  # Hits do not turn the FIFO into an LRU.
    atm.read_unique_atm_tables([str(paths[2])])
    assert len(atm.buffer.ATMOSPHERE_TABLES) == 2
    assert utils.file_source_identity(paths[0]) not in {key[0] for key in atm.buffer.ATMOSPHERE_TABLES}


def write_standard(root: Path, system: str, value: float) -> Path:
    path = root / "zero_points" / f"{system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"system": system, "fluxes": {"bolometric": value}, "reference_magnitudes": {"bolometric": 0.0}}),
    )
    return path


def test_observer_zero_points_source_round_trip_replacement_and_actual_magnitudes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for root, value in ((a, 1.0), (b, 2.0)):
        write_standard(root, "vega", value)
    system = SimpleNamespace(
        additional_light=0.0,
        distance=1.0,
        phase_shift=0.0,
        get_positions_method=lambda: None,
        compute_lightcurve=lambda **_kw: {"bolometric": np.array([1.0, 2.0])},
    )
    observer = Observer(passband="bolometric", system=system)
    phases = np.array([0.0, 0.1])
    monkeypatch.setattr(observer, "manage_time_series", lambda *_args: phases)
    monkeypatch.setattr(observer, "phase_interval_reduce", lambda *_args: (phases, np.array([0, 1])))
    monkeypatch.setattr(settings, "MAGNITUDE_SYSTEM", "vega")
    with patch.object(observer_module, "load_standard", wraps=observer_module.load_standard) as load:
        previous = []
        for root, value in ((a, 1.0), (b, 2.0), (a, 1.0)):
            monkeypatch.setattr(type(settings), "DATA_PATH", root)
            result = observer.lc(phases=phases, flux_unit=u.mag)[1]["bolometric"]
            expected = -2.5 * np.log10(np.array([1.0, 2.0]) / value)
            np.testing.assert_array_equal(result, expected)
            calibration = observer.zero_points
            assert observer._get_zero_points() is calibration
            previous.append(calibration)
        assert load.call_count == 3  # Original per-observer capacity is one.
        path = a / "zero_points" / "vega.json"
        replace_file(path, lambda replacement: replacement.write_text(path.read_text().replace("1.0", "9.0")))
        assert observer._get_zero_points()["fluxes"]["bolometric"] == 9.0
        assert previous[0]["fluxes"]["bolometric"] == 1.0
        assert load.call_count == 4
        updated_magnitudes = observer.lc(phases=phases, flux_unit=u.mag)[1]["bolometric"]
        np.testing.assert_array_equal(updated_magnitudes, -2.5 * np.log10(np.array([1.0, 2.0]) / 9.0))
        write_standard(a, "ab", 3.0)
        monkeypatch.setattr(settings, "MAGNITUDE_SYSTEM", "AB")
        assert observer._get_zero_points()["system"] == "ab"
        assert load.call_count == 5


def test_source_identity_aliases_and_in_place_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "albedo.json"
    path.write_text('{"x": [3, 4], "y": [0.2, 0.2]}')
    monkeypatch.setattr(type(settings), "PATH_TO_ALBEDOS", path)
    first = temperature._load_albedo_data()
    identity = utils.file_source_identity(path)
    assert utils.file_source_identity(path) == identity
    monkeypatch.setattr(type(settings), "PATH_TO_ALBEDOS", str(path.parent / "." / path.name))
    assert temperature._load_albedo_data() is first
    previous = path.stat()
    path.write_text(path.read_text().replace("0.2", "0.7"))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert utils.file_source_identity(path) != identity  # ctime detects the rewrite.
    assert temperature._load_albedo_data()[1] == [0.7, 0.7]


@pytest.mark.parametrize("kind", ["single", "binary"])
def test_actual_mesh_correction_round_trip_matches_uncached_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    paths = [tmp_path / "a.npy", tmp_path / "b.npy"]
    for path, value in zip(paths, (0.7, 0.9), strict=True):
        np.save(path, np.array([[0.01, 0.2], [value, value]]))
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    monkeypatch.setattr(type(settings), "PATH_TO_OVER_CONTACT_CORRECTIONS", paths[0])
    results = []
    for path in (paths[0], paths[1], paths[0]):
        field = "PATH_TO_SINGLE_CORRECTIONS" if kind == "single" else "PATH_TO_DETACHED_CORRECTIONS"
        monkeypatch.setattr(type(settings), field, path)
        star = SimpleNamespace(points=points.copy(), discretization_factor=0.1, has_spots=lambda: False)
        if kind == "single":
            single_mesh.correct_mesh(SimpleNamespace(star=star))
        else:
            binary_mesh.correct_mesh(SimpleNamespace(primary=star, morphology="detached"), 1.0, component="primary")
        expected_factor = utils.discretization_correction_factor(0.1, np.load(path, allow_pickle=False))
        np.testing.assert_array_equal(star.points, expected_factor * points)
        results.append(star.points.copy())
    np.testing.assert_array_equal(results[0], results[2])
    assert not np.array_equal(results[0], results[1])


def test_rv_interval_override_restores_results_and_retained_object() -> None:
    with settings.override(RV_LAMBDA_INTERVAL=(5500.0, 5600.0)):
        original = curves._get_rv_passband()
        with settings.override(RV_LAMBDA_INTERVAL=(6000.0, 6200.0)):
            changed = curves._get_rv_passband()
            expected = init_rv_passband()
            assert changed is curves._get_rv_passband()
            assert changed[1:] == expected[1:]
            pd.testing.assert_frame_equal(changed[0].table, expected[0].table)
        restored = curves._get_rv_passband()
        assert restored[1:] == original[1:]
        pd.testing.assert_frame_equal(restored[0].table, original[0].table)
        # A retained B object keeps the interval with which it was constructed.
        assert changed[0].get_bandwidth() == (6000.0, 6200.0)
        assert original[0].get_bandwidth() == (5500.0, 5600.0)
