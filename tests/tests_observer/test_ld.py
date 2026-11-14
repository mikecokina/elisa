from __future__ import annotations

# The tests package selects Astropy constants before test-module imports.
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.errors
import pytest
from numpy.testing import assert_array_equal
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from elisa import const, ld, settings, utils
from elisa.base.curves import utils as curve_utils
from elisa.base.error import ElisaLimbDarkeningError, ElisaLookupError, ElisaUnexpectedError
from tests import set_astropy_units
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

    import elisa.base.error
    import elisa.binary_system.container
    import elisa.observer.observer


set_astropy_units()


class TestLimbDarkeningModule(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base_path = str(Path(__file__).resolve().parent.parent)

    def test_get_metallicity_from_ld_table_filename(self) -> None:
        filenames = [
            "lin.bolometric.m02.csv",
            "lin.bolometric.p00.csv",
            "lin.Generic.Bessell.B.p10.csv",
            str(Path("path.path.csv").joinpath("lin.Generic.Bessell.B.p10.csv")),
        ]
        expected = [-0.2, 0.0, 1.0, 1.0]
        obtained = [ld.get_metallicity_from_ld_table_filename(filename) for filename in filenames]
        assert_array_equal(expected, obtained)

    def test_get_ld_table_filename(self) -> None:
        expected = [
            "lin.Generic.Bessell.B.m01.csv",
            "lin.Generic.Stromgren.v.m50.csv",
            "lin.Kepler.p01.csv",
            "sqrt.Generic.Bessell.B.p10.csv",
        ]
        params = [
            {"passband": "Generic.Bessell.B", "metallicity": -0.1, "law": "linear"},
            {"passband": "Generic.Stromgren.v", "metallicity": -5, "law": "linear"},
            {"passband": "Kepler", "metallicity": 0.1, "law": "linear"},
            {"passband": "Generic.Bessell.B", "metallicity": 1.0, "law": "square_root"},
        ]
        obtained = [ld.get_ld_table_filename(**param) for param in params]
        assert_array_equal(expected, obtained)

    def test_get_relevant_ld_tables(self) -> None:
        params = [
            {"metallicity": 0.0, "passband": "Kepler", "law": "cosine"},
            {"metallicity": -0.01, "passband": "Generic.Bessell.B", "law": "linear"},
            {"metallicity": 0.02, "passband": "SLOAN.SDSS.i", "law": "logarithmic"},
            {"metallicity": 1.0, "passband": "SLOAN.SDSS.i", "law": "square_root"},
        ]
        obtained = [ld.get_relevant_ld_tables(**param) for param in params]
        expected = [
            ["lin.Kepler.p00.csv", "lin.Kepler.p00.csv"],
            ["lin.Generic.Bessell.B.m01.csv", "lin.Generic.Bessell.B.p00.csv"],
            ["log.SLOAN.SDSS.i.p00.csv", "log.SLOAN.SDSS.i.p01.csv"],
            ["sqrt.SLOAN.SDSS.i.p10.csv", "sqrt.SLOAN.SDSS.i.p10.csv"],
        ]
        assert_array_equal(obtained, expected)

    def test_interpolate_on_ld_grid_lin(self) -> None:
        settings.configure(LIMB_DARKENING_LAW="cosine", LD_TABLES=str(Path(self.base_path).joinpath("data", "ld_grid")))
        temperature = np.array([5500, 5561, 5582, 5932])
        log_g = np.array([4.2, 4.21, 4.223, 4.199]) - 2.0
        metallicity = -0.1
        passband = ["Generic.Bessell.B"]

        # Direct legacy griddata gives 0.708598 for the final query on this fixture.
        expected = np.array([0.7732, 0.76364, 0.76032, 0.70860])

        obtained = ld.interpolate_on_ld_grid(temperature, log_g, metallicity, passband)
        ld_column = settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW].index("xlin")
        obtained = np.round(obtained["Generic.Bessell.B"][:, ld_column], 5)

        np.testing.assert_allclose(obtained, expected, rtol=0, atol=1e-5)

    def test_interpolate_on_ld_grid_log(self) -> None:
        settings.configure(
            LIMB_DARKENING_LAW="logarithmic",
            LD_TABLES=str(Path(self.base_path).joinpath("data", "light_curves", "limbdarkening")),
        )
        temperature = np.array([5500, 5561, 5582, 5932])
        log_g = np.array([4.2, 4.21, 4.223, 4.199]) - 2.0
        metallicity = 0.0
        passband = ["Generic.Bessell.V"]

        expected_xlog = np.round(np.array([0.776400, 0.772272, 0.770870, 0.747420]), 5)
        expected_ylog = np.round(np.array([0.204200, 0.211724, 0.214276, 0.251288]), 5)

        obtained = ld.interpolate_on_ld_grid(temperature, log_g, metallicity, passband)
        ld_column_x = settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW].index("xlog")
        ld_column_y = settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW].index("ylog")
        obtained_xlog = np.round(obtained["Generic.Bessell.V"][:, ld_column_x], 5)
        obtained_ylog = np.round(obtained["Generic.Bessell.V"][:, ld_column_y], 5)

        assert np.all(expected_ylog == obtained_ylog)
        assert np.all(expected_xlog == obtained_xlog)

    def _raise_test_limb_darkening_factor(self, exception_type: type[Exception], msg: str, **kwargs) -> None:
        with pytest.raises(exception_type) as context:
            ld.limb_darkening_factor(**kwargs)
        assert msg in str(context.value)

    def test_limb_darkening_factor_raise(self) -> None:
        normals = np.array([[1, 1, 1], [0.5, 0.5, 0.5], [-1.0, -1, 1.3]])
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        valid = {
            "normal_vector": normals,
            "line_of_sight": const.LINE_OF_SIGHT,
            "coefficients": np.array([[0.1], [0.2], [0.3]]),
            "limb_darkening_law": "linear",
        }
        assert np.isfinite(ld.limb_darkening_factor(**valid)).all()

        # Omit exactly one prerequisite so validation order cannot choose the error.
        for missing, exception_type, diagnostic in (
            ("line_of_sight", ElisaUnexpectedError, "Line of sight"),
            ("normal_vector", ElisaUnexpectedError, "Normal vector"),
            ("coefficients", ElisaUnexpectedError, "coefficients"),
            ("limb_darkening_law", ElisaUnexpectedError, "rule"),
        ):
            with self.subTest(missing=missing):
                kwargs = valid.copy()
                del kwargs[missing]
                self._raise_test_limb_darkening_factor(exception_type, diagnostic, **kwargs)

    def test_limb_darkening_factor_linear(self) -> None:
        ld_law = "linear"
        coefficients = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
        cos_theta = np.array([[-0.23], [0.25], [0.85], [0.55], [0.33], [1.0]])

        expected = np.array([0.0, 0.85, 0.955, 0.82, 0.665, 1.0])
        obtained = ld.limb_darkening_factor(coefficients=coefficients, limb_darkening_law=ld_law, cos_theta=cos_theta)

        assert np.all(expected == np.round(obtained, 3))

    def test_limb_darkening_factor_logarithmic(self) -> None:
        ld_law = "logarithmic"
        coefficients = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.2, 0.3]])
        cos_theta = np.array([[-0.23], [0.25], [0.85], [0.55], [0.33], [1.0]])

        obtained = ld.limb_darkening_factor(coefficients=coefficients, limb_darkening_law=ld_law, cos_theta=cos_theta)

        expected = np.array([0.0, 1.127, 1.052, 1.017, 0.848, 1.0])
        assert np.all(expected == np.round(obtained, 3))


# Synthetic tables keep cache and boundary regressions independent of installed assets.


@pytest.fixture(name="ld_sources")
def synthetic_ld_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(settings, "LD_TABLES", str(tmp_path))
    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "cosine")
    monkeypatch.setattr(settings, "USE_SINGLE_LD_COEFFICIENTS", False)
    points = np.array([[5000.0, 3.0], [6000.0, 3.0], [5000.0, 4.0], [6000.0, 4.0]])
    for law in ("cosine", "logarithmic", "square_root"):
        for band in ("Generic.Bessell.U", "bolometric", "Generic.Bessell.V"):
            for metal in (0.0, 0.1):
                values = {
                    name: (i + 1) * (points[:, 0] / 10000 + points[:, 1] / 10) + metal
                    for i, name in enumerate(settings.LD_LAW_CFS_COLUMNS[law])
                }
                frame = pd.DataFrame(dict(temperature=points[:, 0], gravity=points[:, 1], **values))
                # Include an identical full-row duplicate; intermediate metals conflict at coordinates.
                pd.concat([frame, frame.iloc[:1]]).to_csv(
                    tmp_path / ld.get_ld_table_filename(band, metal, law),
                    index=False,
                )
    curve_utils.interpolate_on_ld_grid_cached.cache_clear()
    from elisa.buffer import buffer

    buffer.LD_CFS_TABLES.clear()
    yield tmp_path
    if hasattr(ld, "clear_ld_caches"):
        ld.clear_ld_caches()
    curve_utils.interpolate_on_ld_grid_cached.cache_clear()
    buffer.LD_CFS_TABLES.clear()


def legacy_source(band: str, metal: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cols = settings.LD_LAW_COLS_ORDER[settings.LIMB_DARKENING_LAW]
    frame = pd.concat(
        [pd.read_csv(Path(settings.LD_TABLES) / name)[cols] for name in ld.get_relevant_ld_tables(band, metal)],
        ignore_index=True,
    ).drop_duplicates()
    return (
        frame[settings.LD_DOMAIN_COLS].to_numpy(),
        frame[settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW]].to_numpy(),
    )


@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
@pytest.mark.parametrize("metal", [0.0, 0.05])
@pytest.mark.parametrize("band", ["Generic.Bessell.U", "bolometric", "Generic.Bessell.V"])
@pytest.mark.usefixtures("ld_sources")
def test_unrounded_griddata_equivalence(
    monkeypatch: pytest.MonkeyPatch,
    law: str,
    metal: float,
    band: str,
) -> None:
    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", law)
    points, values = legacy_source(band, metal)
    queries = np.array(
        [
            [5000.0, 3.0],
            [6000.0, 4.0],
            [5500.0, 3.0],
            [5000.0, 3.5],
            [5523.123456789123, 3.456789123456],
            [5500.0, 3.5],
            [5500.0, 3.5],
            [5000.000001, 3.5],
        ],
    )
    expected = griddata(points, values, queries, method="linear")
    actual = ld.interpolate_on_ld_grid(queries[:, 0], queries[:, 1] - 2, metal, [band])[band]
    assert_array_equal(actual, expected)
    if metal == 0.0:
        # Independent affine-field oracle (not merely another interpolation implementation).
        np.testing.assert_allclose(actual[:, 0], queries[:, 0] / 10000 + queries[:, 1] / 10, rtol=0, atol=4e-16)
    alias = ld.interpolate_on_ld_grid(queries[:, 0], queries[:, 1] - 2, metal, {"rv_band": None})
    assert alias["rv_band"].shape == (len(queries), values.shape[1])


@pytest.mark.parametrize(
    ("defect", "error"),
    [
        ("outside", ElisaLimbDarkeningError),
        ("query_nan", ElisaLimbDarkeningError),
        ("value_nan", ElisaLimbDarkeningError),
        ("domain_nan", ElisaLimbDarkeningError),
        ("degenerate", ElisaLimbDarkeningError),
        ("missing_column", ElisaUnexpectedError),
        ("missing_file", ElisaLimbDarkeningError),
    ],
)
def test_interpolation_errors(
    ld_sources: pathlib.PosixPath,
    defect: str,
    error: type[elisa.base.error.ElisaLimbDarkeningError | elisa.base.error.ElisaUnexpectedError],
) -> None:
    path = ld_sources / ld.get_ld_table_filename("bolometric", 0.0)
    frame = pd.read_csv(path)
    query = [5500.0, 1.5]
    if defect == "outside":
        query[0] = 4999.999999
    elif defect == "query_nan":
        query[0] = np.nan
    elif defect == "value_nan":
        frame["xlin"] = np.nan
    elif defect == "domain_nan":
        frame.loc[0, "temperature"] = np.nan
    elif defect == "degenerate":
        frame["gravity"] = 3.0
    elif defect == "missing_column":
        frame = frame.drop(columns="xlin")
    if defect == "missing_file":
        path.unlink()
    else:
        frame.to_csv(path, index=False)
    with pytest.raises(error) as caught:
        ld.interpolate_on_ld_grid(np.array([query[0]]), np.array([query[1]]), 0.0, ["bolometric"])
    if defect == "degenerate":
        assert isinstance(caught.value.__cause__, QhullError)
    elif defect == "domain_nan":
        assert isinstance(caught.value.__cause__, ValueError)


def test_interpolation_error_identifies_table_and_requested_values(ld_sources: Path) -> None:
    temperature = np.array([5500.0, 4999.999999, 7000.0])
    log_g_si = np.array([1.5, 1.5, 2.5])

    with pytest.raises(ElisaLimbDarkeningError) as raised:
        ld.interpolate_on_ld_grid(temperature, log_g_si, 0.0, ["bolometric"])

    message = str(raised.value)
    table_path = ld_sources / ld.get_ld_table_filename("bolometric", 0.0)
    assert str(table_path) in message
    assert "Passband: bolometric; law: cosine; metallicity: 0.0" in message
    assert "Failed interpolation points: 2 of 3" in message
    assert "input rows (zero-based): [1, 2]" in message
    assert "t_eff [K], log_g [log10(cm/s^2)]" in message
    assert "[[4999.999999, 3.5], [7000.0, 4.5]]" in message


def component(temperature: float = 5500.1234567891) -> SimpleNamespace:
    return SimpleNamespace(
        temperatures=np.array([temperature]),
        log_g=np.array([1.543210987654]),
        metallicity=0.0,
        limb_darkening_coefficients=None,
    )


def evaluate(comp: SimpleNamespace) -> NDArray[np.float64]:
    return curve_utils.get_component_limbdarkening_cfs(comp, ["bolometric"], symmetry_test=False)["bolometric"]


@pytest.mark.usefixtures("ld_sources")
def test_exact_keys_and_output_ownership() -> None:
    a, b = component(), component(5500.1234567892)
    with patch.object(ld, "interpolate_on_ld_grid", wraps=ld.interpolate_on_ld_grid) as call:
        first = evaluate(a)
        second = evaluate(b)
        assert call.call_count == 2
        assert_array_equal(call.call_args.kwargs["temperature"], b.temperatures)
        points, values = legacy_source("bolometric", 0.0)
        assert_array_equal(
            second,
            griddata(points, values, np.column_stack((b.temperatures, b.log_g + 2)), method="linear"),
        )
        c = component()
        c.log_g += 1e-10
        evaluate(c)
        assert call.call_count == 3
        assert_array_equal(call.call_args.kwargs["log_g"], c.log_g)
        first[:] = -999
        assert evaluate(a)[0, 0] > 0
        assert call.call_count == 3
    key = utils.float64_array_to_cache_key
    assert key(np.array([1], dtype="f4")) == key(np.array([1], dtype="f8"))
    assert key(np.array([1.0, 2.0])) != key(np.array([[1.0, 2.0]]))
    assert key(np.array([0.0])) != key(np.array([-0.0]))
    assert key(np.array([np.nan])) == key(np.array([np.nan]))


@pytest.mark.usefixtures("tmp_path")
def test_source_reuse_and_invalidation(
    ld_sources: pathlib.PosixPath,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with patch.object(ld.interpolate, "LinearNDInterpolator", wraps=ld.interpolate.LinearNDInterpolator) as build:
        original = evaluate(component()).copy()
        evaluate(component(5555.0))
        assert build.call_count == 1
        # A same-sized rewrite, even with restored mtime, must invalidate both layers.
        path = ld_sources / ld.get_ld_table_filename("bolometric", 0.0)
        stat = path.stat()
        frame = pd.read_csv(path)
        frame["xlin"] += 1
        frame.to_csv(path, index=False)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert_array_equal(evaluate(component()), original + 1)
        assert build.call_count == 2
        monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "logarithmic")
        assert evaluate(component()).shape == (1, 2)
        assert build.call_count == 3
        other = ld_sources / "other"
        other.mkdir()
        frame["xlin"] += 1
        frame.to_csv(other / ld.get_ld_table_filename("bolometric", 0.0, "cosine"), index=False)
        monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "cosine")
        monkeypatch.setattr(settings, "LD_TABLES", str(other))
        assert_array_equal(evaluate(component()), original + 2)
        assert build.call_count == 4
        ld.clear_ld_caches()
        evaluate(component())
        assert build.call_count == 5


@pytest.mark.usefixtures("ld_sources")
def test_result_lru_eviction() -> None:
    evaluate(component())
    for i in range(128):
        evaluate(component(5501.0 + i))
    before = curve_utils.interpolate_on_ld_grid_cached.cache_info()
    evaluate(component())
    after = curve_utils.interpolate_on_ld_grid_cached.cache_info()
    assert after.currsize == 128
    assert after.misses == before.misses + 1


def test_source_lru_eviction_and_schema(ld_sources: pathlib.PosixPath, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    for i in range(33):
        band = f"test{i}"
        shutil.copyfile(
            ld_sources / ld.get_ld_table_filename("bolometric", 0.0),
            ld_sources / ld.get_ld_table_filename(band, 0.0),
        )
        ld.interpolate_on_ld_grid(np.array([5500.0]), np.array([1.5]), 0.0, [band])
    info = ld.get_ld_source_interpolator.cache_info()
    assert info.currsize == 32
    assert info.misses == 33
    ld.interpolate_on_ld_grid(np.array([5500.0]), np.array([1.5]), 0.0, ["test0"])
    assert ld.get_ld_source_interpolator.cache_info().misses == 34
    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "logarithmic")
    first = evaluate(component())
    monkeypatch.setitem(settings.LD_LAW_CFS_COLUMNS, "logarithmic", ["ylog", "xlog"])
    assert_array_equal(evaluate(component()), first[:, ::-1])


@pytest.mark.usefixtures("ld_sources")
def test_empty_and_nan_payload_keys() -> None:
    key = utils.float64_array_to_cache_key
    assert key(np.array([1.0, 2.0], dtype=">f8")) == key(np.array([1.0, 2.0], dtype="<f8"))
    assert key(np.arange(6.0)[::2]) == key(np.array([0.0, 2.0, 4.0]))
    nan1 = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    nan2 = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)
    assert key(nan1) != key(nan2)
    for values in (np.array([[0.0, -0.0, 1.0]]), nan1, nan2, np.arange(6.0)[::2], np.array([])):
        decoded = utils.float64_array_from_cache_key(key(values))
        assert key(decoded) == key(values)
        assert decoded.shape == values.shape
        assert not decoded.flags.writeable
    result = ld.interpolate_on_ld_grid(np.array([]), np.array([]), 0.0, ["bolometric"])
    assert result["bolometric"].shape == (0, 1)
    with pytest.raises(ElisaLimbDarkeningError):
        ld.interpolate_on_ld_grid(np.array([5500.0]), np.array([1.5]), 99.0, ["bolometric"])


@pytest.mark.requires_ld(passbands=("Generic.Bessell.U", "bolometric"), laws=("cosine",))
def test_circular_complete_curve_lru(monkeypatch: pytest.MonkeyPatch) -> None:
    from elisa.binary_system.curves import c_managed

    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "cosine")
    monkeypatch.setattr(settings, "NUMBER_OF_PROCESSES", 1)
    monkeypatch.setattr(settings, "USE_SINGLE_LD_COEFFICIENTS", False)
    ld.clear_ld_caches()
    original_worker = c_managed.produce_circ_sync_curves_mp

    def worker(*worker_args, **worker_kwargs) -> dict[str, NDArray[np.float64]]:
        before = curve_utils.interpolate_on_ld_grid_cached.cache_info()
        result = original_worker(*worker_args, **worker_kwargs)
        assert curve_utils.interpolate_on_ld_grid_cached.cache_info() == before
        return result

    lc_kwargs = {"from_phase": -0.6, "to_phase": 0.6, "phase_step": 0.01, "normalize": True}
    try:
        with (
            patch.object(c_managed, "produce_circ_sync_curves_mp", worker),
            patch.object(ld.interpolate, "LinearNDInterpolator", wraps=ld.interpolate.LinearNDInterpolator) as build,
            patch.object(ld, "interpolate_on_ld_grid", wraps=ld.interpolate_on_ld_grid) as query,
        ):
            obs = audit_observer()
            first = obs.observe.lc(**lc_kwargs)
            # Only observed passbands enter this cache; reflection queries bolometric LD separately.
            assert curve_utils.interpolate_on_ld_grid_cached.cache_info()[:2] == (0, 2)
            assert query.call_count == 4
            assert build.call_count == 2
            repeated = obs.observe.lc(**lc_kwargs)
            assert curve_utils.interpolate_on_ld_grid_cached.cache_info()[:2] == (2, 2)
            assert query.call_count == 6
            assert build.call_count == 2
            equivalent = audit_observer().observe.lc(**lc_kwargs)
            assert curve_utils.interpolate_on_ld_grid_cached.cache_info()[:2] == (4, 2)
            assert query.call_count == 8
            assert build.call_count == 2
            for actual in (repeated, equivalent):
                assert_array_equal(first[0], actual[0])
                assert_array_equal(first[1]["Generic.Bessell.U"], actual[1]["Generic.Bessell.U"])
    finally:
        ld.clear_ld_caches()


@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
@pytest.mark.parametrize("metal", [-5.0, 0.0, 0.05, 1.0])
@pytest.mark.parametrize("band", ["Generic.Bessell.U", "bolometric", "Generic.Bessell.V"])
@pytest.mark.requires_ld(source="van_hamme")
def test_installed_asset_hull_parity(monkeypatch: pytest.MonkeyPatch, law: str, metal: float, band: str) -> None:
    from scipy.spatial import ConvexHull

    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", law)
    points, values = legacy_source(band, metal)
    hull = points[ConvexHull(points).vertices]
    center = hull.mean(axis=0)
    midpoints = (hull + np.roll(hull, -1, axis=0)) / 2
    # Hull nodes, edges, all source rows (including conflicting coordinates), interiors.
    queries = np.vstack((points, midpoints, center, center, midpoints + 1e-8 * (center - midpoints)))
    gravity = queries[:, 1] - 2
    expected = griddata(points, values, np.column_stack((queries[:, 0], gravity + 2)), method="linear")
    assert_array_equal(ld.interpolate_on_ld_grid(queries[:, 0], gravity, metal, [band])[band], expected)
    # An outward displacement may lie within Qhull's tolerance: follow the oracle's decision exactly.
    for query in (midpoints[0] - 1e-8 * (center - midpoints[0]), np.nextafter(hull[0], np.inf)):
        gravity = np.array([query[1] - 2])
        expected = griddata(points, values, np.column_stack(([query[0]], gravity + 2)), method="linear")
        if np.isnan(expected).any():
            with pytest.raises(ElisaLimbDarkeningError, match=r"produced numpy\.nan/None"):
                ld.interpolate_on_ld_grid(np.array([query[0]]), gravity, metal, [band])
        else:
            assert_array_equal(ld.interpolate_on_ld_grid(np.array([query[0]]), gravity, metal, [band])[band], expected)
    ld.clear_ld_caches()


def audit_observer(
    eccentricity: float = 0.0,
    bands: list[str] | tuple[str] = ("Generic.Bessell.U",),
) -> elisa.observer.observer.Observer:
    from astropy import units as u

    from elisa import BinarySystem, Observer, Star

    primary = Star(
        mass=2.15 * u.solMass,
        surface_potential=3.6,
        synchronicity=1.0,
        t_eff=10000 * u.K,
        gravity_darkening=1.0,
        discretization_factor=5,
        albedo=0.6,
        metallicity=0.0,
        atmosphere="bb",
    )
    secondary = Star(
        mass=1.2 * u.solMass,
        surface_potential=4.0,
        synchronicity=1.0,
        t_eff=7000 * u.K,
        gravity_darkening=1.0,
        albedo=0.6,
        metallicity=0.0,
        atmosphere="bb",
    )
    binary = BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=58 * u.deg,
        gamma=-30.7 * u.km / u.s,
        period=2.5 * u.d,
        eccentricity=eccentricity,
        inclination=85 * u.deg,
        distance=155 * u.pc,
        primary_minimum_time=2440000 * u.d,
        phase_shift=0.0,
    )
    return Observer(passband=list(bands), system=binary)


@pytest.mark.parametrize(
    ("eccentricity", "bands"),
    [(0.0, ["Generic.Bessell.U"]), (0.2, ["Generic.Bessell.U"]), (0.0, ["Generic.Bessell.U", "Generic.Bessell.V"])],
)
@pytest.mark.requires_ld(passbands=("bolometric",), laws=("cosine",))
def test_complete_scientific_unrounded_parity(
    monkeypatch: pytest.MonkeyPatch,
    eccentricity: float,
    bands: list[str],
) -> None:
    from astropy import units as u

    from elisa.binary_system.curves import utils as binary_curve_utils

    monkeypatch.setattr(settings, "LIMB_DARKENING_LAW", "cosine")
    monkeypatch.setattr(settings, "NUMBER_OF_PROCESSES", 1)
    monkeypatch.setattr(settings, "USE_SINGLE_LD_COEFFICIENTS", False)
    lc_kwargs = {"from_phase": -0.6, "to_phase": 0.6, "phase_step": 0.01}
    optimized = ld.interpolate_on_ld_grid
    original_prep = binary_curve_utils.prep_surface_params

    def oracle(
        temperature: NDArray[np.float64],
        log_g: NDArray[np.float64],
        metallicity: float,
        passband: list[str],
    ) -> dict[str, NDArray[np.float64]]:
        result = {}
        for oracle_band in passband:
            points, values = legacy_source(oracle_band, metallicity)
            result[oracle_band] = griddata(
                points,
                values,
                np.column_stack((temperature, np.asarray(log_g) + 2)),
                method="linear",
            )
            assert not np.isnan(result[oracle_band]).any()
        return result

    results = []
    try:
        for interpolate in (oracle, optimized):
            ld.clear_ld_caches()
            captured_coeffs, captured_temperatures = [], []

            def capture(
                *interpolation_args,
                interpolate: Callable[..., dict[str, NDArray[np.float64]] | None]
                | Callable[..., dict[str, NDArray[np.float64]]] = interpolate,
                captured_coeffs: list[NDArray[np.float64]] | list[object] = captured_coeffs,
                **interpolation_kwargs,
            ) -> dict[str, NDArray[np.float64]]:
                result = interpolate(*interpolation_args, **interpolation_kwargs)
                captured_coeffs.extend(value.copy() for value in result.values())
                return result

            def capture_prep(
                system: elisa.binary_system.container.OrbitalPositionContainer,
                captured_temperatures: list[NDArray[np.float64]] | list[object] = captured_temperatures,
                **preparation_kwargs,
            ) -> tuple[
                dict[str, dict[str, NDArray[np.float64]]],
                dict[str, dict[str, NDArray[np.float64]]],
            ]:
                result = original_prep(system, **preparation_kwargs)
                captured_temperatures.extend(
                    getattr(system, name).temperatures.copy() for name in ("primary", "secondary")
                )
                return result

            with (
                patch.object(ld, "interpolate_on_ld_grid", capture),
                patch.object(binary_curve_utils, "prep_surface_params", capture_prep),
            ):
                obs = audit_observer(eccentricity, bands)
                phases, raw = obs.observe.lc(**lc_kwargs, normalize=False, flux_unit=u.W / u.m**2)
                _, normalized = obs.observe.lc(**lc_kwargs, normalize=True)
            results.append((phases, raw, normalized, captured_coeffs, captured_temperatures))
        for before, after in zip(results[0][3], results[1][3], strict=True):
            assert_array_equal(before, after)
        assert len(results[0][3]) == len(results[1][3])
        for before, after in zip(results[0][4], results[1][4], strict=True):
            assert_array_equal(before, after)
        assert_array_equal(results[0][0], results[1][0])
        for band in bands:
            assert_array_equal(results[0][1][band], results[1][1][band])
            assert_array_equal(results[0][2][band], results[1][2][band])
        if eccentricity:
            assert ld.get_ld_source_interpolator.cache_info().hits > 0
            assert any(not np.array_equal(results[1][4][0], temperatures) for temperatures in results[1][4])
    finally:
        ld.clear_ld_caches()


def test_same_size_restored_mtime_revision(ld_sources: pathlib.PosixPath) -> None:
    original = evaluate(component()).copy()
    path = ld_sources / ld.get_ld_table_filename("bolometric", 0.0)
    before = path.stat()
    contents = path.read_text()
    changed = contents.replace("0.8", "1.8")
    assert contents != changed
    assert len(contents) == len(changed)
    path.write_text(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert not np.array_equal(evaluate(component()), original)
    # Atomic replacement has a new inode even when size and mtime match.
    replacement = path.with_suffix(".new")
    replacement.write_text(contents)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert_array_equal(evaluate(component()), original)


LAWS = ("linear", "cosine", "logarithmic", "square_root")


def original_factor(
    mu: NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int64],
    coefficients: NDArray[np.float64],
    law: str,
) -> NDArray[np.float64]:
    """Pre-optimization arithmetic, including both copies for matrix input."""
    coeffs = np.asarray(coefficients, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu = mu[:, np.newaxis] if mu.ndim == 1 else mu.copy()
    mu = mu.copy()
    neg = mu <= 0
    if law in ("linear", "cosine"):
        mu[neg] = 0.0
        result = 1.0 - coeffs + coeffs * mu
    elif law == "logarithmic":
        log_mu = mu.copy()
        mu[neg] = 0.0
        log_mu[neg] = 1.0
        result = 1.0 - coeffs[:, :1] * (1 - mu) - coeffs[:, 1:] * mu * np.log(log_mu)
    else:
        mu[neg] = 0.0
        result = 1.0 - coeffs[:, :1] * (1 - mu) - coeffs[:, 1:] * (1 - np.sqrt(mu))
    result[neg] = 0.0
    return result[:, 0] if result.shape[1] == 1 else result


@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("layout", ["vector", "matrix", "transpose", "strided", "readonly"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_factor_bitwise_and_input_ownership(
    law: str,
    layout: str,
    dtype: type[np.float32 | np.float64 | np.int64],
) -> None:
    values = [-1, 0, 1, 2] if dtype == np.int64 else [-1.0, -0.0, 0.0, 0.2, 1.0, np.nan, np.inf, -np.inf]
    mu = np.tile(np.asarray(values, dtype=dtype), (6, 1))
    if layout == "vector":
        mu = mu[0]
    elif layout == "transpose":
        mu = mu.T
    elif layout == "strided":
        mu = mu[::2, ::2]
    elif layout == "readonly":
        mu.flags.writeable = False
    coeffs = np.tile([0.3] if law in ("linear", "cosine") else [0.3, 0.15], (mu.shape[0], 1))
    before = mu.tobytes()
    coeff_before = coeffs.tobytes()
    with np.errstate(invalid="ignore"):
        expected = original_factor(mu, coeffs, law)
        actual = ld.limb_darkening_factor(cos_theta=mu, coefficients=coeffs, limb_darkening_law=law)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert mu.tobytes() == before
    assert coeffs.tobytes() == coeff_before
    assert not np.shares_memory(actual, mu)


@pytest.mark.parametrize("law", [*LAWS, "invalid", None])
def test_factor_errors_preserve_inputs(law: str | None) -> None:
    mu = np.array([[0.0, -0.2], [0.3, 1.0]])
    before = mu.tobytes()
    if law in LAWS:
        with pytest.raises(ElisaUnexpectedError, match="coefficients were not supplied"):
            ld.limb_darkening_factor(cos_theta=mu, limb_darkening_law=law)
    else:
        with pytest.raises(ElisaUnexpectedError):
            ld.limb_darkening_factor(cos_theta=mu, coefficients=np.array([[0.3], [0.4]]), limb_darkening_law=law)
    assert mu.tobytes() == before


@pytest.mark.parametrize("law", LAWS)
def test_factor_vector_route(law: str) -> None:
    normals = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    los = np.array([1.0, 0.0, 0.0])
    coeff = np.tile([0.4] if law in ("linear", "cosine") else [0.3, 0.15], (3, 1))
    expected = original_factor(np.sum(normals * los, axis=-1), coeff, law)
    result = ld.limb_darkening_factor(
        normal_vector=normals,
        line_of_sight=los,
        coefficients=coeff,
        limb_darkening_law=law,
    )
    assert result.tobytes() == expected.tobytes()
    # Explicit cosines take precedence even when the alternative vectors are supplied.
    supplied_cosines = np.array([0.2, 0.3, 0.4])
    supplied_before = supplied_cosines.copy()
    explicit = ld.limb_darkening_factor(
        normal_vector=normals,
        line_of_sight=los,
        cos_theta=supplied_cosines,
        coefficients=coeff,
        limb_darkening_law=law,
    )
    assert explicit.tobytes() == original_factor(supplied_cosines, coeff, law).tobytes()
    np.testing.assert_array_equal(supplied_cosines, supplied_before)


@pytest.mark.parametrize("band", ["bolometric", "rv_band", "Generic.Bessell.V"])
def test_missing_source_lookup_has_domain_error_and_original_diagnostic(
    ld_sources: pathlib.PosixPath,
    band: str,
) -> None:
    source_band = "bolometric" if band == "rv_band" else band
    path = ld_sources / ld.get_ld_table_filename(source_band, 0.0)
    path.unlink()
    with pytest.raises(ElisaLimbDarkeningError) as caught:
        ld.get_ld_source_identities(0.0, (band,))
    assert type(caught.value) is ElisaLimbDarkeningError
    assert str(caught.value) == f"There is no file like {path}."


def test_source_error_is_explicit_in_fitting_policy() -> None:
    from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES

    assert ElisaLimbDarkeningError in SYNTHESIS_FAILURES
    assert isinstance(ElisaLimbDarkeningError("missing table"), SYNTHESIS_FAILURES)


@pytest.mark.parametrize("metallicity", [-100.0, 100.0])
def test_unsupported_metallicity_is_ld_failure_with_cause(metallicity: float) -> None:
    with pytest.raises(ElisaLimbDarkeningError) as caught:
        ld.get_relevant_ld_tables("bolometric", metallicity)
    assert isinstance(caught.value.__cause__, ElisaLookupError)


@pytest.mark.parametrize("metallicity", [np.nan, np.inf, -np.inf])
def test_nonfinite_metallicity_is_ld_failure_before_source_lookup(metallicity: float) -> None:
    with (
        patch.object(ld.utils, "find_surrounded") as select,
        pytest.raises(ElisaLimbDarkeningError, match="nonfinite metallicity"),
    ):
        ld.get_relevant_ld_tables("bolometric", metallicity)
    select.assert_not_called()


@pytest.mark.usefixtures("ld_sources")
def test_selected_table_disappearing_at_read_is_ld_failure() -> None:
    source = ld.get_ld_source_identities(0.0, ("bolometric",))[0]
    failure = FileNotFoundError("table disappeared")
    with patch.object(ld.pd, "read_csv", side_effect=failure), pytest.raises(ElisaLimbDarkeningError) as caught:
        ld.get_ld_source_interpolator(source)
    assert caught.value.__cause__ is failure


@pytest.mark.usefixtures("ld_sources")
def test_nonfinite_interpolation_result_is_ld_failure() -> None:
    with (
        patch.object(ld, "get_ld_source_interpolator", return_value=lambda query: np.full((len(query), 1), np.inf)),
        pytest.raises(ElisaLimbDarkeningError, match="infinity"),
    ):
        ld.interpolate_on_ld_grid(np.array([5500.0]), np.array([1.5]), 0.0, ["bolometric"])


@pytest.mark.parametrize("failure", [ValueError("bad interpolation input"), RuntimeError("interpolator failed")])
@pytest.mark.usefixtures("ld_sources")
def test_source_interpolator_construction_failure_is_ld_error_with_cause(
    failure: RuntimeError | ValueError,
) -> None:
    source = ld.get_ld_source_identities(0.0, ("bolometric",))[0]
    with (
        patch.object(ld.interpolate, "LinearNDInterpolator", side_effect=failure),
        pytest.raises(ElisaLimbDarkeningError) as caught,
    ):
        ld.get_ld_source_interpolator(source)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    ("temperature", "log_g", "metallicity"),
    [
        (np.array(["invalid"]), np.array([1.5]), 0.0),
        (np.array([5500.0]), np.array(["invalid"]), 0.0),
        (np.array([5500.0]), np.array([1.5]), "invalid"),
        (np.array([5500.0, 5600.0]), np.array([1.5]), 0.0),
        (np.full((2, 2), 5500.0), np.full(2, 1.5), 0.0),
    ],
)
@pytest.mark.usefixtures("ld_sources")
def test_candidate_input_conversion_and_shape_failures_are_ld_errors(
    temperature: NDArray[np.generic],
    log_g: NDArray[np.generic],
    metallicity: float | str,
) -> None:
    with pytest.raises(ElisaLimbDarkeningError) as caught:
        ld.interpolate_on_ld_grid(temperature, log_g, metallicity, ["bolometric"])
    assert isinstance(caught.value.__cause__, (ValueError, TypeError))


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("access denied"),
        OSError("device failure"),
        pd.errors.ParserError("invalid CSV"),
        pd.errors.EmptyDataError("empty CSV"),
        MemoryError("out of memory"),
        TypeError("reader bug"),
    ],
)
@pytest.mark.usefixtures("ld_sources")
def test_table_reader_failures_are_unexpected_with_cause(
    failure: MemoryError
    | OSError
    | PermissionError
    | TypeError
    | pandas.errors.EmptyDataError
    | pandas.errors.ParserError,
) -> None:
    source = ld.get_ld_source_identities(0.0, ("bolometric",))[0]
    with patch.object(ld.pd, "read_csv", side_effect=failure), pytest.raises(ElisaUnexpectedError) as caught:
        ld.get_ld_source_interpolator(source)
    assert caught.value.__cause__ is failure


@pytest.mark.usefixtures("ld_sources")
def test_missing_table_column_is_unexpected_with_cause() -> None:
    source = ld.get_ld_source_identities(0.0, ("bolometric",))[0]
    with (
        patch.object(ld.pd, "read_csv", return_value=pd.DataFrame({"wrong": [1]})),
        pytest.raises(ElisaUnexpectedError) as caught,
    ):
        ld.get_ld_source_interpolator(source)
    assert isinstance(caught.value.__cause__, KeyError)


@pytest.mark.usefixtures("ld_sources")
def test_table_stat_failure_is_unexpected_with_cause() -> None:
    failure = PermissionError("cannot stat LD table")
    original_stat = Path.stat

    def fail_table_stat(path: Path, *args, **kwargs) -> None:
        if path.suffix == ".csv":
            raise failure
        return original_stat(path, *args, **kwargs)

    with patch.object(Path, "stat", fail_table_stat), pytest.raises(ElisaUnexpectedError) as caught:
        ld.get_ld_source_identities(0.0, ("bolometric",))
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("integrated", [False, True])
def test_missing_prepared_coefficients_are_internal_contract_failure(integrated: bool) -> None:
    with pytest.raises(ElisaUnexpectedError, match="coefficients were not supplied"):  # noqa: PT012 - scenario includes context cleanup or selection of the operation under test
        if integrated:
            ld.calculate_integrated_limb_darkening_factor("linear")
        else:
            ld.limb_darkening_factor(cos_theta=np.ones(2), limb_darkening_law="linear")


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("law", ["invalid", None])
def test_unsupported_factor_law_is_internal_contract_failure(integrated: bool, law: str | None) -> None:
    coefficients = np.ones((2, 2))
    with pytest.raises(ElisaUnexpectedError, match=r"law|rule"):  # noqa: PT012 - scenario includes context cleanup or selection of the operation under test
        if integrated:
            ld.calculate_integrated_limb_darkening_factor(law, coefficients)
        else:
            ld.limb_darkening_factor(
                coefficients=coefficients,
                cos_theta=np.ones(2),
                limb_darkening_law=law,
            )


@pytest.mark.usefixtures("ld_sources")
def test_law_override_round_trip_matches_uncached_griddata() -> None:
    star = SimpleNamespace(
        temperatures=np.array([5523.123456789]),
        log_g=np.array([1.456789123]),
        metallicity=0.0,
        limb_darkening_coefficients=None,
    )

    def evaluate() -> dict[str, NDArray[np.float64]]:
        result = curve_utils.get_component_limbdarkening_cfs(star, ["bolometric"], symmetry_test=False)
        points, values = legacy_source("bolometric", 0.0)
        expected = griddata(points, values, np.column_stack((star.temperatures, star.log_g + 2)), method="linear")
        assert_array_equal(result["bolometric"], expected)
        return result

    with settings.override(LIMB_DARKENING_LAW="cosine"):
        original = evaluate()
        with settings.override(LIMB_DARKENING_LAW="logarithmic"):
            changed = evaluate()
            assert changed["bolometric"].shape == (1, 2)
        restored = evaluate()
        assert_array_equal(original["bolometric"], restored["bolometric"])
        assert changed["bolometric"].shape == (1, 2)
    info = curve_utils.interpolate_on_ld_grid_cached.cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 2, 2)
