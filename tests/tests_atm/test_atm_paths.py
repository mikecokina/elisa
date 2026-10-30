"""Path-selection parity with the pre-deduplication atmosphere implementation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from elisa import atm, settings, utils
from elisa.base.error import ElisaSynthesisError
from elisa.atm import atm_file_prefix_to_quantity_list, get_atm_directory, validated_atlas
from elisa.observer.passband import PassbandContainer
from elisa.types import Float


def original_atm_files(
    temperature: NDArray[Float],
    log_g: NDArray[Float],
    metallicity: Float,
    atlas: str,
) -> list[str | Path]:
    """Retain the original implementation as an independent compatibility reference.

    :param temperature: Iterable[float];
    :param log_g: Iterable[float];
    :param metallicity: float;
    :param atlas: str; atmosphere model identificator (see settings.ATLAS_TO_ATM_FILE_PREFIX.keys())
    :return: List[str];
    """
    atlas = validated_atlas(atlas)
    log_g = utils.convert_gravity_acceleration_array(log_g, "log_cgs")

    g_array = np.array(atm_file_prefix_to_quantity_list("gravity", atlas))

    m_array = np.array(atm_file_prefix_to_quantity_list("metallicity", atlas))
    t_array = np.array(atm_file_prefix_to_quantity_list("temperature", atlas))

    g = utils.find_nearest_value_as_matrix(g_array, log_g)[0]
    m = utils.find_nearest_value_as_matrix(m_array, metallicity)[0][0]
    t = utils.find_surrounded_as_matrix(t_array, temperature)

    domain_df = pd.DataFrame(
        {
            "temp": t.flatten("F"),
            "log_g": np.tile(g, 2),
            "mh": np.repeat(m, len(g) * 2),
        },
    )
    directory = get_atm_directory(m, atlas)

    # Vectorized string conversions (faster than pandas .apply())
    # Metallicity: 0.5 -> 'p05', -1.1 -> 'm11'
    signs = np.where(domain_df["mh"].to_numpy() >= 0, "p", "m")
    abs_mh = np.abs(domain_df["mh"].to_numpy() * 10).astype(int)
    mh_name = np.char.add(signs, np.char.zfill(abs_mh.astype(str), 2))

    # Temperature: simple int conversion
    temp_name = domain_df["temp"].to_numpy().astype(int).astype(str)

    # Surface gravity: 0.5 -> 'g05', 1.0 -> 'g10'
    logg_int = (domain_df["log_g"].to_numpy() * 10).astype(int)
    log_g_name = np.char.add("g", np.char.zfill(logg_int.astype(str), 2))

    # Build filenames
    fnames = [f"{atlas}{mh}_{temp}_{g}" for mh, temp, g in zip(mh_name, temp_name, log_g_name)]

    return [str(Path(settings.ATLAS_TO_BASE_DIR[atlas]) / directory / (fname + ".csv")) for fname in fnames]


@pytest.mark.parametrize("atlas", ["castelli", "castelli-kurucz", "ck", "ck04", "kurucz", "k93", "k"])
@pytest.mark.parametrize(("selection", "metallicity"), [
    (selection, metallicity)
    for selection in ("duplicates", "mixed", "boundaries")
    for metallicity in (-2.5, -0.1, -0.0, 0.05, 0.5, 10.0, np.nan)
] + [("ties", None)])
def test_atm_files_matches_original(atlas, metallicity, selection):
    """Preserve ordering, nearest-grid ties, aliases and grid endpoints."""
    grid = np.asarray(atm.atm_file_prefix_to_quantity_list("temperature", atlas))
    if selection == "duplicates":
        temperature = np.repeat(4999.0, 128)
        gravity = np.repeat(1.5, 128)
    elif selection == "mixed":
        temperature = np.array([11500, 4999, 11300, 4999, 10000, 11500, 10001])
        gravity = np.array([2, 1.5, 2, 1.5, -3, 6, 2.25])
    elif selection == "boundaries":
        temperature = grid[[0, -1, 0, len(grid) // 2, -1]]
        gravity = np.array([0, 2, 3, 2, 0])
    else:
        temperature = np.array([10000.0, 10001.0, 10000.0])
        gravity_grid = np.asarray(atm.atm_file_prefix_to_quantity_list("gravity", atlas))
        gravity = np.repeat((gravity_grid[0] + gravity_grid[1]) / 2 - 2, 3)
        metallicity_grid = np.asarray(atm.atm_file_prefix_to_quantity_list("metallicity", atlas))
        metallicity = (metallicity_grid[0] + metallicity_grid[1]) / 2

    expected = original_atm_files(temperature, gravity, metallicity, atlas)
    obtained = atm.NaiveInterpolatedAtm.atm_files(temperature, gravity, metallicity, atlas)
    assert type(obtained) is list
    assert all(type(path) is str for path in obtained)
    assert obtained == expected
    assert atm.unique_atm_fpaths(obtained) == atm.unique_atm_fpaths(expected)


@pytest.mark.parametrize("atlas", ["ck04", "k93"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_mixed_strided_input_dtypes(atlas, dtype):
    grid = np.asarray(atm.atm_file_prefix_to_quantity_list("temperature", atlas))
    rng = np.random.default_rng(42)
    temperature = rng.uniform(grid.min(), grid.max(), 512).astype(dtype)[::2]
    gravity = rng.uniform(-3.0, 6.0, 512).astype(dtype)[::2]
    assert atm.NaiveInterpolatedAtm.atm_files(temperature, gravity, -0.25, atlas) == (
        original_atm_files(temperature, gravity, -0.25, atlas)
    )


@pytest.mark.parametrize("atlas", ["ck04", "k93", "unsupported", "bb"])
@pytest.mark.parametrize("temperature,gravity,metallicity", [
    ([], [], 0.0),
    ([3499.0], [2.0], 0.0),
    ([50001.0], [2.0], 0.0),
    ([np.nan], [2.0], 0.0),
    ([np.inf], [2.0], 0.0),
    ([5000.0, 6000.0], [2.0], 0.0),
    ([5000.0], [], 0.0),
    ([[5000.0, 6000.0]], [2.0], 0.0),
])
def test_atm_files_preserves_errors(atlas, temperature, gravity, metallicity):
    args = np.asarray(temperature), np.asarray(gravity), metallicity, atlas
    with pytest.raises(Exception) as expected:
        original_atm_files(*args)
    with pytest.raises((type(expected.value), ElisaSynthesisError)) as obtained:
        atm.NaiveInterpolatedAtm.atm_files(*args)
    assert str(expected.value) in str(obtained.value)


@pytest.mark.parametrize("atlas", ["ck04", "k93"])
def test_paths_are_formatted_once_per_selected_identity(atlas):
    temperature = np.tile([4999.0, 10000.0, 10001.0], 100)
    gravity = np.tile([1.5, 2.0, 2.0], 100)
    expected = original_atm_files(temperature, gravity, 0.0, atlas)
    with patch.object(atm, "Path", wraps=Path) as constructor:
        obtained = atm.NaiveInterpolatedAtm.atm_files(temperature, gravity, 0.0, atlas)
    assert obtained == expected
    assert constructor.call_count == 1
    # Gathering reuses each unique string, without formatting a string per face.
    assert len({id(path) for path in obtained}) == len(set(expected))


@pytest.mark.parametrize("atlas", ["ck04", "k93"])
def test_base_directory_changes_and_missing_files(atlas, monkeypatch, tmp_path):
    prefix = atm.validated_atlas(atlas)
    args = np.array([10000.0]), np.array([2.0]), 0.0, atlas
    monkeypatch.setattr(settings, "CK04_ATM_TABLES" if prefix == "ck" else "K93_ATM_TABLES", tmp_path / "first directory")
    first = atm.NaiveInterpolatedAtm.atm_files(*args)
    assert first == original_atm_files(*args)
    monkeypatch.setattr(settings, "CK04_ATM_TABLES" if prefix == "ck" else "K93_ATM_TABLES", tmp_path / "second")
    second = atm.NaiveInterpolatedAtm.atm_files(*args)
    assert second == original_atm_files(*args)
    assert first != second
    # Path selection never checks existence; loading retains its existing exception.
    for paths in (original_atm_files(*args), second):
        with pytest.raises(atm.ElisaAtmosphereError) as error:
            atm.read_unique_atm_tables(paths)
        assert str(error.value) == (
            f"file {second[0]} doesn't exist. Your atmosphere tables are "
            "either not properly installed or atmosphere parameters "
            "of your stellar model are not covered by the currently used table."
        )


@pytest.mark.parametrize("atlas", ["ck04", "k93"])
def test_selected_tables_preserve_spectra_and_radiances(atlas, monkeypatch, tmp_path):
    """Exercise both filename families through loading and spectral interpolation."""
    monkeypatch.setattr(settings, "CK04_ATM_TABLES" if atm.validated_atlas(atlas) == "ck" else "K93_ATM_TABLES", tmp_path)
    monkeypatch.setattr(atm.buffer, "ATMOSPHERE_TABLES", {})
    args = np.array([10000.0, 10001.0, 10250.0, 10001.0]), np.repeat(2.0, 4), 0.0, atlas
    wave = np.linspace(1000.0, 8000.0, 29)
    for filename in set(original_atm_files(*args)):
        path = Path(filename)
        path.parent.mkdir(exist_ok=True)
        temperature = atm.parse_domain_quantities_from_atm_table_filename(path.name)[0]
        pd.DataFrame({"wave": wave, "flux": (wave / 1000.0) ** 2 * temperature}).to_csv(path, index=False)

    band = PassbandContainer(
        table=pd.DataFrame({
            settings.PASSBAND_DATAFRAME_WAVE: [3000.0, 3500.0, 4000.0, 4500.0],
            settings.PASSBAND_DATAFRAME_THROUGHPUT: [0.0, 1.0, 0.8, 0.0],
        }),
        passband="test",
    )
    kwargs = {"passband": {"test": band}, "left_bandwidth": 3000.0, "right_bandwidth": 4500.0}
    selector = atm.NaiveInterpolatedAtm.atm_files
    with patch.object(atm.NaiveInterpolatedAtm, "atm_files", side_effect=original_atm_files):
        expected = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
        expected_radiance = atm.NaiveInterpolatedAtm.atlas_radiance(*args, **kwargs)
    with patch.object(atm.NaiveInterpolatedAtm, "atm_files", side_effect=selector):
        obtained = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
        obtained_radiance = atm.NaiveInterpolatedAtm.atlas_radiance(*args, **kwargs)

    assert obtained[1:] == expected[1:]
    for key in expected[0]["test"]:
        np.testing.assert_array_equal(obtained[0]["test"][key], expected[0]["test"][key])
    np.testing.assert_array_equal(obtained_radiance["test"], expected_radiance["test"])
