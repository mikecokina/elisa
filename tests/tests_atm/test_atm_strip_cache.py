"""Exact numeric strip reuse, error parity and mutable container isolation."""

import hashlib
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from elisa import settings
from elisa.atm import atm
from elisa.observer.passband import PassbandContainer


def original_strip(atm_container, left_bandwidth, right_bandwidth, *, inplace=False):
    """Pre-cache strip implementation, retained as an independent reference."""
    container, left, right = atm_container, left_bandwidth, right_bandwidth
    valid = list(np.where(np.logical_and(np.greater(container.model.wavelength, left),
                                        np.less(container.model.wavelength, right)))[0])
    first = valid[0] - 1 if valid[0] >= 1 else 0
    last = valid[-1] + 1 if valid[-1] < container.model.last_valid_index() else valid[-1]
    result = container if inplace else deepcopy(container)
    result.model = result.model[np.unique([first, *valid, last])]
    return atm.extend_atm_container_on_bandwidth_boundary(result, left, right)


def spectrum():
    wave = np.arange(11, dtype=float)
    c = atm.AtmDataContainer(atm.AtmModel(wavelength=wave, flux=1.123456789123456 + np.sin(wave / 3)),
                             5000.0, 2.0, 0.0, fpath="source")
    c.extra = {"nested": [1, 2]}
    return c


def assert_bits(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def assert_model(a, b):
    assert_bits(a.model.wavelength, b.model.wavelength)
    assert_bits(a.model.flux, b.model.flux)
    assert a.left_bandwidth == b.left_bandwidth
    assert a.right_bandwidth == b.right_bandwidth


@pytest.fixture(autouse=True)
def empty_strip_cache():
    atm.clear_atm_strip_cache()
    yield
    atm.clear_atm_strip_cache()


def test_key_contains_framed_array_digest_and_exact_boundary_bits():
    source = spectrum()
    digest = hashlib.blake2b(digest_size=16)
    for values in (source.model.wavelength, source.model.flux):
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    key = atm._atm_strip_cache_key(source, -0.0, 8.75)
    assert key[0] == digest.digest()
    assert len(key[0]) == 16
    assert key[1] == source.model.wavelength.nbytes + source.model.flux.nbytes
    assert key[2:4] == (np.float64(-0.0).tobytes(), np.float64(8.75).tobytes())
    assert max(len(part) for part in key if isinstance(part, bytes)) == 16
    swapped = deepcopy(source)
    swapped.model.wavelength, swapped.model.flux = swapped.model.flux, swapped.model.wavelength
    assert atm._atm_strip_cache_key(swapped, -0.0, 8.75)[0] != key[0]


def test_digest_preserves_legacy_combined_payload_limit(monkeypatch):
    source = spectrum()
    result = original_strip(source, 1.25, 8.75)
    budget = (source.model.wavelength.nbytes + source.model.flux.nbytes + 16
              + result.model.wavelength.nbytes + result.model.flux.nbytes)
    monkeypatch.setattr(atm, "_STRIP_CACHE_MAX_PAYLOAD", budget - 1)
    atm.strip_to_bandwidth(source, 1.25, 8.75)
    assert atm.atm_strip_cache_info()["currsize"] == 0
    assert atm.atm_strip_cache_info()["bypasses"] == 1
    monkeypatch.setattr(atm, "_STRIP_CACHE_MAX_PAYLOAD", budget)
    assert_model(atm.strip_to_bandwidth(source, 1.25, 8.75), result)
    assert atm.atm_strip_cache_info()["currsize"] == 1
    assert_model(atm.strip_to_bandwidth(source, 1.25, 8.75), result)
    assert atm.atm_strip_cache_info()["hits"] == 1


@pytest.mark.parametrize("inplace", [False, True])
def test_cold_and_warm_results_are_independent_and_exact(inplace):
    source = spectrum()
    before = deepcopy(source)
    expected = original_strip(source, 1.25, 8.75)
    outputs = []
    for _ in range(3):
        current = deepcopy(source)
        result = atm.strip_to_bandwidth(current, 1.25, 8.75, inplace=inplace)
        assert_model(result, expected)
        assert (result is current) == inplace
        assert result.model.flux.flags.writeable and result.model.wavelength.flags.writeable
        assert not np.shares_memory(result.model.flux, source.model.flux)
        if not inplace:
            assert_model(current, before)
            assert result.extra is not current.extra
        outputs.append(result)
    assert atm.atm_strip_cache_info()["hits"] == 2
    assert atm.atm_strip_cache_info()["misses"] == 1
    outputs[0].model.flux[:] = -1
    outputs[0].model.wavelength[:] = -1
    outputs[0].extra["nested"].append(99)
    for result in outputs[1:]:
        assert_model(result, expected)
        assert result.extra == source.extra
    assert_model(atm.strip_to_bandwidth(source, 1.25, 8.75), expected)
    assert_model(source, before)
    for wave, flux in atm._STRIP_CACHE.values():
        with pytest.raises(ValueError):
            wave.setflags(write=True)
        with pytest.raises(ValueError):
            flux[0] = -1


@pytest.mark.parametrize("inplace", [False, True])
def test_metadata_is_taken_from_current_container(inplace):
    source = spectrum()
    atm.strip_to_bandwidth(source, 1.25, 8.75)
    source.temperature, source.log_g, source.metallicity = 6500.0, 3.5, -0.5
    source.fpath = "different"
    source.flux_to_si_mult, source.wave_to_si_mult = 3.0, 7.0
    source.extra["nested"].append(4)
    expected = original_strip(deepcopy(source), 1.25, 8.75, inplace=inplace)
    result = atm.strip_to_bandwidth(source, 1.25, 8.75, inplace=inplace)
    assert_model(result, expected)
    for name in ("temperature", "log_g", "metallicity", "fpath", "flux_to_si_mult", "wave_to_si_mult", "extra"):
        assert getattr(result, name) == getattr(expected, name)
    assert atm.atm_strip_cache_info()["hits"] == 1
    assert (result is source) == inplace


def test_cache_hit_preserves_metadata_deepcopy_errors():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("metadata cannot be copied")

    source = spectrum()
    atm.strip_to_bandwidth(source, 1.25, 8.75)
    source.extra = Uncopyable()
    for strip in (original_strip, atm.strip_to_bandwidth):
        with pytest.raises(RuntimeError, match="metadata cannot be copied"):
            strip(source, 1.25, 8.75)
    assert atm.strip_to_bandwidth(source, 1.25, 8.75, inplace=True) is source


@pytest.mark.parametrize("change", ["flux", "grid", "shape", "left", "right", "dtype"])
def test_changed_numeric_inputs_do_not_hit(change):
    source = spectrum()
    atm.strip_to_bandwidth(source, 1.25, 8.75)
    left, right = 1.25, 8.75
    if change == "flux":
        source.model.flux[4] = np.nextafter(source.model.flux[4], np.inf)
    elif change == "grid":
        source.model.wavelength[4] = np.nextafter(4.0, np.inf)
    elif change == "shape":
        source.model.flux = np.append(source.model.flux, 1.0)
        source.model.wavelength = np.append(source.model.wavelength, 11.0)
    elif change == "left":
        left = np.nextafter(left, np.inf)
    elif change == "right":
        right = np.nextafter(right, np.inf)
    else:
        source.model.flux = source.model.flux.astype(np.float32)
    expected = original_strip(source, left, right)
    result = atm.strip_to_bandwidth(source, left, right)
    assert_model(result, expected)
    assert atm.atm_strip_cache_info()["hits"] == 0


def test_signed_zero_boundary_bits_are_distinct():
    source = spectrum()
    for left in (0.0, -0.0):
        assert_model(atm.strip_to_bandwidth(source, left, 8.75), original_strip(source, left, 8.75))
    assert atm.atm_strip_cache_info()["misses"] == 2
    assert np.signbit(atm.strip_to_bandwidth(source, -0.0, 8.75).model.wavelength[0])


def test_strided_readonly_inputs_and_floating_point_context():
    source = spectrum()
    for name in ("wavelength", "flux"):
        values = np.repeat(getattr(source.model, name), 2)[::2]
        values.setflags(write=False)
        setattr(source.model, name, values)
    expected = original_strip(source, 1.25, 8.75)
    assert_model(atm.strip_to_bandwidth(source, 1.25, 8.75), expected)
    assert_model(atm.strip_to_bandwidth(source, 1.25, 8.75, inplace=True), expected)
    assert atm.atm_strip_cache_info()["hits"] == 1
    # A changed NumPy exception policy must not reuse a result from a different policy.
    with np.errstate(over="raise"):
        original = spectrum()
        assert_model(atm.strip_to_bandwidth(original, 1.25, 8.75), original_strip(original, 1.25, 8.75))
    assert atm.atm_strip_cache_info()["misses"] == 2


def test_coverage_fallback_warnings_are_not_cached():
    source = spectrum()
    for _ in range(2):
        with pytest.warns(UserWarning, match="argument bandwidth is out of bound"):
            result = atm.strip_atm_container_by_bandwidth(source, -1.0, 20.0)
        assert_model(result, original_strip(source, 0.0, 10.0))
    assert atm.atm_strip_cache_info()["hits"] == 1


def test_apply_passband_copies_once_and_isolates_each_band():
    source = spectrum()
    before = deepcopy(source)
    first = PassbandContainer.from_name("Generic.Bessell.U")
    second = PassbandContainer.from_name("Generic.Bessell.V")
    for b, value in ((first, 0.5), (second, 0.75)):
        b.left_bandwidth, b.right_bandwidth = 1.25, 8.75
        b.akima = lambda x, value=value: np.full_like(x, value)
    for _ in range(2):
        with patch.object(atm, "deepcopy", wraps=deepcopy) as copies:
            result = atm.apply_passband([source], {"first": first, "second": second})
        assert copies.call_count == 2
        expected = original_strip(source, 1.25, 8.75)
        assert_bits(result["first"][0].model.flux, expected.model.flux * 0.5)
        assert_bits(result["second"][0].model.flux, expected.model.flux * 0.75)
        result["first"][0].model.flux[:] = -1
        result["first"][0].extra["nested"].append(99)
        assert_bits(result["second"][0].model.flux, expected.model.flux * 0.75)
        assert result["second"][0].extra == before.extra
        assert_model(source, before)


def test_apply_passband_validates_after_private_copy():
    source = spectrum()
    band = PassbandContainer.from_name("Generic.Bessell.U")
    band.left_bandwidth, band.right_bandwidth = 1.1, 1.2
    with patch.object(atm, "deepcopy", wraps=deepcopy) as copies:
        with pytest.raises(IndexError):
            atm.apply_passband([source], {"test": band})
    assert copies.call_count == 1
    assert copies.call_args.args[0] is source


def test_alignment_and_changed_selected_grids_preserve_both_stages():
    models = [spectrum(), spectrum(), spectrum()]
    models[1].model.wavelength += 0.125
    models[2].model.wavelength += 0.25
    band = PassbandContainer.from_name("Generic.Bessell.U")
    band.left_bandwidth, band.right_bandwidth = 2.25, 7.75
    band.akima = lambda x: np.ones_like(x)

    def prepare(selected):
        trimmed = atm.strip_atm_containers_by_bandwidth(selected, 1.25, 8.75)
        aligned = atm.arange_atm_to_same_wavelength(trimmed)
        return atm.apply_passband(aligned, {"test": band})["test"]

    for selected in (models[:2], models, models):
        with patch.object(atm, "strip_to_bandwidth", original_strip):
            expected = prepare(selected)
        obtained = prepare(selected)
        for old, new in zip(expected, obtained):
            assert_model(new, old)
        assert all(np.all(np.diff(c.model.wavelength) > 0) for c in obtained)
    assert atm.atm_strip_cache_info()["hits"] > 0


def test_lru_eviction_capacity_and_clear():
    source = spectrum()
    for index in range(128):
        atm.strip_to_bandwidth(source, 1.0 + index / 1024, 8.75)
    assert atm.atm_strip_cache_info()["currsize"] == 128
    atm.strip_to_bandwidth(source, 1.0, 8.75)  # Refresh the oldest entry.
    atm.strip_to_bandwidth(source, 1.5, 8.75)
    before = atm.atm_strip_cache_info()
    atm.strip_to_bandwidth(source, 1.0, 8.75)
    assert atm.atm_strip_cache_info()["hits"] == before["hits"] + 1
    atm.strip_to_bandwidth(source, 1.0 + 1 / 1024, 8.75)
    assert atm.atm_strip_cache_info()["misses"] == before["misses"] + 1
    assert atm.atm_strip_cache_info()["currsize"] == 128
    atm.clear_atm_strip_cache()
    assert atm.atm_strip_cache_info() == {"hits": 0, "misses": 0, "bypasses": 0, "maxsize": 128, "currsize": 0}


@pytest.mark.parametrize("length", [5000, 10000])
def test_oversized_key_or_result_is_not_stored(length):
    wave = np.arange(length, dtype=float)
    source = atm.AtmDataContainer(atm.AtmModel(wavelength=wave, flux=1 + np.sin(wave / 100)), 5000, 2, 0)
    expected = original_strip(source, 1.25, length - 2.25)
    for _ in range(2):
        assert_model(atm.strip_to_bandwidth(source, 1.25, length - 2.25), expected)
    info = atm.atm_strip_cache_info()
    assert info["currsize"] == info["hits"] == 0
    assert info["bypasses"] == 2


@pytest.mark.parametrize("bad", ["empty", "reversed", "outside", "nan_bound", "nan_flux", "duplicate", "unsorted",
                                  "shape", "two_dimensional", "none"])
@pytest.mark.parametrize("inplace", [False, True])
def test_error_type_message_and_partial_mutation_match(bad, inplace):
    source = spectrum()
    left, right = 1.25, 8.75
    if bad == "empty":
        left, right = 1.1, 1.2
    elif bad == "reversed":
        left, right = right, left
    elif bad == "outside":
        left, right = -0.25, 11.0
    elif bad == "nan_bound":
        left = np.nan
    elif bad == "nan_flux":
        source.model.flux[4] = np.nan
    elif bad == "duplicate":
        source.model.wavelength[4] = 3.0
    elif bad == "unsorted":
        source.model.wavelength[4] = 2.5
    elif bad == "shape":
        source.model.flux = source.model.flux[:3]
    elif bad == "two_dimensional":
        source.model.wavelength = source.model.wavelength[np.newaxis]
    else:
        source.model.flux = None
    old_input, new_input = deepcopy(source), deepcopy(source)
    with pytest.raises(Exception) as expected:
        original_strip(old_input, left, right, inplace=inplace)
    with pytest.raises(type(expected.value)) as obtained:
        atm.strip_to_bandwidth(new_input, left, right, inplace=inplace)
    assert str(obtained.value) == str(expected.value)
    for name in ("wavelength", "flux"):
        old, new = getattr(old_input.model, name), getattr(new_input.model, name)
        if old is None:
            assert new is None
        else:
            assert_bits(old, new)
    assert atm.atm_strip_cache_info()["currsize"] == 0


@pytest.mark.parametrize("atlas", ["ck04", "k93"])
def test_table_families_and_throughput_remain_uncached(atlas, monkeypatch, tmp_path):
    atlas = atm.validated_atlas(atlas)
    monkeypatch.setattr(settings, "CK04_ATM_TABLES" if atm.validated_atlas(atlas) == "ck" else "K93_ATM_TABLES", tmp_path)
    monkeypatch.setattr(atm.buffer, "ATMOSPHERE_TABLES", {})
    args = np.array([10000.0, 10001.0]), np.repeat(2.0, 2), 0.0, atlas
    wave = np.arange(2500., 5501., 100.)
    for path in set(atm.NaiveInterpolatedAtm.atm_files(*args)):
        p = Path(path); p.parent.mkdir(exist_ok=True)
        np.savetxt(p, np.column_stack((wave, 1 + wave / 1000)), delimiter=",", header="wave,flux", comments="")
    band = PassbandContainer.from_name("Generic.Bessell.U")
    kwargs = dict(passband={"Generic.Bessell.U": band}, left_bandwidth=band.left_bandwidth,
                  right_bandwidth=band.right_bandwidth)
    with patch.object(atm, "strip_to_bandwidth", original_strip):
        expected = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
    first = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
    second = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
    for bandname in expected[0]:
        for key in expected[0][bandname]:
            assert_bits(first[0][bandname][key], expected[0][bandname][key])
            assert_bits(second[0][bandname][key], expected[0][bandname][key])
    hits = atm.atm_strip_cache_info()["hits"]
    band.akima = lambda x: np.zeros_like(x)
    third = atm.NaiveInterpolatedAtm.get_atm_profiles(*args, **kwargs)
    assert np.count_nonzero(third[0]["Generic.Bessell.U"]["flux"]) == 0
    assert atm.atm_strip_cache_info()["hits"] > hits
