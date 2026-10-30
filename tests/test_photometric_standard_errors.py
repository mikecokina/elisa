"""Missing or malformed packaged calibration resources are fatal contract failures."""

import json

import pytest

from elisa import Observer, settings
from elisa.base.error import ElisaUnexpectedError
from elisa.photometric_standards.standards_handlers import load_standard


@pytest.mark.parametrize("contents,cause", [(None, FileNotFoundError), ("{broken", json.JSONDecodeError)])
def test_load_standard_preserves_resource_failure_cause(tmp_path, contents, cause):
    path = tmp_path / "vega.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(ElisaUnexpectedError, match=str(path)) as caught:
        load_standard("vega", source_path=path)
    assert isinstance(caught.value.__cause__, cause)


@pytest.mark.parametrize("contents,cause", [(None, FileNotFoundError), ("{broken", json.JSONDecodeError)])
def test_observer_cache_rejects_broken_source_and_recovers(tmp_path, monkeypatch, contents, cause):
    expected = load_standard(settings.MAGNITUDE_SYSTEM)
    observer = Observer()
    directory = tmp_path / "zero_points"
    directory.mkdir()
    path = directory / f"{settings.MAGNITUDE_SYSTEM.lower()}.json"
    path.write_text(json.dumps(expected))
    monkeypatch.setattr(type(settings), "DATA_PATH", tmp_path)
    # noinspection PyProtectedMember
    previous = observer._get_zero_points()
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    with pytest.raises(ElisaUnexpectedError) as caught:
        # noinspection PyProtectedMember
        observer._get_zero_points()
    assert isinstance(caught.value.__cause__, cause)
    assert observer.zero_points is previous
    path.write_text(json.dumps(expected))
    # noinspection PyProtectedMember
    assert observer._get_zero_points() == expected
