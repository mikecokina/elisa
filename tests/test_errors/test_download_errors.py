"""Download and extraction failures use the ELISA settings exception category."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import requests

from elisa.base.error import ElisaError, ElisaSettingsError, raise_elisa_settings_error
from elisa.managers import download_manager
from elisa.managers.download_manager import DownloadManager

if TYPE_CHECKING:
    import pathlib


def archive(tmp_path: pathlib.Path, member_name: str) -> str:
    path = tmp_path / "assets.tar"
    payload = b"table data"
    with tarfile.open(path, "w") as output:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    return str(path)


def test_download_error_inheritance_and_helper_cause() -> None:
    assert ElisaSettingsError.__bases__ == (ElisaError,)
    cause = OSError("original failure")
    with pytest.raises(ElisaSettingsError, match="download failed") as caught:
        raise_elisa_settings_error("download failed", cause)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("status", [400, 500])
def test_failed_head_has_download_category(status: int, tmp_path: Path) -> None:
    with (
        patch("elisa.managers.download_manager.requests.head", return_value=Mock(status_code=status)),
        patch.object(DownloadManager, "handler") as handler,
        pytest.raises(ElisaSettingsError, match=f"Status Code: {status}"),
    ):
        DownloadManager.download_file("https://example.com/assets.tar", str(tmp_path / "download"))
    handler.assert_not_called()


@pytest.mark.parametrize("status", [400, 500])
def test_failed_get_has_download_category(status: int, tmp_path: Path) -> None:
    destination = tmp_path / "download"
    with (
        patch("elisa.managers.download_manager.requests.get", return_value=Mock(status_code=status)),
        pytest.raises(ElisaSettingsError, match=f"Status Code: {status}"),
    ):
        DownloadManager.handler(None, "https://example.com/assets.tar", str(destination))
    assert not destination.exists()


def test_unsafe_archive_has_download_category(tmp_path: Path) -> None:
    path = archive(tmp_path, "../outside.txt")
    with pytest.raises(ElisaSettingsError, match="Path Traversal"):
        DownloadManager.extract(path, str(tmp_path / "destination"))
    assert not (tmp_path / "outside.txt").exists()


def test_invalid_archive_path_retains_the_original_cause(tmp_path: Path) -> None:
    path = archive(tmp_path, "table.dat")
    cause = OSError("cannot resolve path")
    with (
        patch.object(Path, "resolve", side_effect=cause),
        pytest.raises(ElisaSettingsError, match=r"Invalid member path table.dat") as caught,
    ):
        DownloadManager.extract(path, str(tmp_path / "destination"))
    assert caught.value.__cause__ is cause


def test_valid_extraction_and_prefix_removal_are_preserved(tmp_path: Path) -> None:
    path = archive(tmp_path, "assets/table.dat")
    destination = tmp_path / "destination"
    DownloadManager.extract(path, str(destination), truncate_tarfile_path="assets/")
    assert (destination / "table.dat").read_bytes() == b"table data"


def test_network_exceptions_are_not_reclassified_in_this_step(tmp_path: Path) -> None:
    cause = requests.Timeout("request timed out")
    with (
        patch("elisa.managers.download_manager.requests.head", side_effect=cause),
        pytest.raises(requests.Timeout) as caught,
    ):
        DownloadManager.download_file("https://example.com/assets.tar", str(tmp_path / "download"))
    assert caught.value is cause


@pytest.mark.parametrize("failure", ["invalid", "staging", "publication", "none"])
def test_replacement_is_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "table.dat").write_bytes(b"old")
    (destination / "personal").write_bytes(b"keep")
    path = archive(tmp_path, "table.dat")
    if failure == "invalid":
        with tarfile.open(path, "a") as output:
            output.addfile(tarfile.TarInfo("../escape"))
    replace = download_manager.os.replace

    def publish(source: Path, target: Path) -> None:
        if Path(source).name == "staged":
            assert not destination.exists()
            raise OSError("publication")
        replace(source, target)

    if failure == "staging":
        monkeypatch.setattr(download_manager.shutil, "copyfileobj", Mock(side_effect=OSError("staging")))
    elif failure == "publication":
        monkeypatch.setattr(download_manager.os, "replace", publish)

    if failure == "none":
        DownloadManager.extract(path, str(destination))
        assert (destination / "table.dat").read_bytes() == b"table data"
    else:
        with pytest.raises((OSError, ElisaSettingsError)):
            DownloadManager.extract(path, str(destination))
        assert (destination / "table.dat").read_bytes() == b"old"
    assert (destination / "personal").read_bytes() == b"keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["assets.tar", "installed"]


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_unsafe_member_types(tmp_path: Path, kind: bytes) -> None:
    path = archive(tmp_path, "table.dat")
    with tarfile.open(path, "a") as output:
        member = tarfile.TarInfo("link")
        member.type = kind
        member.linkname = "table.dat"
        output.addfile(member)
    with pytest.raises(ElisaSettingsError):
        DownloadManager.extract(path, str(tmp_path / "installed"))
    assert not (tmp_path / "installed").exists()


def test_download_failure_cleans_archive(tmp_path: Path) -> None:
    temporary = tmp_path / "download"
    temporary.touch()
    with (
        patch.object(DownloadManager, "temp_file", return_value=str(temporary)),
        patch.object(DownloadManager, "download_file", side_effect=OSError("download")),
        pytest.raises(OSError, match="download"),
    ):
        DownloadManager._download_and_process("mock", str(tmp_path / "installed"))
    assert not temporary.exists()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_publication_failure_retains_original_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback_fails: bool,
) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "table.dat").write_bytes(b"original")
    path = archive(tmp_path, "table.dat")
    replace = download_manager.os.replace

    def fail_replace(source: Path, target: Path) -> None:
        if rollback_fails and Path(source) == destination:
            replace(source, target)
        else:
            raise OSError("rename failed")

    monkeypatch.setattr(download_manager.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failed"):
        DownloadManager.extract(path, str(destination))
    if rollback_fails:
        backups = list(tmp_path.glob(".installed-previous-*"))
        assert len(backups) == 1
        assert (backups[0] / "table.dat").read_bytes() == b"original"
        assert not destination.exists()
    else:
        assert (destination / "table.dat").read_bytes() == b"original"
    staging = [path for path in tmp_path.glob(".installed-*") if "-previous-" not in path.name]
    assert staging == []
