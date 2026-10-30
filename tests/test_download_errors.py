"""Download and extraction failures use their own ELISA exception category."""

import io
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from elisa.base.error import ElisaDownloadError, ElisaError, raise_elisa_download_error
from elisa.managers.download_manager import DownloadManager


def archive(tmp_path, member_name):
    path = tmp_path / "assets.tar"
    payload = b"table data"
    with tarfile.open(path, "w") as output:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    return str(path)


def test_download_error_inheritance_and_helper_cause():
    assert ElisaDownloadError.__bases__ == (ElisaError,)
    cause = OSError("original failure")
    with pytest.raises(ElisaDownloadError, match="download failed") as caught:
        raise_elisa_download_error("download failed", cause)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("status", [400, 500])
def test_failed_head_has_download_category(status, tmp_path):
    with patch("elisa.managers.download_manager.requests.head", return_value=Mock(status_code=status)):
        with patch.object(DownloadManager, "handler") as handler:
            with pytest.raises(ElisaDownloadError, match=f"Status Code: {status}"):
                DownloadManager.download_file("https://example.com/assets.tar", str(tmp_path / "download"))
    handler.assert_not_called()


@pytest.mark.parametrize("status", [400, 500])
def test_failed_get_has_download_category(status, tmp_path):
    destination = tmp_path / "download"
    with patch("elisa.managers.download_manager.requests.get", return_value=Mock(status_code=status)):
        with pytest.raises(ElisaDownloadError, match=f"Status Code: {status}"):
            DownloadManager.handler(None, "https://example.com/assets.tar", str(destination))
    assert not destination.exists()


def test_unsafe_archive_has_download_category(tmp_path):
    path = archive(tmp_path, "../outside.txt")
    with pytest.raises(ElisaDownloadError, match="Path Traversal"):
        DownloadManager.extract(path, str(tmp_path / "destination"))
    assert not (tmp_path / "outside.txt").exists()


def test_invalid_archive_path_retains_the_original_cause(tmp_path):
    path = archive(tmp_path, "table.dat")
    cause = OSError("cannot resolve path")
    with patch.object(Path, "resolve", side_effect=cause):
        with pytest.raises(ElisaDownloadError, match="Invalid member path table.dat") as caught:
            DownloadManager.extract(path, str(tmp_path / "destination"))
    assert caught.value.__cause__ is cause


def test_valid_extraction_and_prefix_removal_are_preserved(tmp_path):
    path = archive(tmp_path, "assets/table.dat")
    destination = tmp_path / "destination"
    DownloadManager.extract(path, str(destination), truncate_tarfile_path="assets/")
    assert (destination / "table.dat").read_bytes() == b"table data"


def test_network_exceptions_are_not_reclassified_in_this_step(tmp_path):
    cause = requests.Timeout("request timed out")
    with patch("elisa.managers.download_manager.requests.head", side_effect=cause):
        with pytest.raises(requests.Timeout) as caught:
            DownloadManager.download_file("https://example.com/assets.tar", str(tmp_path / "download"))
    assert caught.value is cause
