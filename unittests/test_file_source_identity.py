"""Strict resolution preserves resource identities and revision detection."""
import errno
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from elisa.utils import file_source_identity


def previous_identity(path):
    resolved = Path(path).resolve()
    revision = resolved.stat()
    return (str(resolved), revision.st_dev, revision.st_ino, revision.st_size,
            revision.st_mtime_ns, revision.st_ctime_ns)


def test_existing_and_relative_paths(tmp_path, monkeypatch):
    source = tmp_path / "resource"
    source.write_bytes(b"first")
    monkeypatch.chdir(tmp_path)
    for path in (source, str(source), "resource", "./resource"):
        assert file_source_identity(path) == previous_identity(path)
        assert file_source_identity(path) == file_source_identity(source)


@pytest.mark.parametrize("name", ["missing", "missing/parent/resource"])
def test_missing_paths(tmp_path, name):
    for identify in (previous_identity, file_source_identity):
        with pytest.raises(FileNotFoundError) as error:
            identify(tmp_path / name)
        assert error.value.errno == errno.ENOENT


def test_replaced_file_with_preserved_size_and_mtime(tmp_path):
    source = tmp_path / "resource"
    source.write_bytes(b"first")
    before = file_source_identity(source)
    revision = source.stat()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"other")
    os.utime(replacement, ns=(revision.st_atime_ns, revision.st_mtime_ns))
    replacement.replace(source)
    after = file_source_identity(source)
    assert after == previous_identity(source)
    assert before != after
    assert before[3:5] == after[3:5]


def test_symlinks_and_retargeting(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"other")
    link = tmp_path / "link"
    link.symlink_to(first.name)
    chain = tmp_path / "chain"
    chain.symlink_to(link.name)
    directory = tmp_path / "directory"
    directory.symlink_to(tmp_path, target_is_directory=True)
    for path in (link, chain, directory / first.name):
        assert file_source_identity(path) == previous_identity(path)
        assert file_source_identity(path) == file_source_identity(first)
    before = file_source_identity(chain)
    link.unlink()
    link.symlink_to(second.name)
    assert file_source_identity(chain) == previous_identity(chain)
    assert file_source_identity(chain) == file_source_identity(second)
    assert file_source_identity(chain) != before
    second.unlink()
    for identify in (previous_identity, file_source_identity):
        with pytest.raises(FileNotFoundError):
            identify(chain)


def test_symlink_loop(tmp_path):
    link = tmp_path / "loop"
    link.symlink_to(link.name)
    with pytest.raises((OSError, RuntimeError)) as previous:
        previous_identity(link)
    with pytest.raises(type(previous.value)) as current:
        file_source_identity(link)
    assert getattr(current.value, "errno", None) == getattr(previous.value, "errno", None)


def test_only_one_explicit_stat(tmp_path):
    source = tmp_path / "resource"
    source.write_bytes(b"first")
    original_stat = Path.stat
    with patch.object(Path, "stat", autospec=True, side_effect=original_stat) as stat:
        assert file_source_identity(source) == previous_identity(source)
        # Only count the optimized helper, independently of pathlib's defaults.
        stat.reset_mock()
        file_source_identity(source)
        assert stat.call_count == 1
