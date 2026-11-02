"""Operating-system failures must survive threaded table loading as fatal errors."""

import errno
from unittest.mock import Mock, patch

import pytest

from elisa.atm import atm
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaAtmosphereError, ElisaUnexpectedError


@pytest.mark.parametrize("stage", ["identity", "preflight", "read"])
@pytest.mark.parametrize("failure", [
    PermissionError(errno.EACCES, "access denied"),
    OSError(errno.EIO, "device error"),
    OSError(errno.EMFILE, "too many open files"),
])
def test_os_failures_are_unexpected_with_original_cause(tmp_path, stage, failure):
    path = tmp_path / "ckp00_5000_g40.csv"
    path.write_text("flux,wave\n1,10\n2,20\n")
    target, name = {
        "identity": (atm.utils, "file_source_identity"),
        "preflight": (atm.Path, "stat"),
        "read": (atm.pd, "read_csv"),
    }[stage]
    with patch.object(target, name, side_effect=failure):
        with pytest.raises(ElisaUnexpectedError, match=str(path)) as caught:
            if stage == "preflight":
                atm.multithread_atm_tables_reader_runner([str(path)])
            else:
                atm.read_unique_atm_tables([str(path)])
    assert caught.value.__cause__ is failure
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


def test_file_disappearing_during_read_remains_atmosphere_failure(tmp_path):
    path = tmp_path / "ckp00_5000_g40.csv"
    path.write_text("flux,wave\n1,10\n2,20\n")
    failure = FileNotFoundError(str(path))
    with patch.object(atm.pd, "read_csv", side_effect=failure):
        with pytest.raises(ElisaAtmosphereError) as caught:
            atm.read_unique_atm_tables([str(path)])
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("filename,contents,cause", [
    ("broken.csv", "flux,wave\n1,10\n", (ValueError, IndexError)),
    ("ckp00_5000_g40.csv", "", (ValueError,)),
    ("ckp00_5000_g40.csv", "flux,wave\ninvalid,10\n", (ValueError,)),
    ("ckp00_5000_g40.csv", "wrong,wave\n1,10\n", (KeyError,)),
    ("ckp00_5000_g40.csv", "flux,wave\n", (ValueError,)),
])
def test_invalid_installed_table_is_fatal_with_original_cause(tmp_path, filename, contents, cause):
    path = tmp_path / filename
    path.write_text(contents)
    with pytest.raises(ElisaUnexpectedError, match=str(path)) as caught:
        atm.read_unique_atm_tables([str(path)])
    assert isinstance(caught.value.__cause__, cause)
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("failure", [MemoryError("out of memory"), TypeError("reader bug")])
def test_reader_resource_and_programming_errors_remain_fatal(tmp_path, failure):
    path = tmp_path / "ckp00_5000_g40.csv"
    path.write_text("flux,wave\n1,10\n")
    with patch.object(atm.pd, "read_csv", side_effect=failure):
        with pytest.raises(ElisaUnexpectedError) as caught:
            atm.read_unique_atm_tables([str(path)])
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("failure", [
    ElisaAtmosphereError("candidate table unavailable"),
    ElisaUnexpectedError("broken reader contract"),
    KeyboardInterrupt(),
    ValueError("unclassified worker failure"),
    MemoryError("worker resource exhaustion"),
])
def test_runner_restores_worker_failure_policy(tmp_path, failure):
    path = tmp_path / "ckp00_5000_g40.csv"
    path.touch()
    preserved = isinstance(failure, (ElisaAtmosphereError, ElisaUnexpectedError, KeyboardInterrupt))
    expected = type(failure) if preserved else ElisaUnexpectedError
    with patch.object(atm, "_load_atm_container", side_effect=failure):
        with pytest.raises(expected) as caught:
            atm.multithread_atm_tables_reader_runner([str(path)])
    if preserved:
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure


def test_main_thread_interrupt_takes_priority_over_queued_worker_error(tmp_path):
    path = tmp_path / "ckp00_5000_g40.csv"
    path.touch()
    interruption = KeyboardInterrupt()
    queued = []

    def make_thread(*, target, args):
        thread = Mock()
        thread.start.side_effect = lambda: target(*args)
        thread.join.side_effect = interruption
        queued.append(args[1])
        return thread

    with (
        patch.object(atm, "Thread", side_effect=make_thread),
        patch.object(atm, "_load_atm_container", side_effect=ElisaAtmosphereError("missing model")),
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        atm.multithread_atm_tables_reader_runner([str(path)])
    assert caught.value is interruption
    assert not queued[0].empty()
