from __future__ import annotations

import functools
import os
import shutil
import tarfile
import tempfile
from contextlib import suppress
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from tqdm import tqdm

from elisa.base.error import raise_elisa_settings_error

if TYPE_CHECKING:
    from elisa.conf.settings import Settings

logger = getLogger(__name__)

# HTTP status threshold used to detect client/server errors
HTTP_ERROR_STATUS = 400


def _validated_members(
    archive: tarfile.TarFile, destination: Path, prefix: str,
) -> list[tuple[tarfile.TarInfo, Path]]:
    """Validate the entire archive before creating installation resources."""
    members = []
    for member in archive.getmembers():
        name = member.name.removeprefix(prefix) if prefix else member.name
        relative = Path(name)
        target = destination / relative
        try:
            resolved = target.resolve()
        except Exception as err:  # noqa: BLE001 - preserve the original cause
            raise_elisa_settings_error(f"Invalid member path {name}: {err}", err)
        if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(destination.resolve()):
            raise_elisa_settings_error("Attempted Path Traversal in Tar File")
        if not (member.isfile() or member.isdir()) or (relative == Path() and not member.isdir()):
            raise_elisa_settings_error(f"Invalid archive member: {member.name}")
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise_elisa_settings_error(f"Unsafe destination link: {name}")
        members.append((member, relative))
    return members


def _stage_dataset(
    archive: tarfile.TarFile,
    members: list[tuple[tarfile.TarInfo, Path]],
    destination: Path,
    stage: Path,
) -> None:
    """Build the complete replacement without writing to installed tables."""
    if destination.exists():
        # Keep unrelated user files and their metadata in the replacement.
        shutil.copytree(destination, stage, symlinks=True)
    else:
        stage.mkdir()
    for member, relative in members:
        target = stage / relative
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Validation above rejects existing symlink escapes as well.
        mode = target.stat().st_mode if target.exists() else None
        with archive.extractfile(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        if mode is not None:
            target.chmod(mode)


def _publish_dataset(stage: Path, destination: Path) -> None:
    """Publish a completed directory, restoring the previous one on failure.

    Portable rename cannot replace a nonempty directory in one step.
    Readers can see a missing root between this renames; they never
    see partially extracted tables. Concurrent installers are unsupported.
    """
    # Keep the backup outside staging cleanup so failed rollback cannot delete it.
    backup = destination.with_name(f".{destination.name}-previous-{stage.parent.name}")
    had_destination = destination.exists()
    if had_destination:
        destination.replace(backup)
    try:
        stage.replace(destination)
    except BaseException:
        if had_destination:
            _restore_dataset(backup, destination)
        raise
    else:
        if had_destination:
            shutil.rmtree(backup)


def _restore_dataset(backup: Path, destination: Path) -> None:
    """Attempt rollback; leave the original data recoverable if it fails."""
    try:
        backup.replace(destination)
    except OSError:
        logger.exception("Original dataset retained at %s", backup)


class DownloadManager:
    """Utility class to download and extract archived assets.

    The manager downloads remote tar.gz archives and extracts their
    contents into locations configured via a settings-like object.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def temp_file() -> str:
        """Create a temporary filename and return its path.

        Uses mkstemp to avoid leaving an open file handle. Caller is
        responsible for removing the file when appropriate.
        :returns: Path to a temporary file as a string.
        :rtype: str
        """
        fd, path = tempfile.mkstemp()
        try:
            # close the file descriptor; caller will write to the path
            os.close(fd)
        except OSError as err:  # pragma: no cover - defensive
            logger.debug("tempfile close failed: %s", err)
        return path

    @staticmethod
    def extract(archive_path: str, destination_path: str, truncate_tarfile_path: str = "") -> None:
        """Extract a tar archive to destination_path, optionally truncating leading path.

        This function performs a safety check to prevent path traversal when
        extracting tar members.

        :param archive_path: Path to the tar archive file.
        :type archive_path: str
        :param destination_path: Filesystem path where contents will be extracted.
        :type destination_path: str
        :param truncate_tarfile_path: If provided, strip this prefix from
            archive member paths before extraction.
        :type truncate_tarfile_path: str
        :returns: None
        :rtype: None
        :raises ElisaSettingsError: If an archive member path is invalid or escapes the destination.
        """
        logger.info("extract files to %s", destination_path)
        destination = Path(destination_path).absolute()
        with tarfile.open(archive_path, "r") as archive:
            members = _validated_members(archive, destination, truncate_tarfile_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
                stage = Path(temporary) / "staged"
                _stage_dataset(archive, members, destination, stage)
                _publish_dataset(stage, destination)

    @classmethod
    def _download_and_process(cls, url: str, destination_path: str, truncate_tarfile_path: str = "") -> None:
        temp_path = cls.temp_file()
        try:
            cls.download_file(url, temp_path)
            cls.extract(temp_path, destination_path, truncate_tarfile_path=truncate_tarfile_path)
        finally:
            # best-effort cleanup of temporary file
            with suppress(Exception):
                Path(temp_path).unlink()

    def download_ck04(self) -> None:
        url = "https://github.com/mikecokina/elisa-assets/raw/refs/heads/main/atmosphere/ck04.tar.gz"
        logger.info("downloading castelli-kurucz 2004 atmosphere files")
        self._download_and_process(url, self.settings.CK04_ATM_TABLES, truncate_tarfile_path="ck04/")

    def download_k93(self) -> None:
        url = "https://github.com/mikecokina/elisa-assets/raw/refs/heads/main/atmosphere/k93.tar.gz"
        logger.info("downloading kurucz 1993 atmosphere files")
        self._download_and_process(url, self.settings.K93_ATM_TABLES, truncate_tarfile_path="k93/")

    def download_van_hamme(self) -> None:
        url = "https://github.com/mikecokina/elisa-assets/raw/refs/heads/main/limbdarkening/ld_vh19.tar.gz"
        logger.info("downloading van hamme 2019 limb darkening files")
        self._download_and_process(url, self.settings.LD_TABLES, truncate_tarfile_path="ld/")

    def download_all(self) -> None:
        logger.info("Download manager is running...")

        self.download_van_hamme()
        self.download_ck04()
        self.download_k93()

    @staticmethod
    def handler(file_size: int | None, url: str, path: str) -> None:
        """Download the URL to path, optionally showing a progress bar when file_size is known.

        :param file_size: Expected size in bytes or ``None`` if unknown.
        :type file_size: int | None
        :param url: Final URL to download.
        :type url: str
        :param path: Destination file path.
        :type path: str
        :returns: None
        :rtype: None
        :raises ElisaSettingsError: On HTTP errors during GET.
        """
        headers = {}
        if file_size is not None and file_size > 0:
            # Range is inclusive, so request 0..file_size-1
            headers["Range"] = f"bytes=0-{file_size - 1}"

        # Make GET request and allow redirects (use timeout to avoid hangs)
        r = requests.get(url, headers=headers or None, stream=True, allow_redirects=True, timeout=30)

        if r.status_code >= HTTP_ERROR_STATUS:
            msg = f"Failed to download file: {url} (Status Code: {r.status_code})"
            raise_elisa_settings_error(msg)

        # Handle decompression if needed
        r.raw.read = functools.partial(r.raw.read, decode_content=True)

        # Download with tqdm progress bar when file_size is known
        if file_size is not None and file_size > 0:
            with tqdm.wrapattr(r.raw, "read", total=file_size, desc="Downloading") as r_raw, Path(path).open("wb") as f:
                # noinspection PyTypeChecker
                shutil.copyfileobj(r_raw, f)
        else:
            # Unknown size: stream without progress bar
            with Path(path).open("wb") as f:
                # noinspection PyTypeChecker
                shutil.copyfileobj(r.raw, f)

    @classmethod
    def download_file(cls, url: str, path: str) -> None:
        """Download a file given by URL to a local path.

        The method first issues a HEAD request to resolve redirects and to
        attempt to read Content-Length. If Content-Length is missing the
        file is downloaded without a progress bar.

        :param url: Source URL.
        :type url: str
        :param path: Destination path for the downloaded file.
        :type path: str
        :returns: None
        :rtype: None
        :raises ElisaSettingsError: When HEAD request fails or GET fails.
        """
        # Issue a HEAD request to resolve redirects and attempt to read content-length
        r = requests.head(url, allow_redirects=True, timeout=30)

        if r.status_code >= HTTP_ERROR_STATUS:
            msg = f"Failed to access URL: {url} (Status Code: {r.status_code})"
            raise_elisa_settings_error(msg)

        # Get the final URL after redirections
        final_url = r.url

        # Get the file size if available
        try:
            file_size = int(r.headers.get("content-length", 0)) or None
        except (TypeError, ValueError):
            file_size = None

        cls.handler(file_size, final_url, path)
