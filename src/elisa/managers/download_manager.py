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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from elisa.conf.settings import Settings

logger = getLogger("elisa.managers.download_manager")

# HTTP status threshold used to detect client/server errors
HTTP_ERROR_STATUS = 400


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
        """

        def truncate_members(_archive: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
            truncate_length = len(truncate_tarfile_path)
            for member_ in _archive.getmembers():
                if truncate_tarfile_path and member_.path.startswith(truncate_tarfile_path):
                    member_.path = member_.path[truncate_length:]
                yield member_

        logger.info("extract files to %s", destination_path)
        with tarfile.open(archive_path, "r") as archive:
            members = list(truncate_members(archive))

            # Manual safe extraction: write members one by one and prevent
            # path traversal by resolving the target path.
            dest_path = Path(destination_path)
            dest_path.mkdir(parents=True, exist_ok=True)

            for member in members:
                target = dest_path / member.name
                try:
                    target_resolved = target.resolve()
                except Exception as err:  # pragma: no cover - defensive
                    msg = f"Invalid member path {member.name}: {err}"
                    raise RuntimeError(msg) from err

                if dest_path.resolve() not in target_resolved.parents and dest_path.resolve() != target_resolved:
                    msg = "Attempted Path Traversal in Tar File"
                    raise RuntimeError(msg)

                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.issym() or member.islnk():
                    # skip links for safety
                    continue
                else:
                    # ensure parent directory exists
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fobj = archive.extractfile(member)
                    if fobj is None:
                        continue
                    with target.open("wb") as out_f:
                        shutil.copyfileobj(fobj, out_f)

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
        :raises RuntimeError: On HTTP errors during GET.
        """
        headers = {}
        if file_size is not None and file_size > 0:
            # Range is inclusive, so request 0..file_size-1
            headers["Range"] = f"bytes=0-{file_size - 1}"

        # Make GET request and allow redirects (use timeout to avoid hangs)
        r = requests.get(url, headers=headers or None, stream=True, allow_redirects=True, timeout=30)

        if r.status_code >= HTTP_ERROR_STATUS:
            msg = f"Failed to download file: {url} (Status Code: {r.status_code})"
            raise RuntimeError(msg)

        # Handle decompression if needed
        r.raw.read = functools.partial(r.raw.read, decode_content=True)

        # Download with tqdm progress bar when file_size is known
        if file_size is not None and file_size > 0:
            with tqdm.wrapattr(r.raw, "read", total=file_size, desc="Downloading") as r_raw, Path(path).open("wb") as f:
                shutil.copyfileobj(r_raw, f)
        else:
            # Unknown size: stream without progress bar
            with Path(path).open("wb") as f:
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
        :raises RuntimeError: When HEAD request fails or GET fails.
        """
        # Issue a HEAD request to resolve redirects and attempt to read content-length
        r = requests.head(url, allow_redirects=True, timeout=30)

        if r.status_code >= HTTP_ERROR_STATUS:
            msg = f"Failed to access URL: {url} (Status Code: {r.status_code})"
            raise RuntimeError(msg)

        # Get the final URL after redirections
        final_url = r.url

        # Get the file size if available
        try:
            file_size = int(r.headers.get("content-length", 0)) or None
        except (TypeError, ValueError):
            file_size = None

        cls.handler(file_size, final_url, path)
