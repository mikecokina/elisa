import functools
import shutil
import tempfile
import requests
import tarfile

from tqdm import tqdm

from typing import Iterable
from logging import getLogger

logger = getLogger("elisa.managers.download_manager")


class DownloadManager(object):
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def temp_file():
        return tempfile.NamedTemporaryFile().name

    @staticmethod
    def extract(archive_path, destination_path, truncate_tarfile_path=""):
        def truncate_members(_archive) -> Iterable[tarfile.TarInfo]:
            truncate_length = len(truncate_tarfile_path)
            for member in _archive.getmembers():
                if member.path.startswith(truncate_tarfile_path):
                    member.path = member.path[truncate_length:]
                    yield member

        logger.info(f"extract files to {destination_path}")
        with tarfile.open(archive_path, "r") as archive:
            import os

            members = [x for x in truncate_members(archive)]

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(tar, path=".", members_=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members_, numeric_owner=numeric_owner)
                
            safe_extract(archive, destination_path, members_=members)

    @classmethod
    def _download_and_process(cls, url, destination_path, truncate_tarfile_path=""):
        temp_path = cls.temp_file()
        cls.download_file(url, temp_path)
        cls.extract(temp_path, destination_path, truncate_tarfile_path=truncate_tarfile_path)

    def download_ck04(self):
        url = "https://github.com/mikecokina/elisa-assets/raw/refs/heads/main/atmosphere/ck04.tar.gz"
        logger.info("downloading castelli-kurucz 2004 atmosphere files")
        self._download_and_process(url, self.settings.CK04_ATM_TABLES, truncate_tarfile_path="ck04/")

    def download_k93(self):
        url = "https://github.com/mikecokina/elisa-assets/raw/refs/heads/main/atmosphere/k93.tar.gz"
        logger.info("downloading kurucz 1993 atmosphere files")
        self._download_and_process(url, self.settings.K93_ATM_TABLES, truncate_tarfile_path="k93/")

    def download_van_hamme(self):
        url = "https://github.com/mikecokina/elisa-assets/raw/refs/heads/main/limbdarkening/ld_vh19.tar.gz"
        logger.info("downloading van hamme 2019 limb darkening files")
        self._download_and_process(url, self.settings.LD_TABLES, truncate_tarfile_path="ld/")

    def download_all(self):
        logger.info("Download manager is running...")

        self.download_van_hamme()
        self.download_ck04()
        self.download_k93()

    @staticmethod
    def handler(file_size, url, path):
        headers = {'Range': f'bytes=0-{file_size}'}

        # Make GET request and allow redirects
        r = requests.get(url, headers=headers, stream=True, allow_redirects=True)

        # Ensure response is successful
        if r.status_code >= 400:
            raise Exception(f"Failed to download file: {url} (Status Code: {r.status_code})")

        # Handle decompression if needed
        r.raw.read = functools.partial(r.raw.read, decode_content=True)

        # Download with tqdm progress bar
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc="Downloading") as r_raw:
            with open(path, "wb") as f:
                shutil.copyfileobj(r_raw, f)

    @classmethod
    def download_file(cls, url, path):
        # Allow redirects in HEAD request to get the final URL
        r = requests.head(url, headers={'Accept-Encoding': None}, allow_redirects=True)

        # Ensure response is successful
        if r.status_code >= 400:
            raise Exception(f"Failed to access URL: {url} (Status Code: {r.status_code})")

        # Get the final URL after redirections
        final_url = r.url

        # Get the file size
        file_size = int(r.headers.get('content-length', 0))  # Use .get() to avoid KeyError if missing

        if file_size == 0:
            raise Exception("Failed to determine file size. Content-Length missing.")

        cls.handler(file_size, final_url, path)
