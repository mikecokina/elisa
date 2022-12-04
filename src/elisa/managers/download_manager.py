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
        url = "http://astronomy.science.upjs.sk/elisa/assets/ck04.tar.gz"
        logger.info("downloading castelli-kurucz 2004 atmosphere files")
        self._download_and_process(url, self.settings.CK04_ATM_TABLES, truncate_tarfile_path="ck04/")

    def download_k93(self):
        url = "http://astronomy.science.upjs.sk/elisa/assets/k93.tar.gz"
        logger.info("downloading kurucz 1993 atmosphere files")
        self._download_and_process(url, self.settings.K93_ATM_TABLES, truncate_tarfile_path="k93/")

    def download_van_hamme(self):
        url = "http://astronomy.science.upjs.sk/elisa/assets/ld.tar.gz"
        logger.info("downloading van hamme 2016 limb darkening files")
        self._download_and_process(url, self.settings.LD_TABLES, truncate_tarfile_path="ld/")

    def download_all(self):
        logger.info("Download manager is running...")

        self.download_van_hamme()
        self.download_ck04()
        self.download_k93()

    @staticmethod
    def handler(file_size, url, path):
        # specify the starting and ending of the file
        headers = {'Range': 'bytes=%d-%d' % (0, file_size)}

        # request the specified part and get into variable
        r = requests.get(url, headers=headers, stream=True)

        # open the file and write the content of the requested content into file
        r.raw.read = functools.partial(r.raw.read, decode_content=True)  # decompress if needed
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc="") as r_raw:
            with open(path, "wb") as f:
                shutil.copyfileobj(r_raw, f)

    @classmethod
    def download_file(cls, url, path):
        r = requests.head(url, headers={'Accept-Encoding': None})
        file_size = int(r.headers['content-length'])
        cls.handler(file_size, url, path)
