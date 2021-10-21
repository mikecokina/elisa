import functools
import shutil
import tempfile
import requests
import tarfile

from tqdm import tqdm
from .. import settings

from typing import Iterable
from logging import getLogger

logger = getLogger("elisa.managers.download_manager")


class DownloadManager(object):
    def __init__(self):
        self.download_all()

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
            members = [x for x in truncate_members(archive)]
            archive.extractall(destination_path, members=members)

    def _download_and_process(self, url, destination_path, truncate_tarfile_path=""):
        temp_path = self.temp_file()
        self.download_file(url, temp_path)
        self.extract(temp_path, destination_path, truncate_tarfile_path=truncate_tarfile_path)

    def download_ck04(self):
        url = "http://astronomy.science.upjs.sk/elisa/assets/ck04.tar.gz"
        logger.info("downloading castelli-kurucz 2004 atmosphere files")
        self._download_and_process(url, settings.CK04_ATM_TABLES, truncate_tarfile_path="ck04/")

    def download_k93(self):
        url = "http://astronomy.science.upjs.sk/elisa/assets/k93.tar.gz"
        logger.info("downloading kurucz 1993 atmosphere files")
        self._download_and_process(url, settings.K93_ATM_TABLES, truncate_tarfile_path="k93/")

    def download_van_hamme(self):
        url = "http://astronomy.science.upjs.sk/elisa/assets/ld.tar.gz"
        logger.info("downloading van hamme 2016 limb darkening files")
        self._download_and_process(url, settings.LD_TABLES, truncate_tarfile_path="ld/")

    def download_all(self):
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
