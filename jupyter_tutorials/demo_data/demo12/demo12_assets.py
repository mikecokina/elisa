"""Filesystem layout and chain-cache helpers shared by the demo 12 refit scripts.

The MCMC chains are too large for the ELISA repository. They are published as ``.json.gz`` files in the
``elisa-assets`` repository and cached in :data:`CHAIN_CACHE_DIR`, which is ignored by Git. The notebook
``demo12-lc_fitting.ipynb`` downloads missing chains automatically; the scripts in this directory read the
same cache and write newly computed chains back into it.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

TUTORIALS_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = TUTORIALS_DIR / "demo_data"
DEMO_DIR = DATA_DIR / "demo12"
DATA_FILE = DATA_DIR / "lc_data" / "KIC04851217.lc.pulse_removed.dat"
ASSETS_DIR = DATA_DIR / "assets"
CHAIN_CACHE_DIR = DEMO_DIR / "chains"


def report(message: str) -> None:
    """Write one line of progress output.

    :param message: Text to write.
    :type message: str
    """
    sys.stdout.write(f"{message}\n")


def chain_archive(stage: str) -> Path:
    """Return the cached compressed chain of a stage.

    :param stage: Stage identifier, e.g. ``"12_1"``.
    :type stage: str
    :returns: Path of ``lc_mcmc_<stage>_chain.json.gz`` in the chain cache.
    :rtype: Path
    """
    return CHAIN_CACHE_DIR / f"lc_mcmc_{stage}_chain.json.gz"


def read_chain(archive: Path) -> dict[str, Any]:
    """Load a gzip-compressed JSON chain export.

    :param archive: Path of a ``.json.gz`` chain file.
    :type archive: Path
    :returns: Chain export with ``flat_chain``, ``chain_shape``, ``fitable`` and related metadata.
    :rtype: dict[str, Any]
    :raises FileNotFoundError: If the chain is not cached; run the demo 12 notebook to download it.
    """
    if not archive.is_file():
        msg = f"{archive} is missing. Run the demo 12 notebook once to download the chains into the cache."
        raise FileNotFoundError(msg)
    with gzip.open(archive, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def compress_chain(chain_file: Path) -> Path:
    """Compress a plain JSON chain next to itself and remove the plain file.

    The archive is written without a timestamp or file name so that identical content gives identical bytes.

    :param chain_file: Plain JSON chain written by ELISA.
    :type chain_file: Path
    :returns: Path of the created ``.json.gz`` archive.
    :rtype: Path
    """
    archive = chain_file.with_name(f"{chain_file.name}.gz")
    with archive.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        compressed.write(chain_file.read_bytes())
    chain_file.unlink()
    return archive
