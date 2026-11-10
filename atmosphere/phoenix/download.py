# ruff: noqa: T201, INP001, S310, PLR2004, TRY301, C901, PLR0912, PLR0915
# Standalone CLI; urllib handles HTTPS archive URLs; retry and CLI branches stay explicit.
"""Acquire original PHOENIX v1.0 R10000 ZIP members using HTTP ranges."""

import argparse
import gzip
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://phoenix.astro.physik.uni-goettingen.de/data/v1.0/MedResFITS/R10000FITS/"
DATASET = "PHOENIX-ACES-AGSS-COND-2011"
MODEL = re.compile(r"lte(\d+)-([0-9.]+)([+-][0-9.]+)(?:\.Alpha=([+-][0-9.]+))?\.PHOENIX")


def request(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """Reject servers ignoring ranges rather than inadvertently fetching an atlas."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if start is not None:
                    if response.status != 206:
                        msg = "Server does not support byte ranges"
                        raise ValueError(msg)
                    if not response.headers["Content-Range"].startswith(f"bytes {start}-{end}/"):
                        msg = "Unexpected Content-Range"
                        raise ValueError(msg)
                data = response.read()
                if start is not None and len(data) != end - start + 1:
                    msg = "Truncated range"
                    raise ValueError(msg)
                return data
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    msg_0 = "Unreachable retry exhaustion"
    raise RuntimeError(msg_0)


class RemoteZip(io.RawIOBase):
    """Seekable, bounded HTTP reader used by the standard-library ZIP validator."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.position = url, size, 0

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        data = request(self.url, self.position, end - 1)
        self.position = end
        return data


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(path.suffix + ".part")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    stage.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload)
    stage.replace(path)


def discover(base: str) -> dict:
    listing = request(base).decode()
    archives = sorted(set(re.findall(r'href=["\']([^"\']+\.zip)["\']', listing)))
    records = []
    sizes = {}
    for name in archives:
        url = urllib.parse.urljoin(base, name)
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=90) as response:
            size = int(response.headers["Content-Length"])
        sizes[name] = size
        with zipfile.ZipFile(RemoteZip(url, size)) as archive:
            for member in archive.infolist():
                match = MODEL.search(member.filename)
                if not match or not member.filename.endswith(".fits"):
                    continue
                t, g, z, alpha = match.groups()
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    msg = "Unsafe archive member"
                    raise ValueError(msg)
                records.append(
                    {
                        "dataset": DATASET,
                        "version": "1.0",
                        "archive": name,
                        "archive_size": size,
                        "source_url": url,
                        "original_relative_path": member.filename,
                        "local_relative_path": member.filename,
                        "teff": int(t),
                        "logg": float(g),
                        "metallicity": float(z),
                        "alpha": float(alpha or 0),
                        "size": member.file_size,
                        "compressed_size": member.compress_size,
                        "crc32": member.CRC,
                    },
                )
    if not records:
        msg = "No spectra discovered"
        raise ValueError(msg)
    return {
        "base_url": base,
        "archive_sizes": sizes,
        "records": sorted(records, key=lambda r: (r["archive"], r["original_relative_path"])),
    }


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument("--index", type=Path, required=True, help="Complete upstream member inventory JSON")
    parser.add_argument("--refresh-index", action="store_true")
    for key, kind in [("teff", int), ("logg", float), ("metallicity", float), ("alpha", float)]:
        parser.add_argument("--" + key, type=kind, nargs="+")
    parser.add_argument("--all", action="store_true", help="Explicitly select entire available grid")
    parser.add_argument("--dry-run", action="store_true", help="Read index only; write no destination files")
    args = parser.parse_args()
    if args.refresh_index or not args.index.exists():
        inventory = discover(args.base_url)
        if not args.dry_run:
            save(args.index, inventory)
    else:
        inventory = json.loads(
            gzip.decompress(args.index.read_bytes()) if args.index.suffix == ".gz" else args.index.read_bytes(),
        )
    filters = {key: getattr(args, key) for key in ("teff", "logg", "metallicity", "alpha")}
    if not args.all and not any(filters.values()):
        parser.error("Specify parameter filters or --all")
    selected = [
        r for r in inventory["records"] if all(values is None or r[key] in values for key, values in filters.items())
    ]
    print(f"Selected {len(selected)} files; actual FITS bytes {sum(r['size'] for r in selected)}; "
          f"compressed payload bytes {sum(r['compressed_size'] for r in selected)} plus HTTP/index overhead")
    if not selected:
        parser.error("No available models match selection")
    for key, values in filters.items():
        for value in values or []:
            if not any(r[key] == value for r in selected):
                parser.error(f"Unavailable requested {key}={value}")
    if args.dry_run:
        return
    manifest_path = args.destination / "manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"records": []}
    records = {r["local_relative_path"]: r for r in prior["records"]}
    failures = 0
    for model in selected:
        record = dict(model)
        path = args.destination / record["local_relative_path"]
        previous = records.get(record["local_relative_path"], {})
        try:
            if path.exists() and path.stat().st_size == record["size"] and previous.get("sha256") == digest(path):
                record.update(sha256=previous["sha256"], status="verified")
                print(f"Verified {path.name}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                stage = path.with_suffix(path.suffix + ".part")
                with zipfile.ZipFile(RemoteZip(record["source_url"], record["archive_size"])) as archive:
                    # read() validates upstream ZIP CRC; partial members are safely retried whole.
                    data = archive.read(record["original_relative_path"])
                if len(data) != record["size"]:
                    msg = "Incorrect member size"
                    raise ValueError(msg)
                stage.write_bytes(data)
                record.update(sha256=digest(stage), status="verified")
                stage.replace(path)
                print(f"Downloaded {path.name}")
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            failures += 1
            record.update(status="failed", error=type(error).__name__)
            print(f"FAILED {record['original_relative_path']}: {error}")
        records[record["local_relative_path"]] = record
        save(
            manifest_path,
            {
                "dataset": DATASET,
                "version": "1.0",
                "scope": "acquired subset",
                "records": [records[k] for k in sorted(records)],
            },
        )
    if failures:
        msg = f"{failures} downloads failed"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
