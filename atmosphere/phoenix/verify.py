# ruff: noqa: INP001, T201
"""Read an acquired subset with Astropy without altering original spectra."""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from download import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.destination / "manifest.json").read_text())
    for record in manifest["records"]:
        path = args.destination / record["local_relative_path"]
        if digest(path) != record["sha256"] or path.stat().st_size != record["size"]:
            msg = f"Checksum/size mismatch: {path.name}"
            raise ValueError(msg)
        with fits.open(path, checksum=True) as hdus:
            hdus.verify("exception")
            header, flux = hdus[0].header, hdus[0].data
            pixels = np.arange(flux.size) + 1
            # Legacy v1.0 headers encode natural-log air Angstroms, without CRPIX1.
            offset = (pixels - 1) * header["CDELT1"]
            wave = np.exp(header["CRVAL1"] + offset)
            if header["CTYPE1"] != "AWAV-LOG":
                msg = "Unexpected wavelength convention"
                raise ValueError(msg)
            if flux.ndim != 1 or not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
                msg = "Invalid dimensions or wavelength grid"
                raise ValueError(msg)
            for key, field in [
                ("PHXTEFF", "teff"),
                ("PHXLOGG", "logg"),
                ("PHXM_H", "metallicity"),
                ("PHXALPHA", "alpha"),
            ]:
                if not np.isclose(header[key], record[field]):
                    msg = f"Parameter mismatch: {key}"
                    raise ValueError(msg)

            print(path.name, "HDUs", len(hdus), "shape", flux.shape,
                  "wavelength", wave[0], wave[-1], "dln", header["CDELT1"],
                  "nonfinite", np.count_nonzero(~np.isfinite(flux)),
                  "zero", np.count_nonzero(flux == 0), "negative", np.count_nonzero(flux < 0))
            print(repr(header))


if __name__ == "__main__":
    main()
