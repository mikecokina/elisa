# PHOENIX v1.0 R10000 acquisition (stage 1)

Dataset: `PHOENIX-ACES-AGSS-COND-2011`, release `1.0`, product
`MedResFITS/R10000FITS`. This is an acquisition tool, independent of ELISA
runtime and installed scientific tables. No conversion is performed.

## Sources and provenance

- Original release: https://phoenix.astro.physik.uni-goettingen.de/data/v1.0/
- Exact product: https://phoenix.astro.physik.uni-goettingen.de/data/v1.0/MedResFITS/R10000FITS/
- Mirror: https://ftp.gwdg.de/pub/misc/phoenix/v1.0/
- Exact mirror product: https://ftp.gwdg.de/pub/misc/phoenix/v1.0/MedResFITS/R10000FITS/
- Author documentation: https://phoenix.astro.physik.uni-goettingen.de/spectra/
- Husser et al. (2013), *A new extensive library of PHOENIX stellar atmospheres
  and synthetic spectra*, A&A 553, A6, DOI:
  https://doi.org/10.1051/0004-6361/201219058

The author site requests citation of that paper. No explicit data redistribution
license or official cryptographic checksum list was found in the inspected
product listing or author documentation. Public download access is not evidence
of redistribution permission. Seek clarification from the authors before
redistributing spectra. No original binaries are stored in this repository.

Observed 2026-11-10: official ZIP requests redirect to the exact v1.0 GWDG
mirror. HTTPS range requests work there. Inventory URLs retain the original
archive identity; redirect handling is automatic.

## Actual inventory and grid gaps

`inventory.json.gz` is deterministic gzip-compressed JSON containing **the full
upstream member inventory**, not a download manifest. It was obtained by reading
all 51 ZIP central directories without downloading those ZIP payloads. Paths,
CRC32 values, compressed and original sizes are actual central-directory values;
archive sizes are actual HTTP Content-Length values. Inventory has 27,610 FITS
members. ZIP totals: **18,824,590,878 bytes**; uncompressed FITS totals:
**23,536,972,800 bytes**. Individual FITS members are 852,480 bytes. Full acquisition
by this tool transfers compressed members plus ZIP directory and HTTP overhead;
that overhead is not estimated as part of the reported payload total.

Observed union of available parameters:

| Parameter | Values |
| --- | --- |
| Teff, K | 2300–7000 in 100 K steps, 7200–12000 in 200 K steps (73 values) |
| log10(g / cm s^-2) | 0–6 in 0.5 steps |
| [M/H] | -4, -3, -2, -1.5, -1, -0.5, 0, +0.5, +1 |
| [alpha/M] | -0.2, 0, +0.2, +0.4, +0.6, +0.8, +1, +1.2 |

There are 51 metallicity/alpha pairs: nonzero alpha occurs only for metallicities
-3, -2, -1.5, -1, -0.5 and 0. Nonzero-alpha models cover 3500–8000 K,
except metallicity -3 where coverage starts at 4000 K. These bounds do not imply
complete grids. Relative to 73 temperatures × 13 gravities × 51 existing
composition pairs, **20,789 combinations are absent**. Even alpha=0 has gaps:
counts by metallicity -4 through +1 are respectively
828, 848, 848, 848, 846, 844, 841, 837, 819 (949 would be complete).
The exact available tuples and hence the exact missing combinations are encoded
in the inventory. Do not fabricate URLs from these parameter bounds. Filters
select available records; unmatched values and empty selections fail explicitly.
A multi-value filter selects the intersection of its value sets with actual
inventory records, and does not promise every requested Cartesian combination.

ZIP names follow
`PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z-0.0[.Alpha=+0.20].zip`.
Zero alpha omits the suffix. Signs may be URL encoded. Members are flat filenames,
for example `lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits`.
The `HiRes` member suffix is present **inside this medium-resolution product**;
it must not be used to infer the product's resolution.

## FITS observations and interpretation

Upstream documents the product as already resampled to R=10,000 over
3000–25000 Angstrom. No separate wavelength-grid file is needed.
The four acquired spectra each have one primary HDU, float32 flux array
`NAXIS1=212027`, and `BUNIT=erg/s/cm^2/cm`: emergent surface spectral flux
per centimetre of wavelength, as described by the paper. These are not fluxes
per Angstrom; originals are retained without unit conversion.

Actual legacy headers have `CTYPE1=AWAV-LOG`, `CRVAL1=8.006368`,
`CDELT1=1e-5`, and no `CRPIX1` or `CUNIT1`. Reconstruct air wavelengths in
Angstrom for zero-based array index i as:

```python
wavelength = numpy.exp(header['CRVAL1'] + i * header['CDELT1'])
```

This gives 3000.00129704954–24999.92240386198 Angstrom, constant natural-log
step 1e-5, approximately 0.030–0.250 Angstrom per pixel. It is a legacy log-value
encoding, not the modern FITS spectral WCS exponential algorithm applied to
linear CRVAL1. The author page describes two samples per resolution element,
but these actual files have approximately ten at R=10,000. Preserve this
observed discrepancy; instrumental resolution has not been independently
measured. `AWAV` identifies air wavelengths; the precise air conversion formula
is not established here.

Parameter fields `PHXTEFF`, `PHXLOGG`, `PHXM_H`, `PHXALPHA` match filenames
and inventory. Other provenance: `PHXEOS=ACES`, `PHXDUST=F`,
`PHXVER=16.01.00B`, `PHXBUILD=02/Aug/2010`, model calculation `DATE`,
microturbulence `PHXXI_L/M/N`, mass/radius/luminosity and mixing length.
The author documentation corrects header mass and luminosity units to grams
and erg/s (legacy cards incorrectly say kg and W). Some grid members can have
`INTERPOL` or `PHXCONV`; do not assume every model is directly computed.

Validation subset: (Teff, logg, [M/H], alpha) = (3000,4.5,0,0),
(5800,4.5,0,0), (10000,4.5,0,0), (5800,2.5,-1,0). All four had finite,
strictly increasing wavelengths, matching metadata, and no non-finite, zero,
or negative flux values. This observation is not a positivity requirement for
the full grid. No cleaning or modification occurs.

## Use and reconstruction

Downloader requires Python >=3.11, standard library only. FITS verification
requires NumPy and Astropy in your chosen environment (for example
`python3 -m pip install numpy astropy`). No ELISA configuration is used.
Run from repository root. Choose an isolated destination; local layout is flat
original FITS filenames plus `manifest.json`. `atmosphere/phoenix/data/` is
ignored if a repository-local destination is desired.

Reconstruct the validated subset from scratch with these two commands:

```sh
python3 atmosphere/phoenix/download.py --index atmosphere/phoenix/inventory.json.gz --destination /tmp/elisa-phoenix-stage1 --teff 3000 5800 10000 --logg 4.5 --metallicity 0 --alpha 0
python3 atmosphere/phoenix/download.py --index atmosphere/phoenix/inventory.json.gz --destination /tmp/elisa-phoenix-stage1 --teff 5800 --logg 2.5 --metallicity=-1 --alpha 0
python3 atmosphere/phoenix/verify.py /tmp/elisa-phoenix-stage1
```

Explicit complete-grid acquisition (not executed during development):

```sh
python3 atmosphere/phoenix/download.py --index atmosphere/phoenix/inventory.json.gz --destination /path/to/archive --all
```

Append `--dry-run` to inspect counts and actual sizes without writing destination
files. Use `--refresh-index` to discover current upstream members; use a separate
index path to preserve the historical inventory. `--base-url` can select the
exact mirror product directory during discovery; existing inventory URLs take
precedence unless refreshed. Missing indices are discovered automatically.
Discovery performs network reads; a dry run never writes even a new index.

Resume by rerunning the same acquisition command. Files with recorded SHA-256
and size matching local content are skipped. Corruption triggers a fresh member
fetch. Interrupted `.fits.part` files are retried whole, validated with the ZIP
CRC, then atomically renamed. Unverified existing files are fetched again.
Manifests are updated atomically after each member; a crash between spectrum
publication and manifest publication causes a safe refetch. Run one writer per
destination. Failed members are reported, recorded, and cause nonzero exit.
Original ZIP containers are not retained; original FITS bytes are preserved.

`verify.py` checks SHA-256, size, FITS structure, parameters, wavelength ordering
and reports exceptional flux counts and headers without changing data. The
checked-in `subset-manifest.json` pins hashes of the four acquired originals;
the destination manifest describes only files acquired there, independent of
the complete inventory. Neither manifest contains absolute local paths.
For byte-for-byte reconstruction compare reconstructed SHA-256 records against
`subset-manifest.json`. SHA-256 establishes subsequent local integrity, not
upstream authenticity: no official SHA-256 values were available. ZIP CRC32
checks transfer/decompression integrity but is not a cryptographic guarantee.

Development validation used an isolated `/tmp/elisa-phoenix-stage1` directory:
first acquisition, repeated acquisition with unchanged mtime, same-size
corruption detection/replacement, abandoned partial recovery with identical
SHA-256, FITS readability and metadata checks, and full-grid dry-run without
destination creation. No full atlas or scientific tests were run.

## Stage 2 boundary

Retain release, product, composition, original filenames, model headers and
hashes when comparing with future `v1.0/LimbDarkFITS` products. Shared release
identity alone does not establish compatible parameters, abundance conventions,
wavelengths or model physics. No LD files were acquired or integrated here.
