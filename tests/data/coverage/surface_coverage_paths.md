`surface_coverage_paths.npz` records the unmodified `compute_surface_coverage`
from `feature/optimizations` commit `36c04301d9b804666caa5f1bbed9bf01f08c1457`.
It was generated from the original source preserved before the coverage refactor,
with the canonical `ElisaTestCase` settings and `PARAMS` from `test_face_visibility.py` (Astropy 2.0 constants).

Keys are `{position_index}_{component}`; values are full face coverage arrays
at semi-major axis 1. Positions, including conservative eclipse contact
boundaries and immediately exterior phases, are defined by
`coverage_positions()` in `test_surface_coverage_paths.py`. Face ordering and
zero coverage are part of the exact regression contract.

To regenerate deliberately against the original implementation, initialize
`ElisaTestCase.setUp()`, construct `BinarySystem.from_json(PARAMS)`, enumerate
`coverage_positions(binary)`, and evaluate each `positioned_container()` with
`semi_major_axis=1.0` and its specified `in_eclipse`. Save both returned arrays
using `numpy.savez_compressed`, then call the test case's `doCleanups()`.
