# Fitting test data and system parameters

Both files are required by the browser tests. Tests read them without modifying
them and create temporary subsets for upload.

| File                      | Format                                                                               | Test input                                   |
|---------------------------|--------------------------------------------------------------------------------------|----------------------------------------------|
| `synthetic_bessell_U.dat` | 400 rows: time (JD − 2400000), normalized `Generic.Bessell.U` flux, flux uncertainty | 20 evenly spaced rows                        |
| `rv.json`                 | `primary` and `secondary` arrays, each containing 61 velocities in m/s               | Every fourth sample: 16 points per component |

## LC system

The light curve describes a detached, circular eclipsing binary with the
following generating model parameters.

| System parameter                           |                Value |
|--------------------------------------------|---------------------:|
| Orbital period                             |             2.5 days |
| Eccentricity                               |                    0 |
| Inclination                                |                  85° |
| Argument of periastron                     |                  58° |
| Mass ratio M₂/M₁                           | 1.2 / 2.15 ≈ 0.55814 |
| Primary minimum epoch                      |   JD 2454953.5388437 |
| Epoch in the observation file's time units |        54953.5388437 |
| Phase shift                                |                    0 |
| Systemic velocity                          |           −30.7 km/s |
| Distance                                   |               155 pc |
| Passband                                   |  `Generic.Bessell.U` |

| Stellar parameter     |          Primary |                          Secondary |
|-----------------------|-----------------:|-----------------------------------:|
| Mass                  |          2.15 M☉ |                             1.2 M☉ |
| Effective temperature |          10000 K |                             7000 K |
| Surface potential     |              3.6 |                                4.0 |
| Synchronicity         |                1 |                                  1 |
| Gravity darkening     |                1 |                                  1 |
| Albedo                |              0.6 |                                0.6 |
| Metallicity           |                0 |                                  0 |
| Atmosphere            | Blackbody (`bb`) | Configured atlas (`ck04` in tests) |

### LC fitting setup

| Parameter                       | Initial value | Mode        | Bounds / expression                        |
|---------------------------------|--------------:|-------------|--------------------------------------------|
| Inclination                     |           85° | Free        | 84–86°                                     |
| Primary effective temperature   |       10000 K | Free        | 9800–10200 K                               |
| Secondary effective temperature |        7000 K | Constrained | `0.7 * primary@t_eff`                      |
| Mass ratio - Community          |    1.2 / 2.15 | Fixed       | -                                          |
| Semi-major axis - Community     |       11.6 R☉ | Fixed       | Rounded value near the generating solution |
| Primary mass - Standard         |       2.15 M☉ | Fixed       | -                                          |
| Secondary mass - Standard       |        1.2 M☉ | Fixed       | -                                          |

Period, epoch, eccentricity, periastron argument, phase shift and stellar surface
parameters stay fixed at the values above; additional light is fixed at zero.
The temperature and inclination checks use the known model, while the short
MCMC chains are not required to recover its exact parameters.

## RV system

The velocities describe a circular binary with these reference parameters:

| System parameter  |                      Value |
|-------------------|---------------------------:|
| Orbital period    |                   4.5 days |
| Eccentricity      |                          0 |
| Mass ratio M₂/M₁  |                        0.5 |
| Systemic velocity |                    20 km/s |
| Phase grid        | `arange(-0.6, 0.62, 0.02)` |

Individual masses, inclination and the exact generating `a·sin(i)` are not
specified by this fixture. The projected semi-major axis below is a fit starting
value, not an asserted generating value. For the circular orbit, the test fixes
the periastron argument at 0°.

The tests convert velocities from m/s to km/s for the UI and assign
uncertainties of 0.5 km/s. Phase observations do not require an epoch.

### RV fitting setup

| Parameter                            |         Initial value | Mode        | Bounds / expression |
|--------------------------------------|----------------------:|-------------|---------------------|
| Projected semi-major axis `a·sin(i)` | 16.445535068958847 R☉ | Free        | 15–18 R☉            |
| Systemic velocity                    |               20 km/s | Free        | 18–22 km/s          |
| Eccentricity                         |                     0 | Constrained | `0 * system@asini`  |
| Mass ratio                           |                   0.5 | Fixed       | -                   |
| Orbital period                       |              4.5 days | Fixed       | -                   |
| Argument of periastron               |                    0° | Fixed       | -                   |

Only Community RV fitting is currently supported by the UI.

## Runtime settings

Each supported case has two free parameters. MCMC uses 10 walkers, 10 sampling
steps, zero burn-in and fixed noise parameter `ln_f=-20`. LS uses `max_nfev=3`.
LC calculations use 10-degree surface discretization. These settings validate
workflow execution, not posterior convergence.

Atmosphere and limb-darkening tables must be configured separately; see the
[setup guide](../README.md).
