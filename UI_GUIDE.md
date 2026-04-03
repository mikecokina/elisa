# ELISa Desktop App - User Guide

Welcome to **ELISa**! This guide will help you get started using the interactive desktop application to model and
analyze eclipsing binary star systems.

---

## What is ELISa?

ELISa is an interactive tool for **modeling and analyzing eclipsing binary star systems** - pairs of stars that orbit
each other and periodically eclipse one another.

### What You Can Do

With ELISa, you can:

- **Generate synthetic light curves and radial velocity measurements** - simulate what observations of a binary star
  system would look like
- **Visualize binary star systems** - see 3D representations of stars, their surfaces, and how they interact
- **Fit observational data** - automatically adjust model parameters to match real observations using advanced fitting
  methods
- **Analyze surface features** - model starspots and pulsations
- **Export results** - save your analysis in standard formats

### Real-World Example

Imagine you've observed a distant eclipsing binary star system and have measurements of its light (how bright it appears
over time). ELISa helps you:

1. Create a model of the two stars (mass, temperature, size, etc.)
2. Simulate what the light curve should look like
3. Automatically fit your model to match your observations
4. Extract physical properties of the stars from the best-fit model

---

## Getting Started

### Starting the App

**On Linux/Mac:**

```bash
python -m elisa.ui
```

**On Windows:**

```bash
python -m elisa.ui
```

The app will open in your web browser at `http://localhost:7860`. You should see the ELISa interface with multiple tabs
at the top.

### First Time Setup

On first launch, ELISa will create a configuration file. This is automatic - just follow any prompts that appear. You
don't need to do anything special; the defaults are suitable for most users.

---

### Environment Variables

Several aspects of the UI can be tuned without editing any code by setting environment variables **before** launching
the app. All variables are optional - the defaults work well for most setups.

| Variable                   | Default                                          | Accepted values                            | What it controls                                 |
|----------------------------|--------------------------------------------------|--------------------------------------------|--------------------------------------------------|
| `ELISA_UI_SERVER_PORT`     | 7860 (default when running `python -m elisa.ui`) | Any valid port number, e.g. `7860`         | TCP port the app listens on                      |
| `ELISA_UI_SERVER_HOST`     | `127.0.0.1` (localhost only)                     | IP address or `0.0.0.0` for all interfaces | Network interface the app binds to               |
| `ELISA_UI_THEME`           | `light`                                          | `light`, `dark`, `system`                  | Color theme of the interface                     |
| `ELISA_UI_MAX_SPOTS`       | `3`                                              | Any positive integer                       | Max number of spots per star in the UI           |
| `ELISA_UI_MAX_PULSE_MODES` | `3`                                              | Any positive integer                       | Max number of pulsation modes per star in the UI |

**How to set them (Linux / Mac):**

```bash
export ELISA_UI_SERVER_PORT=7860
export ELISA_UI_THEME=dark
export ELISA_UI_MAX_SPOTS=5
python -m elisa.ui
```

**How to set them (Windows Command Prompt):**

```bat
set ELISA_UI_SERVER_PORT=7860
set ELISA_UI_THEME=dark
set ELISA_UI_MAX_SPOTS=5
python -m elisa.ui
```

> ⚠️ **Performance note for `ELISA_UI_MAX_SPOTS` and `ELISA_UI_MAX_PULSE_MODES`:**
> Every spot and pulsation mode slot is rendered in the browser even when it is not active.
> Raising these values above the defaults will make the interface noticeably slower to load
> and interact with. Increase them only if you genuinely need more than 3 spots or modes per star.

---

## Main Features & Tabs

The app has **5 main sections** (tabs) at the top:

### 1. **Light Curve Modeling**

Generate synthetic light curves for a binary star system.

**What it does:**

- You set the properties of two stars (mass, temperature, size)
- You define the orbit (how they circle each other)
- The app calculates what the light curve would look like as the stars eclipse each other

**When to use:**

- Testing different star configurations
- Learning how stellar properties affect observations
- Creating synthetic data for teaching or validation

---

### 2. **Radial Velocity Modeling**

Generate synthetic radial velocity measurements (how fast the stars move toward or away from us).

**What it does:**

- Like the light curve tool, but shows the motion of the stars instead of brightness changes
- Two computation methods are available via a dropdown:
    - **Kinematic** (default, fast) - models only the center-of-mass velocity of each star; best for most systems
    - **Radiometric** (slower) - integrates flux across every visible surface element; needed to model the
      Rossiter-McLaughlin effect, spot-induced RV variations, or pulsation effects on RV shape

**When to use:**

- Modeling systems where you have spectroscopic (velocity) measurements
- Understanding the orbital dynamics of binary systems
- Simulating the Rossiter-McLaughlin effect (use Radiometric method)

---

### 3. **Light Curve Fitting**

**Automatically fit a model to your observed light curve data.**

This is the main "inverse problem" tool - if you have real observations, use this to find the best-matching star
properties.

**Step-by-step workflow:**

1. **Upload your data**
    - Click the upload area and select one or more light curve data files
    - For each file, specify the filter/passband (e.g., U, V, I passbands)
    - Choose whether your data is in flux units or magnitude
    - If using magnitude, enter the reference magnitude for normalization

2. **Choose a parameter format**
    - **Standard** - specify individual masses for the primary and secondary star
    - **Community** - specify `semi_major_axis` and `mass_ratio` instead of masses (common in the eclipsing binary
      literature)
    - Both produce equivalent physical models; choose whichever matches your prior knowledge

3. **Choose system morphology**
    - **Detached** - the two stars don't fill their Roche lobes; two distinct, well-separated eclipses
    - **Over-contact** - the stars share a common envelope; shallow, merged eclipses with characteristic shape

4. **Set initial guesses for star and orbital properties**
    - Primary star: temperature, surface potential, synchronicity, etc.
    - Secondary star: same parameters
    - Orbital parameters: inclination, period, eccentricity, distance

5. **Set bounds and constraints for each parameter**
    - **Free** - fitted freely within a `min` and `max` bound you specify
    - **Fixed** - held constant at the initial value
    - **Constrained** - derived from a mathematical expression referencing other parameters, e.g.:
      `16.5 / sin(radians(system@inclination))`
      where `system@inclination` refers to the current value of the inclination parameter

6. **(Optional) Add spots to fit**
    - Expand the spot section to add one or more starspots with fittable position and temperature

7. **Choose a fitting method**
    - **Least Squares (Fast)**: Quick, deterministic fit. Best for initial exploration.
    - **MCMC (Thorough)**: Slower but more complete. Gives confidence intervals and full posterior.

8. **Run the fit**
    - Click "Start Fit"
    - Watch the terminal output to see progress
    - For MCMC, this may take hours, depending on complexity

9. **Review results**
    - View the best-fit parameters and their uncertainties
    - See plots comparing your data to the model
    - For MCMC: view correlation plots and trace diagnostics

10. **Download your results**
    - Save the results as a JSON file for later use or publication

**Tips:**

- Start with a **Least Squares fit** to get a quick baseline
- If satisfied, transfer that result as the starting point for MCMC
- Make sure your initial guesses are reasonably close to the true values
- Fix parameters you're confident about to speed up fitting

---

### 4. **Radial Velocity Fitting**

Works similarly to light curve fitting, but for radial velocity (RV) data.

**Differences from light curve fitting:**

- Upload RV data files (one for the primary star, optionally one for secondary)
- Fewer parameters typically need adjustment
- Fitting supports only the kinematic / center-of-mass model (the default "Kinematic" method): the fitter
  compares data to the stars' center-of-mass Doppler velocities. It does NOT account for surface-integrated
  effects such as the Rossiter-McLaughlin effect, spot-induced RV variations, or pulsation-induced
  line-profile effects. Those phenomena require the radiometric / surface-integrated model available in the
  "Radial Velocity Modeling" tab but are not supported by the automatic fitting routines.
- Usually very fast to fit since there's less data and the model is less complex

**Same workflow:**

1. Upload data
2. Set initial parameters
3. Choose fitting method
4. Run fit
5. Review and download results

---

### 5. **System Visualization**

**Create visualizations** of your binary system across five different modes.

**Visualization modes (select one before clicking Visualize):**

- **surface** - 3D colored surface model of one or both stars at a given phase; most detailed view
- **mesh** - 3D dot-surface preview (surface points) at a given phase
- **wireframe** - 3D wireframe plotter showing triangulation of the stellar surfaces at a given phase
- **orbit** - 2D top-down plot of the orbital path
- **equipotential** - cross-section plot of Roche equipotential surfaces

**Surface colormaps** (available in `surface` mode):

- `temperature` - surface temperature distribution
- `gravity_acceleration` - local gravitational acceleration
- `velocity` - total surface velocity
- `radial_velocity` - velocity component along the line of sight
- `radiance` / `normal_radiance` - emitted intensity
- `radius` - distance from star center

**How to use:**

1. Select a visualization mode from the dropdown
2. Set the star and orbital parameters
3. (Optionally) Add starspots by specifying location and temperature
4. (Optionally) Add pulsations using spherical harmonics
5. Choose which components to show (both / primary only / secondary only)
6. For `surface` mode: choose a colormap and set the viewing elevation and azimuth
7. For `wireframe` mode: set the viewing elevation and azimuth (colormap is not used)
8. Click "Visualize"

**Tips:**

- Experiment with different phases (e.g., -0.5, 0.0, 0.25, 0.5) to watch eclipses unfold
- Use `orbit` mode first to verify the orbital geometry is correct
- Use `equipotential` to check whether the system is detached, semi-detached, or over-contact
- Use `mesh` when you want a quick point-based surface preview
- Use `wireframe` when you want to inspect triangulation and surface shape without colormap shading
- In ELISa, discretization is an angular step in degrees: lower values mean denser surfaces and more detail; higher
  values mean sparser surfaces and faster computation
- Use higher discretization (e.g., 10+) for fast preview, lower (e.g., 3-5) for publication renders
- Use `surface` + `temperature` colormap to verify spot placement visually before fitting

---

## Common Workflows

### Workflow 1: Quick Visual Exploration

*Goal: See what different binary star configurations look like*

1. Go to **Light Curve Modeling**
2. The default values are already filled in
3. Click "Generate Light Curve"
4. Explore the plot
5. Try changing parameters and re-generating to see the effect

### Workflow 2: Fit Your Observations

*Goal: Extract star properties from observed data*

1. Go to **Light Curve Fitting**
2. Upload your data file(s)
3. Leave the default parameters as starting guesses (or enter your own estimates)
4. Click "Run Least Squares Fit"
5. Review the results
6. (Optional) Run MCMC for complete uncertainty analysis
7. Download the JSON results

### Workflow 3: Validate a Model

*Goal: Check if your fitted model matches observations visually*

1. Go to **System Visualization**
2. Enter the fitted parameters from your analysis
3. Select `surface` mode and a colormap (e.g., `temperature`)
4. Generate and explore the model at different phases
5. Use `equipotential` mode to confirm detached / over-contact morphology

### Workflow 4: Fit a Spotty Binary

*Goal: Model light curve asymmetries caused by stellar spots*

1. Go to **Light Curve Fitting**
2. First fit the clean binary model (no spots) with Least Squares
3. Inspect residuals - persistent asymmetry suggests a spot
4. Expand the spot section and add one spot per suspected feature
5. Set initial longitude, latitude, angular radius, and temperature factor
6. Run MCMC to sample the spot parameters along with orbital parameters
7. Use **System Visualization** with the fitted spot to verify placement visually

---

## Understanding the Parameters

The app uses different parameters depending on which tab you're in. Here's a complete reference of all available
parameters organized by category.

---

### **Primary & Secondary Star Parameters**

These parameters describe the properties of each star in the binary system. They appear in all tabs (Modeling and
Fitting). Each star has identical parameter options.

#### **Mandatory Star Parameters**

**Mass (M☉)**

- How heavy the star is (in units of our Sun's mass)
- Affects the star's size and temperature via physics
- Typical range: 0.1 to 100 solar masses
- Must be fitted or fixed in Fitting tabs

**Effective Temperature (T_eff) [K]**

- Surface temperature in Kelvin
- Hotter stars are bluer; cooler stars are redder
- Typical ranges: 3,500 K (red dwarf) to 50,000 K (hot O-type)
- Default typically 9,500 K for a main-sequence star

**Surface Potential (Ω)**

- Controls the size and shape of the star via the Roche-lobe potential
- Higher values = smaller, more compact star; lower values = larger, more inflated star
- Must be above the critical (Roche-lobe) potential for the star to stay within its lobe
- Typical range: 3.0 to 10.0 - detached systems often use 3.5 to 6.0; over-contact systems typically need 7.0 to 10.0
- The app will report an error if the value you enter is below the critical potential for the given mass ratio and
  separation

**Synchronicity (F)**

- Rotation-to-orbital frequency ratio
- 1.0 = star rotates at the same rate as it orbits (synchronous rotation, most common)
- Values near 1.0 typical for eclipsing binaries
- Range: 0.1 to 10.0

#### **Optional Star Parameters**

**Gravity Darkening (β)** [0, 1]

- Describes how surface temperature and brightness vary with local gravity across the star
- Lower gravity regions are cooler and dimmer, higher gravity regions are hotter and brighter
- Typical values: ~1.0 for radiative envelopes, ~0.2–0.32 for convective envelopes
- Leave empty to auto-infer from stellar properties
- Affects surface brightness distribution and light curve shape, especially in distorted or close binary systems

**Albedo (A)** [0, 1]

- Fraction of incident radiation that is reprocessed and re-emitted by the stellar surface
- 0 = absorbs all incident energy, 1 = fully re-emits incident energy
- Typical values: ~1.0 for radiative envelopes, ~0.5 for convective envelopes
- Leave empty to auto-infer from stellar properties
- Affects mutual irradiation ("reflection effect") in close binary systems and influences the light curve

**Metallicity [M/H]**

- Logarithmic measure of metal abundance relative to the Sun
- 0.0 = solar metallicity, negative = metal-poor, positive = metal-rich
- Typical range: -2.0 to +1.0
- Affects opacity, atmospheric structure, and emergent spectrum

**Discretization Factor [degrees]**

- Controls how finely the star's surface is divided into mesh elements
- In ELISa, this value is an angular step in degrees between neighboring surface points
- Lower values = smaller average angle = denser surface sampling = more detail but slower
- Higher values = larger average angle = sparser surface sampling = less detail but faster
- Typical range: 3 to 10 degrees
- Leave empty to use ELISa default

**Atmosphere Model**

- Type of stellar atmosphere used for intensity calculations and limb darkening
- Three options are available:
    - `ck04` **(default)** - Castelli & Kurucz (2004) model; realistic physics, valid for ~3,500-50,000 K
    - `k93` - Kurucz (1993) model; similar to ck04, alternative grid
    - `bb` - blackbody approximation; fast, simpler, no metallicity or gravity dependence; useful for very hot stars (
      T_eff > 50,000 K) or quick testing
- Limb darkening coefficients are automatically interpolated from the chosen model
- If the star's temperature is outside the model's supported range, switch to `bb`

---

### **Orbital & System Parameters**

These define how the two stars orbit each other and what the system looks like from Earth.

**Inclination (i) [degrees]**

- Viewing angle of the orbital plane
- 0° = pole-on (we see the star face-on, no eclipse)
- 90° = edge-on (we see the orbit edge-on, maximum eclipse)
- For eclipsing binaries, must be close to 90° (typically 80-90°)

**Orbital Period (P) [days]**

- Time for one complete orbit
- Determines how frequently eclipses occur
- Typical range: 0.01 to 1,000 days
- Usually fixed in Fitting tabs (known from observations)

**Eccentricity (e)**

- How elliptical the orbit is
- 0.0 = perfectly circular orbit (most common for close binaries)
- Values > 0 = elliptical orbits
- Typical range: 0.0 to 0.5
- Allowed range: 0.0 to 1.0 (typical values: 0.0 to 0.5)

**Argument of Periastron (ω) [degrees]**

- Orientation of the elliptical orbit in space
- Only matters if eccentricity > 0
- Range: 0° to 360°

**Primary Minimum Time (T₀) [Julian days]**

- Time of the first primary eclipse (when the primary star is in front)
- Very large number (e.g., 2,440,000+) representing the date
- Used as reference point for all orbital phases
- Usually fixed in Fitting tabs

**Systemic Velocity (γ) [km/s]**

- Radial velocity of the system’s center of mass relative to the observer
- Shifts the entire radial velocity curve up or down by a constant offset
- Primarily important for radial velocity modeling; does not affect light curve shape
- Optional in light curve modeling (leave empty to default to 0.0 km/s)
- Typical range: -200 to +200 km/s

**Semi-major Axis (a) [R☉]**

- Orbital separation between the centers of the two stars (semi-major axis of the relative orbit, in solar radii)
- Sets the physical scale of the system (together with orbital period via Kepler’s third law)
- Available in the "Community" LC fitting format as an alternative to specifying individual masses
- Typical range: 5 to 30 solar radii (depends strongly on period and masses)

**Mass Ratio (q = M₂/M₁)**

- Ratio of secondary mass to primary mass
- q = 1.0 means equal-mass binary; q < 1.0 means secondary is less massive
- Strongly constrained by radial velocity data (especially double-lined systems)
- Available in the "Community" LC fitting format instead of individual masses

**Distance (d) [parsecs]**

- Distance of the binary system from the observer
- Scales the absolute observed flux, but does not affect the shape of normalized light curves
- Typical range: 10 to 10,000 parsecs
- Not required if the light curve is normalized

**Additional Light (l₃)**

- Fraction of total observed light that does not originate from the binary (e.g. third star or background source)
- 0.0 = all light comes from the binary; higher values indicate additional flux contribution
- Typical range: 0.0 to 0.5
- Reduces (dilutes) eclipse depths without changing their timing

**Phase Shift (Δφ)**

- Offset applied to the orbital phase (e.g. if primary eclipse is not at phase 0)
- Useful for aligning model with observed data timing
- Typical range: -0.5 to 0.5
- Usually fixed at 0.0 unless fitting phase offsets

---

### **Observation & Data Parameters**

These parameters control what you're measuring and how to display it.

#### **Light Curve Observation Parameters**

**Passband / Filter**

- The wavelength range being observed (e.g., U, V, I, Gaia G, RP, BP)
- Different filters show different features of eclipses
- Must be specified when uploading light curve data
- Common choices: Generic Bessell bands (U, V, R, I), Gaia bands, TESS, etc.

**From Phase / To Phase**

- Start and end of the orbital phase range to model
- Phase 0.0 = primary eclipse, 0.5 = secondary eclipse, 1.0 = back to primary
- Typical range: -0.5 to 0.5 (centered on primary eclipse)
- Can use any range to focus on specific region

**Phase Step**

- Sampling resolution: how finely the light curve is computed
- Smaller values = more points = finer resolution but slower
- Typical: 0.001 to 0.01
- Value 0.01 gives ~100 points per orbit

**Normalize**

- Whether to rescale light curve to maximum brightness = 1.0
- When enabled, the Distance parameter is not needed
- Useful for comparing systems at different distances

#### **Radial Velocity Parameters**

**a·sin(i) [R☉]**

- Product of semi-major axis and sine of inclination
- Directly measurable from Doppler shifts
- Typical range: 5 to 20 solar radii
- Constrains orbit size without needing individual masses

**Data Units**

- **Flux**: Relative brightness (1.0 = normalized, values < 1.0 = dimmed)
- **Magnitude**: Logarithmic scale (higher numbers = fainter)
- Must match your data file format

**Reference Magnitude** (when using magnitude units)

- The magnitude corresponding to flux = 1.0
- Only used when data is in magnitude units
- Needed to convert magnitude back to flux internally

**X-axis Units** (for fitting data)

- **Julian Days (JD)**: Absolute calendar dates (large numbers like 2,459,000)
- **Phases (dimensionless)**: Orbital phase (-0.5 to 1.5, centered on primary eclipse)
- Phase is often more convenient for fitting

---

### **Surface Features Parameters**

Used in Modeling and System Visualization tabs to add realistic details.

#### **Starspots**

Up to **3 spots** can be defined per star (default). This limit can be raised by setting the `ELISA_UI_MAX_SPOTS`
environment variable before launching the app, though increasing it above the default will slow down the interface. Each
spot has 4 properties:

**Longitude [degrees]**

- East-West position on the star's surface when generated in `[0, 0]` of coordinate system
- Range: 0° to 360°
- 0° = facing vector `[1, 0, 0]` in coordinate system, 180° = far side

**Latitude [degrees]**

- North-South position on the star's surface when generated in `[0, 0]` of coordinate system
- Range: 0° to 90° (pole to equator)
- 0° = north pole, 90° = equator

**Angular Radius [degrees]**

- Size of the spot (angular size as seen from the star's center)
- Typical range: 5° to 80° - small spots are 5-15°, large spots can exceed 50°
- Larger values = bigger spot; very large spots can cover significant fraction of the visible disk

**Temperature Factor**

- Ratio of spot temperature to surrounding photosphere temperature (T_spot / T_star)
- Values < 1.0 = cool (dark) spot; values > 1.0 = hot (bright) spot
- Cool spots are most common (e.g., sunspots at ~0.85); hot spots (faculae) also occur
- Typical range: 0.8 to 1.05
- Affects light curve depth and shape

#### **Pulsation Modes**

Up to **3 pulsation modes** can be defined per star (default). This limit can be raised by setting the
`ELISA_UI_MAX_PULSE_MODES` environment variable before launching the app, though increasing it above the default will
slow down the interface. Each mode has the following parameters:

Note: ELISa models pulsations as small perturbations of an equilibrium stellar surface and computes the
resulting surface variation as a linear superposition of all defined modes (see the ELISa Handbook §5). Amplitude
values are surface velocity perturbations and are expressed in m/s in the UI. Frequency follows the handbook
definition f_mode = ω_p / (2π); the UI expects frequency in cycles/day (d⁻¹).

**Spherical Harmonic Degree (l)**

- Degree of the spherical harmonic describing the surface pattern of the pulsation
- l = 0 (radial), 1 (dipole), 2 (quadrupole), etc.
- Higher l → more surface nodes and smaller-scale structure
- Typical range: 0 to 10 (higher values are usually not observable due to cancellation effects)

**Azimuthal Order (m)**

- Number of nodal lines in longitude; |m| ≤ l
- m = 0 → axisymmetric mode; m ≠ 0 → non-axisymmetric mode
- Sign of m indicates propagation direction (prograde vs retrograde) in rotating stars
- Typical range: -l to +l

**Amplitude [m/s]**

- Amplitude of the surface velocity perturbation (typically radial component)
- Determines strength of the pulsation signal
- Typical range: ~100 to 1,000 m/s for β Cephei-type stars (can vary by class)
- Larger amplitudes increase both radial velocity and photometric variability

**Frequency [cycles/day]**

- Pulsation frequency in cycles per day (d⁻¹)
- Determines the timescale of variability
- Typical range depends on pulsator type (e.g. ~1–10 d⁻¹ for β Cephei, up to ~100 d⁻¹ for δ Scuti)

**Start Phase [dimensionless]**

- Initial phase of the pulsation oscillation at time T₀
- Range: 0 to 1 (fraction of the pulsation cycle)
- Leave empty to use default (0.0)

**Mode Axis Theta / Phi [degrees]**

- Direction the pulsation mode axis points (polar and azimuthal angles)
- Range: 0-180° (theta), 0-360° (phi)
- Affects how pulsations appear in light curve

**Temperature Perturbation Phase Shift [radians]**

- Phase lag between velocity and temperature pulsations
- Leave empty to use default
- Affects light curve amplitude

**Horizontal-to-Radial Amplitude Ratio**

- Ratio of horizontal to radial velocity amplitudes
- Leave empty to auto-calculate
- Affects detailed pulsation shape

**Temperature Amplitude Factor**

- Strength of temperature variations induced by pulsation
- Typical range: 0.001 to 0.1
- Affects light curve depth
- Note: this parameter represents the quantity \~T / T_eff (handbook notation T̃ / T_eff).

**Tidally Locked**

- Whether the pulsation is tidally locked to the orbit
- When enabled, pulsation rotates with the orbit
- Affects phase relationships

---

### **Nuisance Parameters (MCMC Fitting Only)**

**Log Noise (ln_f)** [natural log]

- Represents an extra fractional noise term in the likelihood
- Fitted parameter used by MCMC to account for underestimated observational errors
- Typical range: -10 to 0; use -10 to +2 if you suspect the noise is underestimated
- More negative = model trusts the data errors as given; near 0 = large extra scatter allowed

---

### **Summary: Which Parameters Do I Need?**

**For Modeling** (Light Curve or RV Modeling):

- All mandatory star parameters (Mass, T_eff, Surface Potential, Synchronicity)
- Key orbital parameters (Inclination, Period, Distance or normalize)
- Optional: spots, pulsations, surface features

**For Fitting** (Light Curve or RV Fitting):

- Same star and orbital parameters as Modeling
- Plus: bounds and fit/fix/constrain flags for each parameter
- Upload your data file and specify its format (passband, units, etc.)

**For Visualization**:

- All star and orbital parameters
- Optional: spots, pulsations for detailed rendering

---

## Tips & Best Practices

### General Tips

1. **Check the plots** - Always look at generated light curves against your data to verify the fit quality
2. **Save your work** - Download JSON results so you can load them later
3. **Understand your data** - Know what units and filters your observations are in

### Fitting Tips

1. **Start simple, then complex**
    - First, fit with most parameters fixed
    - Gradually unfix parameters that matter most
    - This prevents the fitter from getting lost

2. **Bounds are critical**
    - Too wide: fitting takes forever, might find wrong solution
    - Too narrow: fitting might miss the true solution
    - Use physics to constrain: e.g., temperatures of stars on main sequence are 3,500-10,000 K

3. **Least Squares first**
    - Run a quick Least Squares fit first (~minutes or hours, depends on complexity of system)
    - Inspect the results visually
    - If good, use as starting point for MCMC (runs longer but gives you confidence intervals)

4. **MCMC is worth the wait**
    - Takes longer but gives you confidence intervals
    - Shows parameter correlations (which parameters are degenerate)
    - More suitable for publishing results

5. **File format matters**
    - Data files should be plain text with columns:
        - For light curves: `phase` (or `time`), `flux` (or `magnitude`), and optionally `error`
        - For RV data: `phase` (or `time`), `velocity`, and optionally `error`
    - One row per measurement

### Visualization Tips

1. **Change the viewing angle** to see eclipses from different perspectives
2. **Use high discretization** (e.g., 5-10+) for fast preview
3. **Use low discretization** (e.g., 3-5) for publication-quality 3D renders
4. **Try different phases** (e.g., -0.5, 0.0, 0.25, 0.5) to see the full orbit

---

## Troubleshooting

### **"The light curve doesn't look right"**

- Verify your parameter values make physical sense (e.g., both stars shouldn't be massive red giants)
- Check your units: inclinination should be ~90° for eclipsing binaries
- Try the visualization to see if the geometry looks correct

### **"Fitting doesn't converge / takes very long"**

- Your bounds might be too wide - try tightening them
- Some parameters might be poorly constrained by your data - fix those
- Try The Least Squares first instead of MCMC
- Verify your data file format is correct

### **"I get an error when uploading data"**

- Check that your file is plain text (not Excel or other format)
- Verify your data has at least 2 columns (time/phase and measurement)
- Remove any useless lines or comments
- Make sure there are no blank lines in the middle of your data

### **"The fitting result doesn't match my visual inspection"**

- Sometimes the best mathematical fit isn't realistic physically
- Try fixing some "unphysical" parameters (e.g., if the fitter suggests a 100 solar mass star)
- Review the residuals - systematic offsets suggest a real problem
- Try a different initial guess or bounds

### **"MCMC is running very slowly"**

This is normal! MCMC can take hours depending on:

- Number of data points
- Number of free parameters
- Quality of the initial guess
- Computer speed
- Pralelism settings in ELISa (see configuration guide in main README file)

**To speed it up:**

- Fix more parameters
- Use fewer data points (thin your data in time/phase)
- Use a higher-performance computer

### **"I can't find my downloaded file"**

- Check your browser's downloads folder
- Results are saved as `.json` files (JSON format)
- You can open them with any text editor to inspect

---

## About Your Data

### Light Curve Data Format

Your data file should be a plain text file with columns separated by whitespace (spaces or tabs). Lines
beginning with `#` are treated as comments and skipped. The reader accepts either **two** or **three** columns:

- x (independent variable): photometric phase (dimensionless) or time (Julian Day)
- y (dependent variable): flux (dimensionless) or magnitude (mag)
- y_err (optional): measurement uncertainty (same units as y)

Notes:

- The default parser expects whitespace-delimited files (this matches the example files in
  `jupyter_tutorials/demo_data/lc_data/`). Comma-separated (CSV) files are not handled by the default UI
  loader; convert them to whitespace-delimited format or pre-process them before upload if needed.
- If your data are in magnitudes, provide the **Reference Magnitude** in the UI so ELISa can convert magnitudes
  to flux internally. Omitting the reference magnitude when uploading magnitudes will raise an error.
- Empty lines are ignored; missing third-column (errors) is allowed.

Example (phase, flux, error):

```
# phase  flux    error
-0.5  1.00000  0.0001
-0.4  0.98532  0.0001
-0.3  0.95234  0.0001
...
```

Or with time (Julian Day) instead of phase:

```
# time  flux    error
2459000.0  1.00000  0.0001
2459001.0  0.98532  0.0001
...
```

### Radial Velocity Data Format

```
# phase, velocity (km/s), error
-0.5  0.00   0.5
-0.4  2.34   0.5
-0.3  4.21   0.5
...
```

---

## Understanding Results

### Least Squares Results

- **Parameter values**: Best-fit values for each parameter
- **Residuals plot**: Shows how well the model matches your data

### MCMC Results

**Corner Plot**

- Visualizes posterior distributions and parameter correlations from MCMC samples
- Diagonal panels: marginal distributions of individual parameters
- Off-diagonal panels: joint distributions showing correlations and degeneracies
- Narrow, well-defined peaks indicate strong constraints
- Broad or multimodal distributions indicate higher uncertainty or parameter degeneracy

**Trace Plot**

- Shows the sampled parameter values as a function of MCMC step
- Used to diagnose convergence and sampling quality
- A well-mixed chain appears as stationary "noise" after burn-in (no visible trends)
- Long-term trends, drifts, or sudden jumps suggest non-convergence or poor mixing
- Multiple chains should overlap and explore the same region of parameter space

**Summary Table**

- Mean and standard deviation for each parameter
- Gives you the best estimate and its uncertainty
- Use these numbers in papers or reports

---

## Getting More Help

### Resources

- **ELISa Handbook** - Comprehensive technical documentation (PDF)
- **Example Notebooks** - There are Jupyter notebooks demonstrating features
- **Error messages** - Check the "Terminal Output" panel at the bottom of the screen for diagnostic information

### When Something Goes Wrong

1. Check the **Terminal Output** section at the bottom of the page
2. Read any error messages carefully - they often suggest the fix
3. Try the steps in **Troubleshooting** section above
4. If still stuck, try creating a simple test case (use default parameters)
5. Create a GitHub issue with details of your problem, including:
    - What you were trying to do
    - What you expected to happen
    - What actually happened (error messages, screenshots)
    - Your system information (OS, Python version)
    - Any relevant data files (if possible)

---

## Summary

**ELISa Desktop App** lets you:

1. ✅ **Model** binary star systems and see what they'd look like
2. ✅ **Fit** your observational data to extract star properties
3. ✅ **Visualize** systems in 3D
4. ✅ **Export** results for publications or further analysis

**To get started:**

1. Launch the app: `python -m elisa.ui`
2. Choose a tab (start with **Light Curve Modeling** for exploration)
3. Adjust parameters and click the main action button
4. Explore the results

---

**Enjoy exploring the universe of binary stars!** 🌟🌟
