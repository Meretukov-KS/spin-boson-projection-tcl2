# Spin-boson solver

Numerical integration of coupled TCL2 equations for spin expectation values and
bosonic mode populations. Optional comparison with the QuTiP HEOM solver.

## Requirements

- Python 3.11 or newer.
- NumPy and Matplotlib.
- QuTiP for HEOM comparisons. The code has been tested with QuTiP 5.2.3.

Install the required packages in the Python environment used to run the program:

```sh
python -m pip install numpy matplotlib
python -m pip install qutip
```

The second command is optional for TCL2-only runs.

## Running

Run these commands from the directory containing `run.py`:

```sh
python run.py numeric
python run.py compare
python run.py compare-frozen
```

| Mode | Calculation |
| --- | --- |
| `numeric` | TCL2 with evolving mode populations and the selected spectral density. Supports one or more spins. |
| `compare` | TCL2 with evolving mode populations and a HEOM reference calculation. Uses the Drude spectrum. |
| `compare-frozen` | TCL2 with mode populations fixed at their initial values and a HEOM reference calculation. Uses the Drude spectrum. |

Omitting the mode selects `numeric`. By default, the program reads `config.toml`
next to the source files. To use another configuration:

```sh
python run.py numeric --config experiment.toml
```

Use a separate process for each configuration. Run calculations through `run.py`,
not by executing `numerical.py` or `heom.py` directly.

## Configuration

Edit `config.toml` before running. Use consistent units for energy, frequency,
time, and inverse temperature; the code uses `hbar = k_B = 1`.

### Spins: `[[spins]]`

| Parameter | Meaning | Shipped value |
| --- | --- | --- |
| `delta` | Coefficient in `H = -delta*sigma_x/2 + field*sigma_z/2`. | `2.0` |
| `field` | Coefficient of `sigma_z` in the spin Hamiltonian. | `0.0` |
| `coupling` | Individual spin coupling multiplier, in addition to `bath.lambda`. | `1.0` |
| `initial_bloch` | Initial `[<sigma_x>, <sigma_y>, <sigma_z>]`. Its norm must not exceed one. | `[0.0, 0.0, 1.0]` |

For multiple spins, add another `[[spins]]` block with its own values. The TCL2
solver uses a shared bath and factorized inter-spin averages; it does not evolve
a full multi-spin density matrix. HEOM comparison supports one spin only.

### Bath: `[bath]`

| Parameter | Meaning | Shipped value |
| --- | --- | --- |
| `beta` | Inverse temperature. Must be positive; TOML `inf` selects zero temperature in `numeric`. | `5.0` |
| `lambda` | Interaction parameter. Must satisfy `0 < lambda < 1`. | `0.1` |
| `spectral_density` | `ohmic`, `ohmic_lorentz`, `drude`, or `custom`. | `"ohmic"` |

The solver applies `lambda^2` separately. Do not include this factor again in
the spectral-density function.

### Spectral-density parameters

Each spectrum has an independent section. Changing parameters in one section
does not change any other spectrum. Omitted parameters use the defaults below.

| Section | Parameter | Meaning | Default |
| --- | --- | --- | --- |
| `[spectral_parameters.ohmic]` | `xi` | Spectral strength. | `0.2` |
| | `ohmic_exponent` | Power-law exponent `s`; `s=1` gives the Ohmic spectrum. | `1.0` |
| | `wc` | Positive exponential cutoff frequency. | `7.5` |
| `[spectral_parameters.ohmic_lorentz]` | `xi`, `ohmic_exponent`, `wc` | Parameters of this spectrum's own Ohmic component. | `0.2`, `1.0`, `7.5` |
| | `lorentz_center` | Frequency of the additional peak. | `2.7` |
| | `lorentz_width` | Positive half-width at half-maximum. | `0.35` |
| | `lorentz_weight` | Nonnegative peak weight. | `0.2` |
| `[spectral_parameters.drude]` | `drude_reorganization` | Drude strength before applying interaction multipliers. | `0.1875` |
| | `drude_gamma` | Positive Drude cutoff frequency. | `7.5` |
| `[spectral_parameters.custom]` | `amplitude` | Amplitude of the supplied example function. | `0.0005` |
| | `exponent` | Power-law exponent of the example function. | `3.0` |
| | `cutoff` | Positive cutoff of the example function. | `7.5` |

The stationary functions are:

```text
ohmic:
    J(w) = xi/2 * w^s * wc^(1-s) * exp(-w/wc)

ohmic_lorentz:
    J(w) = J_ohmic(w) + lorentz_weight * lorentz_width
           / (pi * ((w-lorentz_center)^2 + lorentz_width^2))

drude:
    J(w) = 2 * drude_reorganization * drude_gamma * w
           / (w^2 + drude_gamma^2)

custom (supplied example):
    J(w) = amplitude * w^exponent * exp(-w/cutoff)
```

#### Time modulation

Add these optional parameters to the section of the spectrum being used:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `time_amplitude` | Relative modulation amplitude; absolute value must be below one. | `0.0` |
| `time_frequency` | Angular modulation frequency. | `0.05` |
| `time_phase` | Phase in radians. | `-1.5707963267948966` |

The built-in functions and the supplied custom example multiply their stationary
spectra by:

```text
1 + time_amplitude * cos(time_frequency*t + time_phase)
```

Zero amplitude gives a stationary spectrum. Time-dependent calculations use an
adiabatic approximation: parameters must vary slowly over the bath correlation
time. The program's warning is a heuristic check, not an automatic validity test.

#### Custom function

Set `bath.spectral_density = "custom"` and edit
`custom(omega, t, parameters)` in `spectral_density.py`.

- `omega` is a NumPy array of frequencies, or a scalar.
- `t` is the current time.
- `parameters` contains the entries from `[spectral_parameters.custom]` and defaults.
- Return finite, real, nonnegative values with the same shape as `omega`, or a broadcastable value.
- Add any parameters required by your function to `[spectral_parameters.custom]`.
- If you replace the function, include time modulation explicitly if needed.

By default, `CUSTOM_SEPARABLE = False` uses direct quadrature. Set it to `True`
in `spectral_density.py` only when the function satisfies
`J(w,t) = J(w,0)*modulation(t)/modulation(0)`. This enables kernel precomputation.
A stationary function satisfies this condition when `time_amplitude = 0`.

### Time grid: `[time]`

| Parameter | Meaning | Shipped value |
| --- | --- | --- |
| `maximum` | Positive final time; the initial time is zero. | `20.0` |
| `points` | Number of uniformly spaced time points, including both endpoints; at least two. | `801` |

The TCL2 step is `maximum / (points - 1)`. HEOM uses the same output times,
with internal steps chosen by its integrator.

### Frequency grids: `[frequency]`

| Parameter | Meaning | Shipped value |
| --- | --- | --- |
| `minimum` | Positive lower frequency bound. | `0.05` |
| `maximum` | Finite upper bound, greater than `minimum`. Must be a number, not `"auto"`. | `49.7876405099536` |
| `physical_grid_points` | Base number of nodes used to evolve `N(w,t)`; at least two. | `200` |
| `physical_mode_count` | Positive physical mode-count normalization, independent of numerical node counts. | `200` |
| `quadrature_points` | Requested number of frequency-integration nodes; at least two. | `5200` |
| `probe_frequencies` | Optional frequencies for population traces. Must lie inside the frequency interval. | `[2.0, 2.7, 3.2]` |

The mode density and individual mode coupling are defined by:

```text
p = physical_mode_count / (maximum - minimum)
|g(w,t)|^2 = J(w,t) / p
```

Changing a numerical node count does not change `p`. Changing the frequency
interval at fixed `physical_mode_count` does change `p`.

The population grid has a uniform base plus deterministic refinement bands around
the spin transition frequencies. Band half-widths are `max(Omega/2, 2*pi/T_MAX)`;
their maximum spacing is `pi/(8*T_MAX)`. Exact transition and probe frequencies
are included. Thus the actual node count may exceed `physical_grid_points`.
Population nodes do not depend on the spectral density, its parameters,
temperature, or initial populations. Keep the same spin, frequency, probe, and
time settings to compare runs by direct subtraction on identical nodes.

Only the integration grid concentrates nodes according to `J(w,0)*(2*N(w,0)+1)`.
It also includes uniformly spaced guard nodes and may exceed `quadrature_points`
to resolve oscillatory phases. Both grids are built before time integration and
remain fixed during the run. Population changes are interpolated onto quadrature
nodes inside the integral; no interpolation between compared runs is necessary.

### Initial populations: `[initial_occupation]`

| Parameter | Meaning | Default |
| --- | --- | --- |
| `excess_amplitude` | Nonnegative amplitude of an additional Gaussian peak. | `0.0` |
| `excess_width` | Positive Gaussian standard deviation. | `0.2` |
| `excess_center` | Center frequency of the Gaussian peak. | `2.7` |

```text
N(w,0) = 1 / (exp(beta*w) - 1)
         + excess_amplitude * exp(-(w-excess_center)^2 / (2*excess_width^2))
```

Set `excess_amplitude = 0` for a thermal initial bath. The section can remain
in the file or be omitted. A positive amplitude adds a nonthermal population peak.

## HEOM comparison requirements

Both comparison modes select `[spectral_parameters.drude]` automatically,
regardless of `bath.spectral_density`. They require:

- Exactly one `[[spins]]` block.
- Positive temperature, meaning finite positive `beta`.
- `excess_amplitude = 0`.
- `time_amplitude = 0` in the Drude section, or that parameter omitted.

The spin Hamiltonian, initial spin state, temperature, `lambda`, and individual
`coupling` are shared by the two calculations. Modulation settings in inactive
spectral sections do not affect the comparison. TCL2 uses the configured finite
frequency interval; HEOM uses the full Drude spectrum.

### HEOM convergence check

From the directory containing `run.py`, run:

```sh
python tests/check_heom_convergence.py
```

The script uses the Drude comparison settings in `config.toml` and requires
QuTiP. It compares Padé expansion sizes `10, 12, 14, 16` against `18` at
hierarchy depth `5`, and depths `3, 4, 5` against `6` with `16` Padé terms.
For each spin component, the plotted value is the maximum absolute difference
over the output time grid relative to the highest setting tested. The reference
setting is not assumed to be exact.

The plots are saved as `heom_check/pade_convergence.png` and
`heom_check/depth_convergence.png`. The folder is created next to `run.py`
when the check starts. The Padé plot is saved after the five expansion runs;
the depth plot is saved after the remaining three runs. Progress is printed
to the terminal. Rerunning the check replaces these two plots. This check can
take substantially longer than a single comparison run.

## Numerical settings

Internal settings are constants, not TOML parameters:

- `numerical.py`: absolute tolerance `1e-9`, relative tolerance `1e-7`,
  maximum `40` fixed-point iterations, physicality tolerance `1e-7`.
- `heom.py`: `N_DEPTH = 5`, `N_EXPANSION = 16`, `USE_PADE = True`.
- The kernel lookup-grid size is selected automatically.

TCL2 uses fixed-step TR-BDF2. Its tolerances control implicit iterations, not
the total integration error. Nonconvergence stops the run; increase `time.points`
and retry. Check time and frequency resolution when changing the calculation range
or introducing narrow spectral features. HEOM defaults do not guarantee convergence
for arbitrary parameters.

Negative populations and Bloch-vector norms above one produce warnings and are
not clipped. Check these warnings before using the results.

## Output

Each run creates a new timestamped directory under `numeric/`, `compare/`, or
`compare-frozen/` next to the source files. Existing results are not overwritten.

| File | Contents |
| --- | --- |
| `spin_boson_results.png` | TCL2 plots for `numeric`. |
| `compare_dynamic_N.png` or `compare_frozen_N.png` | Comparison plots. |
| `solution.npz` | Time and frequency arrays, spin trajectories, per-spin purities, and populations. |
| `compare_solution.npz` | TCL2 and HEOM trajectories and purities, plus TCL2 populations. |
| `bath_data.npz` | Initial `J`, `N0`, mode density, couplings, both frequency grids, quadrature weights, and final populations. |
| `bath_initial.csv` | Columns `omega,J,N0,p,g_squared` on the population grid. |
| `config.toml` | Copy of the input configuration. |
| `metadata.json` | Effective parameters, numerical settings, diagnostics, comparison metrics, and timings. |
| `summary.txt` | Run mode and separate TCL2/HEOM timing summaries. |

For comparisons, the copied input may select another spectrum; `metadata.json`
records the Drude spectrum actually used. Solver timings exclude kernel or HEOM
setup; total-stage timings include setup. Timings are not printed on plots.

Load an output array with NumPy:

```python
import numpy as np

with np.load("numeric/<timestamp>/solution.npz") as data:
    t = data["t"]
    spin = data["sigma"]          # (time, spin, component)
    populations = data["occupations"]  # (time, frequency)
```

## Tests

```sh
python -m unittest discover -s tests -v
```

The tests cover configuration, spectral functions, grids, equation signs,
correlations, output files, multi-spin runs, and both comparison modes.
HEOM tests are skipped if QuTiP is not installed.
