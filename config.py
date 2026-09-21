"""Read physical parameters and grid sizes from one TOML file."""

from dataclasses import dataclass, replace
from pathlib import Path
import math
import tomllib

from spectral_density import parameters_for

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class SpinConfig:
    DELTA: float
    H_FIELD: float
    COUPLING: float
    PAULI_I: tuple[float, float, float]


@dataclass(frozen=True)
class SimulationConfig:
    SOURCE_PATH: str
    SPINS: tuple[SpinConfig, ...]
    BETA: float
    LAMBDA: float
    SPECTRAL_DENSITY: str
    SPECTRAL_PARAMETERS: tuple
    T_MAX: float
    T_POINTS: int
    OMEGA_MIN: float
    OMEGA_MAX: float
    MODE_GRID_POINTS: int
    PHYSICAL_MODE_COUNT: float
    QUAD_POINTS: int
    PROBE_FREQUENCIES: tuple
    Delta_N: float = 0.0
    sigma: float = 0.2
    omega_car: float = 2.7

    @property
    def TEMP(self):
        return 1.0 / self.BETA

    @property
    def CONFIG_NAME(self):
        return Path(self.SOURCE_PATH).name

    def spectral_parameters(self, name=None):
        name = name or self.SPECTRAL_DENSITY
        sections = dict(self.SPECTRAL_PARAMETERS)
        return parameters_for(name, dict(sections.get(name, ())))


def load_config(path=DEFAULT_CONFIG_PATH):
    source = Path(path).resolve()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    # Old result configs remain intact, but internal tuning is no longer read.
    unknown = data.keys() - {"spins", "bath", "spectral_parameters", "time", "frequency", "initial_occupation"}
    if unknown:
        raise ValueError(f"Remove obsolete configuration sections: {', '.join(sorted(unknown))}.")
    bath, time, frequency = data["bath"], data["time"], data["frequency"]
    initial = data.get("initial_occupation", {})
    spectra = data.get("spectral_parameters", {})
    if not all(isinstance(section, dict) for section in spectra.values()):
        raise ValueError("Put spectral parameters in named sections, e.g. [spectral_parameters.ohmic].")
    config = SimulationConfig(
        SOURCE_PATH=str(source),
        SPINS=tuple(SpinConfig(float(s["delta"]), float(s["field"]),
                              float(s.get("coupling", 1.0)), tuple(s["initial_bloch"]))
                    for s in data["spins"]),
        BETA=float(bath["beta"]), LAMBDA=float(bath["lambda"]),
        SPECTRAL_DENSITY=bath["spectral_density"],
        SPECTRAL_PARAMETERS=tuple((name, tuple(section.items())) for name, section in spectra.items()),
        T_MAX=float(time["maximum"]), T_POINTS=int(time["points"]),
        OMEGA_MIN=float(frequency["minimum"]), OMEGA_MAX=float(frequency["maximum"]),
        MODE_GRID_POINTS=int(frequency["physical_grid_points"]),
        PHYSICAL_MODE_COUNT=float(frequency["physical_mode_count"]),
        QUAD_POINTS=int(frequency["quadrature_points"]),
        PROBE_FREQUENCIES=tuple(frequency.get("probe_frequencies", [])),
        Delta_N=float(initial.get("excess_amplitude", 0.0)),
        sigma=float(initial.get("excess_width", 0.2)),
        omega_car=float(initial.get("excess_center", 2.7)),
    )
    validate_config(config)
    return config


def validate_config(c):
    if math.isnan(c.BETA) or c.BETA <= 0:
        raise ValueError("bath.beta must be positive (inf is allowed for zero temperature).")
    if not 0 < c.LAMBDA < 1:
        raise ValueError("bath.lambda must lie in (0, 1).")
    if not math.isfinite(c.T_MAX) or c.T_MAX <= 0 or c.T_POINTS < 2:
        raise ValueError("Use positive time.maximum and at least two time points.")
    if not 0 < c.OMEGA_MIN < c.OMEGA_MAX < math.inf:
        raise ValueError("Use finite frequencies with 0 < minimum < maximum.")
    if min(c.MODE_GRID_POINTS, c.QUAD_POINTS) < 2 or not 0 < c.PHYSICAL_MODE_COUNT < math.inf:
        raise ValueError("Use at least two grid points and a positive physical_mode_count.")
    if not 0 < c.sigma < math.inf or not 0 <= c.Delta_N < math.inf or not math.isfinite(c.omega_car):
        raise ValueError("The initial excess must be nonnegative with a positive width and finite center.")
    if any(not math.isfinite(w) or not c.OMEGA_MIN <= w <= c.OMEGA_MAX for w in c.PROBE_FREQUENCIES):
        raise ValueError("probe_frequencies must lie inside the frequency interval.")
    if not c.SPINS:
        raise ValueError("At least one spin is required.")
    for spin in c.SPINS:
        values = (spin.DELTA, spin.H_FIELD, spin.COUPLING, *spin.PAULI_I)
        if len(spin.PAULI_I) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("Each spin requires finite parameters and a three-component initial_bloch.")
        if sum(v * v for v in spin.PAULI_I) > 1 + 1e-7:
            raise ValueError("The initial Bloch vector must have norm at most one.")
    c.spectral_parameters()


_active = None


def get_config():
    global _active
    if _active is None:
        _active = load_config()
    return _active


def set_active_config(path=DEFAULT_CONFIG_PATH, comparison=False):
    """Select the model before importing either solver (one config per process)."""
    global _active
    config = load_config(path)
    if comparison:
        if len(config.SPINS) != 1 or config.TEMP == 0:
            raise ValueError("HEOM comparison requires one spin and a positive temperature.")
        if config.Delta_N != 0 or config.spectral_parameters("drude")["time_amplitude"] != 0:
            raise ValueError("HEOM comparison requires a thermal initial bath and a stationary density.")
        config = replace(config, SPECTRAL_DENSITY="drude")
        validate_config(config)
    _active = config
    return config
