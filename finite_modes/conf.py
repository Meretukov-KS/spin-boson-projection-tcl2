"""Parameters for one spin and explicit bosonic modes."""

from dataclasses import dataclass
from pathlib import Path
import math
import tomllib

DEFAULT_CONFIG_PATH = Path(__file__).with_name("params.toml")


@dataclass(frozen=True)
class SpinConfig:
    DELTA: float
    H_FIELD: float
    PAULI_I: tuple[float, float, float]


@dataclass(frozen=True)
class ModesConfig:
    g: float
    w: float
    levels: int


@dataclass(frozen=True)
class Params:
    SPINS: tuple[SpinConfig, ...]
    MODES: tuple[ModesConfig, ...]
    BETA: float
    LAMBDA: float
    T_MAX: float
    T_POINT: int


def load(path=DEFAULT_CONFIG_PATH):
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    points = data["time"]["points"]
    if type(points) is not int or points < 2:
        raise ValueError("time.points must be an integer >= 2.")
    modes = []
    for m in data["modes"]:
        levels = m.get("levels", 12)
        if type(levels) is not int or levels < 2:
            raise ValueError("Each mode needs an integer levels >= 2.")
        modes.append(ModesConfig(float(m["g"]), float(m["w"]), levels))
    config = Params(
        tuple(SpinConfig(float(s["delta"]), float(s["field"]),
                         tuple(float(v) for v in s["initial_bloch"])) for s in data["spins"]),
        tuple(modes), float(data["bath"]["beta"]), float(data["bath"]["lambda"]),
        float(data["time"]["maximum"]), points,
    )
    if len(config.SPINS) != 1 or not modes:
        raise ValueError("Equation (16) requires one spin and at least one mode.")
    spin = config.SPINS[0]
    if len(spin.PAULI_I) != 3 or sum(v*v for v in spin.PAULI_I) > 1+1e-12:
        raise ValueError("initial_bloch must have three components and norm <= 1.")
    values = [spin.DELTA, spin.H_FIELD, *spin.PAULI_I, config.LAMBDA, config.T_MAX,
              *(v for m in modes for v in (m.g, m.w))]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Parameters must be finite (except beta=inf).")
    if not config.BETA > 0 or not 0 <= config.LAMBDA < 1 or config.T_MAX <= 0:
        raise ValueError("Require beta > 0, 0 <= lambda < 1 and maximum > 0.")
    if any(m.w <= 0 for m in modes):
        raise ValueError("Mode frequencies must be positive.")
    return config
