"""Spectral-density functions used by the numerical solver."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


SpectralFunction = Callable[[np.ndarray, float, Mapping[str, Any]], np.ndarray]

# Only the selected spectrum's parameters are used. TOML overrides these values.
DEFAULTS = {
    "ohmic": {"xi": 0.2, "ohmic_exponent": 1.0, "wc": 7.5},
    "ohmic_lorentz": {"xi": 0.2, "ohmic_exponent": 1.0, "wc": 7.5,
                      "lorentz_center": 2.7, "lorentz_width": 0.35, "lorentz_weight": 0.2},
    "drude": {"drude_reorganization": 0.1875, "drude_gamma": 7.5},
    "custom": {"amplitude": 0.0005, "exponent": 3.0, "cutoff": 7.5},
}


def parameters_for(name, supplied):
    get_spectral_density(name)
    parameters = {"time_amplitude": 0.0, "time_frequency": 0.05,
                  "time_phase": -np.pi / 2, **DEFAULTS[name]}
    parameters.update(supplied if name == "custom" else
                      {key: value for key, value in supplied.items() if key in parameters})
    if not all(np.isfinite(parameters[key]) for key in ("time_amplitude", "time_frequency", "time_phase")):
        raise ValueError("Time modulation parameters must be finite.")
    if abs(parameters["time_amplitude"]) >= 1:
        raise ValueError("Use abs(time_amplitude) < 1.")
    for key in ("wc", "lorentz_width", "drude_gamma", "cutoff"):
        if key in parameters and not 0 < parameters[key] < np.inf:
            raise ValueError(f"{key} must be finite and positive.")
    for key in ("xi", "lorentz_weight", "drude_reorganization"):
        if key in parameters and not 0 <= parameters[key] < np.inf:
            raise ValueError(f"{key} must be finite and nonnegative.")
    return parameters


def _modulation(t: float, parameters: Mapping[str, Any]) -> float:
    amplitude = float(parameters.get("time_amplitude", 0.0))
    frequency = float(parameters.get("time_frequency", 0.0))
    phase = float(parameters.get("time_phase", 0.0))
    value = 1.0 + amplitude * np.cos(frequency * t + phase)
    if value <= 0.0:
        raise ValueError("The spectral-density modulation became non-positive.")
    return float(value)


def ohmic(omega, t: float, parameters: Mapping[str, Any]):
    """Return an Ohmic or power-law density with exponential cutoff."""
    w = np.asarray(omega, dtype=float)
    exponent = float(parameters["ohmic_exponent"])
    cutoff = float(parameters["wc"])
    static = (
        0.5
        * float(parameters["xi"])
        * np.power(w, exponent)
        * cutoff ** (1.0 - exponent)
        * np.exp(-w / cutoff)
    )
    return static * _modulation(t, parameters)


def ohmic_lorentz(omega, t: float, parameters: Mapping[str, Any]):
    """Return an Ohmic density with an additional Lorentzian peak."""
    w = np.asarray(omega, dtype=float)
    static_parameters = {**parameters, "time_amplitude": 0.0}
    base = ohmic(w, 0.0, static_parameters)
    width = float(parameters["lorentz_width"])
    peak = (
        float(parameters["lorentz_weight"])
        * width
        / np.pi
        / ((w - float(parameters["lorentz_center"])) ** 2 + width**2)
    )
    return (base + peak) * _modulation(t, parameters)


def drude(omega, t: float, parameters: Mapping[str, Any]):
    """Return the Drude-Lorentz density used for HEOM comparison."""
    w = np.asarray(omega, dtype=float)
    reorganization = float(parameters["drude_reorganization"])
    gamma = float(parameters["drude_gamma"])
    static = 2.0 * reorganization * gamma * w / (w**2 + gamma**2)
    return static * _modulation(t, parameters)


def custom(omega, t: float, parameters: Mapping[str, Any]):
    """Return the user-editable custom density selected by `custom` in TOML."""
    w = np.asarray(omega, dtype=float)
    amplitude = float(parameters.get("amplitude", 0.0005))
    exponent = float(parameters.get("exponent", 3.0))
    cutoff = float(parameters.get("cutoff", 7.5))
    return (
        amplitude
        * np.power(w, exponent)
        * np.exp(-w / cutoff)
        * _modulation(t, parameters)
    )


# Set True only if custom(omega,t) = custom(omega,0)*modulation(t)/modulation(0).
# False also permits an arbitrary, slowly varying frequency shape.
CUSTOM_SEPARABLE = False


SPECTRAL_DENSITIES: dict[str, SpectralFunction] = {
    "ohmic": ohmic,
    "ohmic_lorentz": ohmic_lorentz,
    "drude": drude,
    "custom": custom,
}


def get_spectral_density(name: str) -> SpectralFunction:
    """Return one spectral-density function by its TOML name."""
    try:
        return SPECTRAL_DENSITIES[name]
    except KeyError as exc:
        choices = ", ".join(SPECTRAL_DENSITIES)
        raise ValueError(
            f"Unknown spectral density {name!r}; choose one of: {choices}."
        ) from exc


def evaluate(name: str, omega, t: float, parameters: Mapping[str, Any]):
    """Evaluate and validate a selected spectral density."""
    w = np.asarray(omega, dtype=float)
    values = np.asarray(get_spectral_density(name)(w, float(t), parameters))
    if np.any(np.abs(values.imag) > 1.0e-12):
        raise ValueError("A spectral density must be real-valued.")
    values = np.asarray(values.real, dtype=float)
    if values.shape != w.shape:
        values = np.broadcast_to(values, w.shape).copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("The spectral density returned non-finite values.")
    if np.any(values < -1.0e-14):
        raise ValueError("The spectral density returned negative values.")
    return np.maximum(values, 0.0)


def time_factor(name: str, t: float, parameters: Mapping[str, Any]) -> float | None:
    """Return J(omega,t)/J(omega,0) when the density is separable in time."""
    if name == "custom" and not CUSTOM_SEPARABLE:
        return None
    return _modulation(t, parameters) / _modulation(0.0, parameters)


def describe(name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready description for result metadata."""
    return {"name": name, "parameters": dict(parameters)}
