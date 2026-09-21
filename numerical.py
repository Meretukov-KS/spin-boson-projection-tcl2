"""Numerical TCL2 solution of the spin-boson equations.

For one spin, the solver evaluates

  d<sigma_alpha>/dt = 4 lambda^2 int dt1 int dω J(ω,t) sum_lk c_k(t)c_l(t1)
      [sum_bg eps(k,a,b)eps(l,b,g)<sigma_g>(2N+1)cos(ω tau)
       + eps(k,a,l)sin(ω tau)],

  dN(ω,t)/dt = 2 lambda^2 J(ω,t)/p(ω) int dt1
      [cos(ω tau) sum_l c_l(t)c_l(t1)
       - sin(ω tau)(2N+1)sum_lkg c_l(t1)c_k(t)eps(l,k,g)<sigma_g>].

The multi-spin version uses a shared bath and factorized inter-spin averages.
Spin averages and N(ω,t) are advanced together with one TR-BDF2 step. The
physical mode density p(ω) is independent of the quadrature-node density.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import os
from time import perf_counter
import warnings

import numpy as np

from config import get_config
from spectral_density import describe, evaluate, get_spectral_density, time_factor


pm = get_config()
ACTIVE_SPECTRAL_DENSITY = pm.SPECTRAL_DENSITY

# Internal numerical defaults; these are deliberately not read from TOML.
MAX_ITERATIONS = 40
ABSOLUTE_TOLERANCE = 1e-9
RELATIVE_TOLERANCE = 1e-7
PHYSICALITY_TOLERANCE = 1e-7
ADAPTIVE_STRENGTH = 0.8
MAX_PHASE_INCREMENT = np.pi / 4
# Resolve the highest frequency on the kernel interpolation grid automatically.
KERNEL_TAU_POINTS = max(24001, int(np.ceil(8 * pm.OMEGA_MAX * pm.T_MAX)) + 1)


# ---------------------------------------------------------------------
# Spin operators and parameters
# ---------------------------------------------------------------------

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
pauli = (sx, sy, sz)

eps = np.zeros((3, 3, 3))
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1.0

N_SPINS = len(pm.SPINS)
DELTA_j = np.asarray([item.DELTA for item in pm.SPINS], dtype=float)
H_FIELD_j = np.asarray([item.H_FIELD for item in pm.SPINS], dtype=float)
ETA_j = np.asarray([item.COUPLING for item in pm.SPINS], dtype=float)
ETA2_j = ETA_j**2
BETA0 = np.asarray([item.PAULI_I for item in pm.SPINS], dtype=float)

Omega_j = DELTA_j**2 + H_FIELD_j**2
sqrt_Omega_j = np.sqrt(Omega_j)


def H_S_of(j: int) -> np.ndarray:
    """H_S^(j) = -DELTA_j sigma_x/2 + H_FIELD_j sigma_z/2."""
    return -0.5 * DELTA_j[j] * sx + 0.5 * H_FIELD_j[j] * sz


# ---------------------------------------------------------------------
# Spectral density, p(ω), |g|^2, and initial occupation
# ---------------------------------------------------------------------

def resolve_spectral_density(name: str | None = None) -> str:
    """Return and validate the selected spectral-density name."""
    name = ACTIVE_SPECTRAL_DENSITY if name is None else name
    get_spectral_density(name)
    return name


def spectral_description(name: str | None = None) -> dict:
    """Return the selected density and its parameters for metadata."""
    name = resolve_spectral_density(name)
    return describe(name, pm.spectral_parameters(name))


def adiabaticity_parameter(
    spectral_density: str | None = None,
) -> float:
    """Estimate tau_B*max|d ln J/dt| for the selected modulation."""
    density = resolve_spectral_density(spectral_density)
    parameters = pm.spectral_parameters(density)
    amplitude = float(parameters.get("time_amplitude", 0.0))
    frequency = float(parameters.get("time_frequency", 0.0))
    if amplitude == 0.0:
        return 0.0
    bath_scale = float(parameters.get("drude_gamma", parameters.get("cutoff", parameters.get("wc", 7.5))))
    if density == "ohmic_lorentz":
        bath_scale = min(bath_scale, parameters["lorentz_width"])
    max_log_rate = abs(amplitude * frequency) / (1.0 - abs(amplitude))
    return float(max_log_rate / bath_scale)


def J(
    omega,
    t: float = 0.0,
    density: str | None = None,
):
    """Return the instantaneous spectral density J(ω,t)."""
    name = resolve_spectral_density(density)
    return evaluate(name, omega, t, pm.spectral_parameters(name))


def mode_density(omega):
    """Return uniform physical mode density p(ω), independent of grid nodes."""
    w = np.asarray(omega, dtype=float)
    p0 = pm.PHYSICAL_MODE_COUNT / (pm.OMEGA_MAX - pm.OMEGA_MIN)
    return np.full_like(w, p0, dtype=float)


def coupling_squared(
    omega,
    t: float = 0.0,
    density: str | None = None,
):
    """Return |g_tilde(ω,t)|^2 = J(ω,t)/p(ω) where p(ω)>0."""
    p = mode_density(omega)
    if np.any(p <= 0):
        raise ValueError("The physical mode density must be positive.")
    return J(omega, t=t, density=density) / p


def bose_occupation(omega):
    w = np.asarray(omega, dtype=float)
    if pm.TEMP == 0:
        return np.zeros_like(w)
    x = w / pm.TEMP
    result = np.zeros_like(x)
    mask = x < 700.0
    result[mask] = 1.0 / np.expm1(x[mask])
    return result


def N_I(omega):
    """Return Bose occupation plus an optional Gaussian excess."""
    w = np.asarray(omega, dtype=float)
    bump = pm.Delta_N * np.exp(-0.5 * ((w - pm.omega_car) / pm.sigma) ** 2)
    return bose_occupation(w) + bump


# ---------------------------------------------------------------------
# Frequency grids and quadrature weights
# ---------------------------------------------------------------------

t_arr = np.linspace(0.0, pm.T_MAX, pm.T_POINTS)
dt = t_arr[1] - t_arr[0]

def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("A quadrature grid must be strictly increasing.")
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def _importance_nodes(
    count: int,
    spectral_density: str,
    strength: float,
) -> np.ndarray:
    """Place nodes by a hybrid uniform and correlation-importance density."""
    if count <= 2 or strength == 0.0:
        return np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, count)
    dense_count = max(20000, 10 * count)
    dense = np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, dense_count)
    importance = J(dense, density=spectral_density) * (
        2.0 * N_I(dense) + 1.0
    )
    importance = np.maximum(importance, 0.0)
    increments = 0.5 * (importance[1:] + importance[:-1]) * np.diff(dense)
    normalization = float(np.sum(increments))
    if not np.isfinite(normalization) or normalization <= 0.0:
        return np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, count)
    density = (
        (1.0 - strength) / (pm.OMEGA_MAX - pm.OMEGA_MIN)
        + strength * importance / normalization
    )
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(dense)))
    )
    cumulative /= cumulative[-1]
    return np.interp(np.linspace(0.0, 1.0, count), cumulative, dense)


def build_mode_grid(density):
    base = _importance_nodes(pm.MODE_GRID_POINTS, density, ADAPTIVE_STRENGTH)
    anchors = np.asarray(tuple(sqrt_Omega_j) + pm.PROBE_FREQUENCIES, dtype=float)
    anchors = anchors[(anchors >= pm.OMEGA_MIN) & (anchors <= pm.OMEGA_MAX)]
    return np.unique(np.concatenate((base, anchors)))


omega_arr = build_mode_grid(ACTIVE_SPECTRAL_DENSITY)
N0 = N_I(omega_arr)
def build_quadrature_grid(
    spectral_density: str | None = None,
) -> np.ndarray:
    """Build a hybrid importance grid while retaining phase resolution."""
    density = resolve_spectral_density(spectral_density)
    interval = pm.OMEGA_MAX - pm.OMEGA_MIN
    maximum_gap = MAX_PHASE_INCREMENT / pm.T_MAX
    guard_count = max(2, int(np.ceil(interval / maximum_gap)) + 1)
    target_count = max(pm.QUAD_POINTS, guard_count)
    if guard_count > pm.QUAD_POINTS:
        warnings.warn(
            "quadrature_points was increased to satisfy the phase-resolution limit.",
            RuntimeWarning,
            stacklevel=2,
        )

    guard = np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, guard_count)
    adaptive_count = target_count - guard_count
    if adaptive_count == 0:
        return guard

    adaptive = _importance_nodes(adaptive_count + 2, density, ADAPTIVE_STRENGTH)[1:-1]
    return np.unique(np.concatenate((guard, adaptive)))


_quadrature_density_key = ACTIVE_SPECTRAL_DENSITY
quad_omega = build_quadrature_grid(ACTIVE_SPECTRAL_DENSITY)
quad_weights = trapezoid_weights(quad_omega)


def configure_quadrature(spectral_density: str) -> None:
    """Rebuild density-dependent adaptive nodes before a solver run."""
    global _quadrature_density_key, quad_omega, quad_weights, omega_arr, N0
    density = resolve_spectral_density(spectral_density)
    key = density
    if key == _quadrature_density_key:
        return
    omega_arr = build_mode_grid(density)
    N0 = N_I(omega_arr)
    quad_omega = build_quadrature_grid(density)
    quad_weights = trapezoid_weights(quad_omega)
    _quadrature_density_key = key
    _kernel_cache.clear()
    _phase_cache.clear()


def quadrature_diagnostics() -> dict[str, float | int | bool]:
    """Return compact diagnostics for the actual quadrature grid."""
    spacing = np.diff(quad_omega)
    return {
        "adaptive": True,
        "kernel_tau_points": KERNEL_TAU_POINTS,
        "mode_grid_points": len(omega_arr),
        "requested_points": int(pm.QUAD_POINTS),
        "actual_points": int(len(quad_omega)),
        "minimum_spacing": float(np.min(spacing)),
        "maximum_spacing": float(np.max(spacing)),
        "maximum_phase_increment": float(np.max(spacing) * pm.T_MAX),
        "omega_minimum": float(quad_omega[0]),
        "omega_maximum": float(quad_omega[-1]),
    }


# ---------------------------------------------------------------------
# Coefficients in sigma_z(t)=sum_k c_k(t)sigma_k
# ---------------------------------------------------------------------

def c_coef_spin(t: float, j: int) -> np.ndarray:
    Om, sOm = Omega_j[j], sqrt_Omega_j[j]
    if Om == 0.0:
        return np.array([0.0, 0.0, 1.0])
    D, h = DELTA_j[j], H_FIELD_j[j]
    return np.array(
        [
            D * h * (np.cos(t * sOm) - 1.0) / Om,
            -D * np.sin(t * sOm) / sOm,
            (D**2 * np.cos(t * sOm) + h**2) / Om,
        ]
    )


def c_coef_all(t: float) -> np.ndarray:
    return np.asarray([c_coef_spin(t, j) for j in range(N_SPINS)])


c_coef_arr = np.asarray([c_coef_all(t) for t in t_arr])


# ---------------------------------------------------------------------
# Correlation kernel and short-lived phase caches
# ---------------------------------------------------------------------

_kernel_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_phase_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()


def _phase_matrices(grid_name: str, grid: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cache phase matrices across fixed-point iterations of one solver stage."""
    tau = np.asarray(tau, dtype=float)
    key = (grid_name, len(tau), tau.tobytes())
    cached = _phase_cache.get(key)
    if cached is not None:
        _phase_cache.move_to_end(key)
        return cached
    phase = np.outer(grid, tau)
    matrices = (np.cos(phase), np.sin(phase))
    _phase_cache[key] = matrices
    while len(_phase_cache) > 4:
        _phase_cache.popitem(last=False)
    return matrices


def _build_kernel_lookup(
    spectral_density: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density = resolve_spectral_density(spectral_density)
    key = density
    cached = _kernel_cache.get(key)
    if cached is not None:
        return cached

    tau_grid = np.linspace(0.0, pm.T_MAX, KERNEL_TAU_POINTS)
    weighted_j = quad_weights * J(quad_omega, density=density)
    equilibrium_factor = 2.0 * N_I(quad_omega) + 1.0
    corr = np.empty_like(tau_grid, dtype=complex)
    occupation_kernel = np.empty((len(tau_grid), len(omega_arr)), dtype=float)

    right = np.searchsorted(omega_arr, quad_omega, side="right")
    right = np.clip(right, 1, len(omega_arr) - 1)
    left = right - 1
    fraction = (quad_omega - omega_arr[left]) / (omega_arr[right] - omega_arr[left])

    chunk = 256
    for start in range(0, len(tau_grid), chunk):
        stop = min(start + chunk, len(tau_grid))
        phase = np.outer(quad_omega, tau_grid[start:stop])
        cos_phase = np.cos(phase)
        corr[start:stop] = (
            (weighted_j * equilibrium_factor) @ cos_phase
            - 1j * (weighted_j @ np.sin(phase))
        )
        projected = np.zeros((len(omega_arr), stop - start), dtype=float)
        weighted_cos = weighted_j[:, None] * cos_phase
        np.add.at(projected, left, (1.0 - fraction)[:, None] * weighted_cos)
        np.add.at(projected, right, fraction[:, None] * weighted_cos)
        occupation_kernel[start:stop] = projected.T

    _kernel_cache[key] = (tau_grid, corr, occupation_kernel)
    return tau_grid, corr, occupation_kernel


def _base_correlation(
    tau: np.ndarray,
    spectral_density: str,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    if np.any(tau < -1.0e-12) or np.any(tau > pm.T_MAX + 1.0e-12):
        raise ValueError("The delay tau lies outside the precomputed interval.")
    tau_grid, corr_grid, _ = _build_kernel_lookup(spectral_density)
    tau_clip = np.clip(tau, 0.0, pm.T_MAX)
    real = np.interp(tau_clip, tau_grid, corr_grid.real)
    imag = np.interp(tau_clip, tau_grid, corr_grid.imag)
    return real + 1j * imag


def _occupation_correlation(
    tau: np.ndarray,
    N_vec: np.ndarray,
    spectral_density: str,
) -> np.ndarray:
    """Evaluate the N-dependent cosine integral from a precomputed projection."""
    tau_grid, _, kernel = _build_kernel_lookup(spectral_density)
    tau_clip = np.clip(np.asarray(tau, dtype=float), 0.0, pm.T_MAX)
    position = tau_clip / (tau_grid[1] - tau_grid[0])
    left = np.minimum(position.astype(int), len(tau_grid) - 2)
    fraction = position - left
    interpolated = (
        (1.0 - fraction)[:, None] * kernel[left]
        + fraction[:, None] * kernel[left + 1]
    )
    return interpolated @ N_vec


def correlation_parts(
    tau: np.ndarray,
    N_vec: np.ndarray,
    t: float,
    spectral_density: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Re C and int J sin(ω tau)dω for the current N(ω,t)."""
    tau = np.asarray(tau, dtype=float)
    density = resolve_spectral_density(spectral_density)
    occupation_change = N_vec - N0
    factor = time_factor(density, t, pm.spectral_parameters(density))
    if factor is not None:
        base = factor * _base_correlation(tau, density)
        re_c = base.real + 2.0 * factor * _occupation_correlation(
            tau, occupation_change, density
        )
        int_j_sin = -base.imag
        return re_c, int_j_sin

    N_quad = N_I(quad_omega) + np.interp(
        quad_omega, omega_arr, occupation_change
    )
    weighted_j = quad_weights * J(quad_omega, t, density=density)
    cos_phase, sin_phase = _phase_matrices("quadrature", quad_omega, tau)
    re_c = (weighted_j * (2.0 * N_quad + 1.0)) @ cos_phase
    int_j_sin = weighted_j @ sin_phase
    return re_c, int_j_sin


def C_corr(
    tau: float,
    N_arr: np.ndarray,
    t: float = 0.0,
    density: str | None = None,
) -> complex:
    """Evaluate the correlation function on the adaptive quadrature grid."""
    density = resolve_spectral_density(density)
    occupation_change = np.asarray(N_arr, dtype=float) - N0
    N_quad = N_I(quad_omega) + np.interp(quad_omega, omega_arr, occupation_change)
    phase = quad_omega * tau
    integrand = J(quad_omega, t, density=density) * (
        (2.0 * N_quad + 1.0) * np.cos(phase) - 1j * np.sin(phase)
    )
    return quad_weights @ integrand


def _history_weights(t1: np.ndarray) -> np.ndarray:
    if len(t1) == 1:
        return np.zeros(1)
    return trapezoid_weights(t1)


# ---------------------------------------------------------------------
# Right-hand sides of the coupled equations
# ---------------------------------------------------------------------

_lam2 = pm.LAMBDA**2


def RHS_sigma(
    sigma_all: np.ndarray,
    N_vec: np.ndarray,
    t_eval: float,
    t1_arr: np.ndarray,
    c_hist: np.ndarray,
    c_now: np.ndarray,
    density: str,
) -> np.ndarray:
    if len(t1_arr) < 2:
        return np.zeros_like(sigma_all)

    tau = t_eval - t1_arr
    re_c, int_j_sin = correlation_parts(tau, N_vec, t_eval, density)
    wt = _history_weights(t1_arr)
    A_re = np.einsum("h,h,hsl->sl", wt, re_c, c_hist)
    A_im = np.einsum("h,h,hsl->sl", wt, int_j_sin, c_hist)

    M = np.einsum("kab,lbg,pk,pl->pag", eps, eps, c_now, A_re)
    term_thermal = np.einsum("pag,pg->pa", M, sigma_all)
    term_sine = np.einsum("kal,pk,pl->pa", eps, c_now, A_im)
    result = 4.0 * _lam2 * ETA2_j[:, None] * (term_thermal + term_sine)

    # Shared-bath cross-spin term under factorization of spin averages.
    projected = np.einsum("il,il->i", A_im, sigma_all)
    cross_sum = np.sum(ETA_j * projected) - ETA_j * projected
    cross_vec = np.einsum("kad,pk,pd->pa", eps, c_now, sigma_all)
    result += 4.0 * _lam2 * (ETA_j * cross_sum)[:, None] * cross_vec
    return result


def RHS_N(
    N_vec: np.ndarray,
    sigma_all: np.ndarray,
    t_eval: float,
    t1_arr: np.ndarray,
    c_hist: np.ndarray,
    c_now: np.ndarray,
    density: str,
) -> np.ndarray:
    if len(t1_arr) < 2:
        return np.zeros_like(N_vec)

    d_hist = np.einsum("hsl,sl->hs", c_hist, sigma_all)
    b_now = np.einsum("sk,sk->s", c_now, sigma_all)

    diagonal = np.einsum("hsl,sl->h", c_hist, ETA2_j[:, None] * c_now)
    eta_d = d_hist @ ETA_j
    eta_b = np.sum(ETA_j * b_now)
    off_diagonal = eta_d * eta_b - (
        d_hist * (ETA2_j * b_now)[None, :]
    ).sum(axis=1)
    W_cos = diagonal + off_diagonal

    levi_vec = np.einsum("lkg,sk,sg->sl", eps, c_now, sigma_all)
    W_sin = np.einsum("hsl,sl->h", c_hist, ETA2_j[:, None] * levi_vec)

    tau = t_eval - t1_arr
    wt = _history_weights(t1_arr)
    cos_phase, sin_phase = _phase_matrices("physical", omega_arr, tau)
    integral_cos = cos_phase @ (wt * W_cos)
    integral_sin = sin_phase @ (wt * W_sin)

    g2 = coupling_squared(omega_arr, t=t_eval, density=density)
    return 2.0 * _lam2 * g2 * (
        integral_cos - (2.0 * N_vec + 1.0) * integral_sin
    )


def _rhs_coupled(
    sigma_all: np.ndarray,
    N_vec: np.ndarray,
    t_eval: float,
    t1_arr: np.ndarray,
    c_hist: np.ndarray,
    density: str,
    evolve_occupations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    c_now = c_hist[-1]
    rhs_n = (
        RHS_N(N_vec, sigma_all, t_eval, t1_arr, c_hist, c_now, density)
        if evolve_occupations
        else np.zeros_like(N_vec)
    )
    return (
        RHS_sigma(sigma_all, N_vec, t_eval, t1_arr, c_hist, c_now, density),
        rhs_n,
    )


# ---------------------------------------------------------------------
# Coupled TR-BDF2 integrator and diagnostics
# ---------------------------------------------------------------------

GAMMA_TRBDF2 = 2.0 - np.sqrt(2.0)


@dataclass
class SolverDiagnostics:
    spectral_density: str
    occupation_mode: str
    max_fixed_point_iterations: int = 0
    max_fixed_point_residual: float = 0.0
    max_bloch_norm: float = 0.0
    min_occupation: float = np.inf
    solver_seconds: float = 0.0


def _state_residual(
    sigma_new: np.ndarray,
    sigma_old: np.ndarray,
    N_new: np.ndarray,
    N_old: np.ndarray,
) -> float:
    sigma_scale = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.maximum(
        np.abs(sigma_new), np.abs(sigma_old)
    )
    occupation_scale = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.maximum(
        np.abs(N_new), np.abs(N_old)
    )
    sigma_residual = np.max(np.abs(sigma_new - sigma_old) / sigma_scale)
    occupation_residual = np.max(np.abs(N_new - N_old) / occupation_scale)
    return float(max(sigma_residual, occupation_residual))


def _handle_nonconvergence(stage: str, step: int, residual: float) -> None:
    message = (
        f"TR-BDF2 did not converge: step={step}, stage={stage}, "
        f"scaled residual={residual:.3e}, iterations={MAX_ITERATIONS}. Increase time.points."
    )
    raise RuntimeError(message)


def trbdf2_coupled_step(
    sigma_all: np.ndarray,
    N_vec: np.ndarray,
    step: int,
    density: str,
    diagnostics: SolverDiagnostics,
    evolve_occupations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    g = GAMMA_TRBDF2
    t0 = t_arr[step]
    t_mid = t0 + g * dt
    t_end = t_arr[step + 1]

    t1_start = t_arr[: step + 1]
    c_start = c_coef_arr[: step + 1]
    F0_sigma, F0_N = _rhs_coupled(
        sigma_all, N_vec, t0, t1_start, c_start, density, evolve_occupations
    )

    t1_mid = np.append(t1_start, t_mid)
    c_mid = np.concatenate((c_start, c_coef_all(t_mid)[None]), axis=0)
    sigma_mid = sigma_all + g * dt * F0_sigma
    N_mid = N_vec + g * dt * F0_N

    residual = np.inf
    for iteration in range(1, MAX_ITERATIONS + 1):
        F_mid_sigma, F_mid_N = _rhs_coupled(
            sigma_mid, N_mid, t_mid, t1_mid, c_mid, density, evolve_occupations
        )
        sigma_candidate = sigma_all + 0.5 * g * dt * (F0_sigma + F_mid_sigma)
        N_candidate = N_vec + 0.5 * g * dt * (F0_N + F_mid_N)
        sigma_next = sigma_candidate
        N_next = N_candidate if evolve_occupations else N_vec
        residual = _state_residual(sigma_next, sigma_mid, N_next, N_mid)
        sigma_mid, N_mid = sigma_next, N_next
        if residual <= 1.0:
            break
    else:
        _handle_nonconvergence("TR", step, residual)
    diagnostics.max_fixed_point_iterations = max(
        diagnostics.max_fixed_point_iterations, iteration
    )
    diagnostics.max_fixed_point_residual = max(
        diagnostics.max_fixed_point_residual, residual
    )

    t1_end = np.append(t1_mid, t_end)
    c_end = np.concatenate((c_mid, c_coef_all(t_end)[None]), axis=0)
    a1 = 1.0 / (g * (2.0 - g))
    a2 = (1.0 - g) ** 2 / (g * (2.0 - g))
    a3 = dt * (1.0 - g) / (2.0 - g)

    sigma_end = a1 * sigma_mid - a2 * sigma_all
    N_end = a1 * N_mid - a2 * N_vec
    if not evolve_occupations:
        N_end = N_vec.copy()
    residual = np.inf
    for iteration in range(1, MAX_ITERATIONS + 1):
        F_end_sigma, F_end_N = _rhs_coupled(
            sigma_end, N_end, t_end, t1_end, c_end, density, evolve_occupations
        )
        sigma_candidate = a1 * sigma_mid - a2 * sigma_all + a3 * F_end_sigma
        N_candidate = a1 * N_mid - a2 * N_vec + a3 * F_end_N
        sigma_next = sigma_candidate
        N_next = N_candidate if evolve_occupations else N_vec
        residual = _state_residual(sigma_next, sigma_end, N_next, N_end)
        sigma_end, N_end = sigma_next, N_next
        if residual <= 1.0:
            break
    else:
        _handle_nonconvergence("BDF2", step, residual)
    diagnostics.max_fixed_point_iterations = max(
        diagnostics.max_fixed_point_iterations, iteration
    )
    diagnostics.max_fixed_point_residual = max(
        diagnostics.max_fixed_point_residual, residual
    )

    if not (np.all(np.isfinite(sigma_end)) and np.all(np.isfinite(N_end))):
        raise FloatingPointError(f"Non-finite values appeared at step {step}.")
    return sigma_end, N_end


# ---------------------------------------------------------------------
# Transformation to the Schrodinger picture and main solver
# ---------------------------------------------------------------------

def rotation_matrix(j: int, t: float) -> np.ndarray:
    frequency = sqrt_Omega_j[j]
    if frequency == 0.0:
        return np.eye(3)
    generator = (-DELTA_j[j] * sx + H_FIELD_j[j] * sz) / frequency
    U = (
        np.cos(0.5 * frequency * t) * np.eye(2)
        - 1j * np.sin(0.5 * frequency * t) * generator
    )
    result = np.zeros((3, 3))
    for alpha in range(3):
        transformed = U.conj().T @ pauli[alpha] @ U
        for beta in range(3):
            result[alpha, beta] = 0.5 * np.real(np.trace(transformed @ pauli[beta]))
    return result


R_arr = np.asarray(
    [[rotation_matrix(j, t) for j in range(N_SPINS)] for t in t_arr]
)


def solve(
    return_diagnostics: bool = False,
    occupation_mode: str = "dynamic",
    spectral_density: str | None = None,
):
    """Solve TCL2 with either dynamic or frozen mode occupations."""
    if occupation_mode not in {"dynamic", "frozen"}:
        raise ValueError("occupation_mode must be 'dynamic' or 'frozen'.")
    evolve_occupations = occupation_mode == "dynamic"
    density = resolve_spectral_density(spectral_density)
    configure_quadrature(density)
    adiabaticity = adiabaticity_parameter(density)
    if adiabaticity > 0.1:
        warnings.warn(
            f"The adiabatic estimate tau_B*|d ln J/dt|={adiabaticity:.3g} "
            "is not small.",
            RuntimeWarning,
            stacklevel=2,
        )
    if time_factor(density, 0.0, pm.spectral_parameters(density)) is not None:
        _build_kernel_lookup(density)

    diagnostics = SolverDiagnostics(
        spectral_density=density,
        occupation_mode=occupation_mode,
        max_bloch_norm=float(np.max(np.linalg.norm(BETA0, axis=1))),
        min_occupation=float(np.min(N0)),
    )
    sigma_all = BETA0.copy()
    N_vec = N0.copy()

    sigma_S_hist = [np.einsum("jab,jb->ja", R_arr[0], sigma_all)]
    N_hist = [N_vec.copy()]

    print(
        f"[TCL2] spectral_density={density}, spins={N_SPINS}, dt={dt:.5f}, "
        f"modes={len(omega_arr)}, quadrature={len(quad_omega)}, "
        f"occupations={occupation_mode}"
    )
    solver_start = perf_counter()
    for step in range(pm.T_POINTS - 1):
        sigma_all, N_vec = trbdf2_coupled_step(
            sigma_all, N_vec, step, density, diagnostics, evolve_occupations
        )
        sigma_s = np.einsum("jab,jb->ja", R_arr[step + 1], sigma_all)
        sigma_S_hist.append(sigma_s)
        N_hist.append(N_vec.copy())

        diagnostics.max_bloch_norm = max(
            diagnostics.max_bloch_norm, float(np.max(np.linalg.norm(sigma_s, axis=1)))
        )
        diagnostics.min_occupation = min(
            diagnostics.min_occupation, float(np.min(N_vec))
        )
        report_stride = max(1, (pm.T_POINTS - 1) // 10)
        if (step + 1) % report_stride == 0:
            print(
                f"  t={t_arr[step + 1]:.2f}/{pm.T_MAX:.2f}, "
                f"<sz>={sigma_s[0, 2]:+.5f}, min N={np.min(N_vec):+.3e}"
            )
    diagnostics.solver_seconds = perf_counter() - solver_start

    sigma_S = np.asarray(sigma_S_hist)
    N_t = np.asarray(N_hist)
    purity = 0.5 * (1.0 + np.sum(sigma_S**2, axis=2))
    if diagnostics.max_bloch_norm > 1.0 + PHYSICALITY_TOLERANCE:
        warnings.warn(
            f"TCL2 violated |Bloch|<=1: max={diagnostics.max_bloch_norm:.8f}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if diagnostics.min_occupation < -PHYSICALITY_TOLERANCE:
        warnings.warn(
            f"TCL2 produced a negative occupation: min N={diagnostics.min_occupation:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    result = (t_arr, sigma_S, purity, N_t)
    if return_diagnostics:
        return (*result, diagnostics)
    return result


def solve_unitary():
    sigma_S = np.zeros((pm.T_POINTS, N_SPINS, 3))
    for j in range(N_SPINS):
        for i in range(pm.T_POINTS):
            sigma_S[i, j] = R_arr[i, j] @ BETA0[j]
    return t_arr, sigma_S


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_results(
    t,
    sigma_S,
    purity,
    N_t,
    t_u,
    sigma_u,
    file_path: str = ".",
    density: str | None = None,
    show: bool = False,
):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    density = resolve_spectral_density(density)
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    colors = plt.cm.tab10(np.linspace(0, 1, max(N_SPINS, 1)))

    for idx, label in enumerate(
        [r"$\langle\sigma_x\rangle$", r"$\langle\sigma_y\rangle$", r"$\langle\sigma_z\rangle$"]
    ):
        ax = fig.add_subplot(gs[0, idx])
        for j in range(N_SPINS):
            spin_label = f"spin {j + 1}" if N_SPINS > 1 else None
            ax.plot(t, sigma_S[:, j, idx], color=colors[j], lw=1.8, label=spin_label)
            ax.plot(t_u, sigma_u[:, j, idx], "--", color=colors[j], lw=1.0, alpha=0.5)
        ax.set(xlabel="$t$", ylabel=label, xlim=(0, pm.T_MAX))
        if N_SPINS > 1:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for j in range(N_SPINS):
        spin_label = f"spin {j + 1}" if N_SPINS > 1 else None
        ax.plot(t, purity[:, j], color=colors[j], lw=1.8, label=spin_label)
    ax.axhline(1.0, ls="--", color="black", lw=0.8)
    ax.axhline(0.5, ls="--", color="grey", lw=0.8)
    ax.set(
        xlabel="$t$",
        ylabel=r"$\mathrm{Tr}(\rho^2)$",
        xlim=(0, pm.T_MAX),
        ylim=(0, max(1.05, float(np.max(purity)) * 1.05)),
    )
    if N_SPINS > 1:
        ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    w_plot = np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, 600)
    ax.plot(w_plot, J(w_plot, 0.0, density=density), color="C4", lw=2)
    ax.set(
        xlabel=r"$\omega$",
        ylabel=r"$J(\omega,0)$",
        xlim=(0, pm.OMEGA_MAX),
        ylim=(0, None),
    )
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    for j in range(N_SPINS):
        ax.plot(t, np.linalg.norm(sigma_S[:, j], axis=1), color=colors[j], lw=1.8)
    ax.axhline(1.0, ls=":", color="grey", lw=0.8)
    ax.set(
        xlabel="$t$",
        ylabel=r"$|\boldsymbol{\beta}|$",
        xlim=(0, pm.T_MAX),
        ylim=(0, None),
    )
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :2])
    probe_frequencies = tuple(sqrt_Omega_j) + pm.PROBE_FREQUENCIES
    resonance_indices = {
        int(np.argmin(np.abs(omega_arr - frequency))) for frequency in probe_frequencies
    }
    selected_indices = sorted(resonance_indices)
    for index in selected_indices:
        ax.plot(t, N_t[:, index], lw=1.8, label=rf"$\omega={omega_arr[index]:.3f}$")
    occupation_minimum = float(np.min(N_t[:, selected_indices]))
    occupation_bottom = (
        occupation_minimum if occupation_minimum < -PHYSICALITY_TOLERANCE else 0.0
    )
    ax.set(
        xlabel="$t$",
        ylabel=r"$N(\omega,t)$",
        xlim=(0, pm.T_MAX),
        ylim=(occupation_bottom, None),
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    tau_plot = np.linspace(0.0, min(10.0, pm.T_MAX), 300)
    corr = np.asarray([C_corr(tau, N0, density=density) for tau in tau_plot])
    ax.plot(tau_plot, corr.real, label=r"Re $C$", color="C5")
    ax.plot(tau_plot, corr.imag, "--", label=r"Im $C$", color="C6")
    ax.set(
        xlabel=r"$\tau$",
        ylabel=r"$C(\tau)$",
        xlim=(0, float(tau_plot[-1])),
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    os.makedirs(file_path, exist_ok=True)
    output = os.path.join(file_path, "spin_boson_results.png")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


if __name__ == "__main__":
    raise SystemExit("Use run.py to select a configuration and output mode.")
