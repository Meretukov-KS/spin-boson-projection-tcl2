"""HEOM solver for the matched Drude comparison model.

HEOM is not used as a reference for the article's primary Ohmic density.
The comparison uses
J(omega)=2*DRUDE_REORG*DRUDE_GAMMA*omega/(omega^2+DRUDE_GAMMA^2),
with the external LAMBDA^2 factor absorbed into the reorganization energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np

from config import get_config
pm = get_config()

# Internal HEOM defaults, not configurable through TOML.
N_DEPTH = 5
N_EXPANSION = 6
USE_PADE = True


@dataclass
class HEOMResult:
    t: np.ndarray
    sigma: np.ndarray
    purity: np.ndarray
    depth: int
    n_exp: int
    use_pade: bool
    setup_seconds: float
    solver_seconds: float
    total_seconds: float
    qutip_version: str
    terminator_delta: float
    max_bloch_norm: float
    minimum_purity: float


def _qutip_objects():
    try:
        import qutip as qt
        from qutip.solver.heom import (
            DrudeLorentzBath,
            DrudeLorentzPadeBath,
            HEOMSolver,
        )
    except ImportError as exc:
        raise RuntimeError("HEOM requires the qutip package.") from exc
    return qt, HEOMSolver, DrudeLorentzBath, DrudeLorentzPadeBath


def solve_heom(
    depth: int = N_DEPTH,
    n_exp: int = N_EXPANSION,
    use_pade: bool = USE_PADE,
) -> HEOMResult:
    """Run one HEOM calculation without import-time side effects."""
    total_start = perf_counter()
    qt, HEOMSolver, DrudeLorentzBath, DrudeLorentzPadeBath = _qutip_objects()
    parameters = pm.spectral_parameters("drude")
    reorganization = parameters["drude_reorganization"]
    gamma = parameters["drude_gamma"]
    if len(pm.SPINS) != 1 or pm.TEMP == 0 or pm.Delta_N != 0 or parameters["time_amplitude"] != 0:
        raise ValueError("HEOM requires one spin, a stationary density, and a thermal bath at T > 0.")
    spin = pm.SPINS[0]

    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    H_s = -0.5 * spin.DELTA * sx + 0.5 * spin.H_FIELD * sz
    bloch = np.asarray(spin.PAULI_I, dtype=float)
    rho0 = 0.5 * (
        qt.identity(2) + bloch[0] * sx + bloch[1] * sy + bloch[2] * sz
    )

    bath_class = DrudeLorentzPadeBath if use_pade else DrudeLorentzBath
    bath = bath_class(
        Q=sz,
        T=pm.TEMP,
        # Absorb both coupling factors into the bath correlation amplitude.
        lam=(pm.LAMBDA * spin.COUPLING)**2 * reorganization,
        gamma=gamma,
        Nk=n_exp,
    )
    t = np.linspace(0.0, pm.T_MAX, pm.T_POINTS)
    terminator_delta, terminator = bath.terminator()
    system_liouvillian = qt.liouvillian(H_s) + terminator
    solver = HEOMSolver(H=system_liouvillian, bath=bath, max_depth=depth)
    setup_seconds = perf_counter() - total_start
    solver_start = perf_counter()
    raw = solver.run(rho0, t, e_ops=[sx, sy, sz])
    solver_seconds = perf_counter() - solver_start
    sigma = np.column_stack(raw.expect).real
    purity = 0.5 * (1.0 + np.sum(sigma**2, axis=1))
    max_bloch_norm = float(np.max(np.linalg.norm(sigma, axis=1)))
    if max_bloch_norm > 1.0 + 1e-7:
        warnings.warn(
            f"HEOM produced a Bloch-vector norm of {max_bloch_norm:.8f}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return HEOMResult(
        t=t,
        sigma=sigma,
        purity=purity,
        depth=depth,
        n_exp=n_exp,
        use_pade=use_pade,
        setup_seconds=setup_seconds,
        solver_seconds=solver_seconds,
        total_seconds=perf_counter() - total_start,
        qutip_version=str(qt.__version__),
        terminator_delta=float(terminator_delta.real),
        max_bloch_norm=max_bloch_norm,
        minimum_purity=float(np.min(purity)),
    )
