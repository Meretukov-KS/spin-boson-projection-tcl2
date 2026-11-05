"""Compare discrete-mode Eq. (16) with full spin-boson evolution."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import argparse
import json
import warnings

import numpy as np

from conf import DEFAULT_CONFIG_PATH, load


def c_coef_spin(t, pm):
    """Coefficients of exp(i Hs t) sigma_z exp(-i Hs t)."""
    d, h = pm.SPINS[0].DELTA, pm.SPINS[0].H_FIELD
    omega = np.hypot(d, h)
    if omega == 0:
        return np.array([0., 0., 1.])
    return np.array([d*h*(np.cos(omega*t)-1)/omega**2,
                     -d*np.sin(omega*t)/omega,
                     (d*d*np.cos(omega*t)+h*h)/omega**2])


def mode_integrals(t, pm):
    """Integrate c(u) cos/sin(w_m*(t-u)) over u=0..t analytically.

    The sinc expression also handles exact resonance without division by zero.
    """
    d, h = pm.SPINS[0].DELTA, pm.SPINS[0].H_FIELD
    omega = np.hypot(d, h)
    w = np.array([m.w for m in pm.MODES])
    def integral(k):
        return t*np.exp(0.5j*k*t)*np.sinc(k*t/(2*np.pi))
    if omega == 0:
        kernel = integral(-w)[:, None]*np.array([0., 0., 1.])
    else:
        a = np.array([-d*h, 0., h*h])/omega**2
        b = np.array([d*h, 0., d*d])/omega**2
        c = np.array([0., -d/omega, 0.])
        plus, minus = integral(omega-w), integral(-omega-w)
        kernel = (integral(-w)[:, None]*a + 0.5*(plus+minus)[:, None]*b
                  + ((plus-minus)/(2j))[:, None]*c)
    kernel *= np.exp(1j*w*t)[:, None]
    return kernel.real, kernel.imag


def rhs_projected(t, state, pm):
    """Eq. (16), state = [sx, sy, sz, N_0, ..., N_(M-1)].

    Spin components are in the interaction picture. No continuum or
    rotating-wave approximation is made; all means on the RHS are at t.
    """
    s, n = state[:3], state[3:]
    c = c_coef_spin(t, pm)
    cosine, sine = mode_integrals(t, pm)
    g2 = np.array([m.g**2 for m in pm.MODES])
    a = (g2*(2*n+1)) @ cosine
    b = g2 @ sine
    ds = 4*pm.LAMBDA**2*(a*np.dot(c, s)-s*np.dot(c, a)-np.cross(c, b))
    dn = 2*pm.LAMBDA**2*g2*(cosine@c-(2*n+1)*(np.cross(sine, c)@s))
    return np.concatenate((ds, dn))


def solver_step(t, state, dt, pm):
    """One fourth-order Runge-Kutta step."""
    k1 = rhs_projected(t, state, pm)
    k2 = rhs_projected(t+dt/2, state+dt*k1/2, pm)
    k3 = rhs_projected(t+dt/2, state+dt*k2/2, pm)
    k4 = rhs_projected(t+dt, state+dt*k3, pm)
    return state+dt*(k1+2*k2+2*k3+k4)/6


def initial_bath(pm):
    """Truncated thermal probabilities and their means, shared by both solvers.

    The projected Gaussian state has the same means, not the same higher
    moments as the truncated state. Increase levels to remove this cutoff error.
    """
    probabilities, means, tails, ideal = [], [], [], []
    for m in pm.MODES:
        x = pm.BETA*m.w
        p = np.zeros(m.levels)
        if np.isinf(x):
            p[0] = 1.
            mean = tail = 0.
        else:
            p = np.exp(-x*np.arange(m.levels))
            p /= p.sum()
            mean = 1/np.expm1(x) if x < 700 else 0.
            tail = float(np.exp(-x*m.levels))
        probabilities.append(p)
        means.append(np.dot(np.arange(m.levels), p))
        tails.append(tail)
        ideal.append(mean)
    return probabilities, np.array(means), np.array(tails), np.array(ideal)


def solve_projected(pm):
    """Integrate Eq. (16); return spin means in the Schrodinger picture."""
    start = perf_counter()
    t = np.linspace(0., pm.T_MAX, pm.T_POINT)
    state = np.empty((len(t), 3+len(pm.MODES)))
    state[0] = np.concatenate((pm.SPINS[0].PAULI_I, initial_bath(pm)[1]))
    for i, dt in enumerate(np.diff(t)):
        state[i+1] = solver_step(t[i], state[i], dt, pm)
        if not np.all(np.isfinite(state[i+1])):
            raise FloatingPointError("Projected solution diverged; reduce dt or coupling/time.")
    spin = pm.SPINS[0]
    axis = np.array([-spin.DELTA, 0., spin.H_FIELD])
    omega = np.linalg.norm(axis)
    s = state[:, :3]
    if omega:
        axis /= omega
        angle = omega*t[:, None]
        sigma = (s*np.cos(angle)+np.cross(axis, s)*np.sin(angle)
                 +(s@axis)[:, None]*axis*(1-np.cos(angle)))
    else:
        sigma = s.copy()
    return dict(t=t, sigma=sigma, occupations=state[:, 3:],
                sigma_interaction=s.copy(), seconds=perf_counter()-start)


def solve_exact(pm):
    """Full Hamiltonian evolution, subject to Fock and numerical truncations."""
    import qutip as qt
    start = perf_counter()
    dims = [2]+[m.levels for m in pm.MODES]
    def embed(op, index):
        factors = [qt.qeye(d) for d in dims]
        factors[index] = op
        return qt.tensor(factors)
    spin = pm.SPINS[0]
    pauli = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    sigma = [embed(op, 0) for op in pauli]
    hs = -spin.DELTA*sigma[0]/2+spin.H_FIELD*sigma[2]/2
    hb, interaction = 0*hs, 0*hs
    numbers, upper = [], []
    for i, m in enumerate(pm.MODES, 1):
        b = embed(qt.destroy(m.levels), i)
        numbers.append(b.dag()*b)
        hb += m.w*numbers[-1]
        interaction += pm.LAMBDA*m.g*sigma[2]*(b+b.dag())
        upper.append(embed(qt.basis(m.levels, m.levels-1).proj(), i))
    hamiltonian = hs+hb+interaction
    rho_spin = (qt.qeye(2)+sum(v*op for v, op in zip(spin.PAULI_I, pauli)))/2
    rho0 = qt.tensor([rho_spin]+[qt.Qobj(np.diag(p)) for p in initial_bath(pm)[0]])
    t = np.linspace(0., pm.T_MAX, pm.T_POINT)
    max_step = min(pm.T_MAX/(pm.T_POINT-1), 0.2/max(m.w for m in pm.MODES))
    raw = qt.mesolve(hamiltonian, rho0, t, c_ops=[],
                    e_ops=sigma+numbers+[hs, hb, interaction, hamiltonian]+upper,
                    options=dict(atol=1e-10, rtol=1e-8, nsteps=100000,
                                 max_step=max_step, store_states=False))
    values = np.asarray(raw.expect).real.T
    m = len(pm.MODES)
    return dict(t=t, sigma=values[:, :3], occupations=values[:, 3:3+m],
                energies=values[:, 3+m:7+m], top_level_population=values[:, 7+m:],
                seconds=perf_counter()-start, qutip_version=qt.__version__)


def plot_and_save(pm, projected, exact, folder):
    """Save all calculated observables, parameters, diagnostics, and shared plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/"solutions.npz", t=projected["t"],
                        **{f"projected_{k}": v for k, v in projected.items() if isinstance(v, np.ndarray) and k != "t"},
                        **{f"exact_{k}": v for k, v in exact.items() if isinstance(v, np.ndarray) and k != "t"})
    _, n0, tails, ideal = initial_bath(pm)
    summary = dict(parameters=asdict(pm),
                   initial_mode_means=n0.tolist(), infinite_thermal_means=ideal.tolist(),
                   omitted_initial_thermal_probability=tails.tolist(),
                   initial_state_note="Exact: normalized truncated thermal product. Projected: Gaussian closure with identical initial means.",
                   projected_seconds=projected["seconds"], exact_seconds=exact["seconds"],
                   qutip_version=exact["qutip_version"],
                   projected_step="RK4; analytic time integrals; time.points defines dt",
                   exact_tolerances=dict(atol=1e-10, rtol=1e-8),
                   max_spin_difference=float(np.max(np.linalg.norm(projected["sigma"]-exact["sigma"], axis=1))),
                   max_population_difference_by_mode=np.max(np.abs(projected["occupations"]-exact["occupations"]), axis=0).tolist(),
                   max_top_level_population=np.max(exact["top_level_population"], axis=0).tolist(),
                   exact_max_energy_drift=float(np.max(np.abs(exact["energies"][:, 3]-exact["energies"][0, 3]))),
                   projected_min_population=float(projected["occupations"].min()),
                   projected_max_bloch_norm=float(np.linalg.norm(projected["sigma"], axis=1).max()))
    if np.isinf(pm.BETA):
        summary["parameters"]["BETA"] = "inf"
    (folder/"parameters_and_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    columns = [projected["t"], projected["sigma"], exact["sigma"],
               projected["occupations"], exact["occupations"], exact["energies"], exact["top_level_population"]]
    names = ["t"]+[f"{method}_s{a}" for method in ("projected", "exact") for a in "xyz"]
    names += [f"{method}_N{i}" for method in ("projected", "exact") for i in range(len(pm.MODES))]
    names += ["exact_E_spin", "exact_E_bath", "exact_E_interaction", "exact_E_total"]
    names += [f"exact_top_level_{i}" for i in range(len(pm.MODES))]
    np.savetxt(folder/"observables.csv", np.column_stack(columns), delimiter=",", header=",".join(names), comments="")
    fig, axes = plt.subplots(3+len(pm.MODES), 1, figsize=(8, 2.5*(3+len(pm.MODES))), sharex=True)
    data_p = np.column_stack((projected["sigma"], projected["occupations"]))
    data_e = np.column_stack((exact["sigma"], exact["occupations"]))
    labels = [rf"$\langle\sigma_{a}\rangle$" for a in "xyz"]
    labels += [rf"$\langle N_{{{i}}}\rangle$ ($\omega={m.w:g}$)" for i, m in enumerate(pm.MODES)]
    for i, ax in enumerate(axes):
        ax.plot(projected["t"], data_e[:, i], color="black", label="Full model (QuTiP)")
        ax.plot(projected["t"], data_p[:, i], "--", label="Projected TCL2")
        ax.set_ylabel(labels[i])
        ax.set_xlim(0, pm.T_MAX)
        if i >= 3 and min(data_e[:, i].min(), data_p[:, i].min()) >= 0:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(folder/"comparison.png", dpi=180)
    fig.savefig(folder/"comparison.pdf")
    plt.close(fig)
    if np.max(tails) > 1e-8 or np.max(exact["top_level_population"]) > 1e-6:
        warnings.warn("Fock cutoff may be insufficient: increase modes.levels and compare trajectories.")
    if summary["projected_min_population"] < -1e-8 or summary["projected_max_bloch_norm"] > 1+1e-6:
        warnings.warn("Projected dynamics left the physical domain; saved values are not clipped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    pm = load(args.config)
    print("Solving discrete projected equations...", flush=True)
    projected = solve_projected(pm)
    print("Solving the full model with QuTiP...", flush=True)
    exact = solve_exact(pm)
    folder = Path(__file__).parent/"results"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    plot_and_save(pm, projected, exact, folder)
    (folder/"params.toml").write_bytes(args.config.read_bytes())
    print(f"Projected: {projected['seconds']:.3f} s; QuTiP: {exact['seconds']:.3f} s")
    print(f"Saved: {folder}")


if __name__ == "__main__":
    main()
