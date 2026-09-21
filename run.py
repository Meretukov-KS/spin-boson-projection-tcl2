"""Command-line entry point for TCL2 calculations and HEOM comparisons."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from config import DEFAULT_CONFIG_PATH, get_config, set_active_config
CODE_DIR = Path(__file__).resolve().parent
VALID_MODES = ("numeric", "compare", "compare-frozen")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="numeric", choices=VALID_MODES)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a TOML configuration file.",
    )
    return parser.parse_args()


def output_directory(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    path = CODE_DIR / mode / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _save_config(path: Path) -> None:
    config = get_config()
    shutil.copy2(config.SOURCE_PATH, path / "config.toml")


def _save_bath_data(
    path: Path,
    tcl2,
    spectral_density: str,
    occupations: np.ndarray | None,
) -> None:
    physical_omega = tcl2.omega_arr
    initial_j = tcl2.J(physical_omega, 0.0, density=spectral_density)
    initial_n = tcl2.N0
    physical_mode_density = tcl2.mode_density(physical_omega)
    coupling = tcl2.coupling_squared(
        physical_omega, 0.0, density=spectral_density
    )
    quadrature_n0 = tcl2.N_I(tcl2.quad_omega)
    final_occupation = initial_n if occupations is None else occupations[-1]
    np.savez_compressed(
        path / "bath_data.npz",
        physical_omega=physical_omega,
        spectral_density=initial_j,
        initial_occupation=initial_n,
        mode_density=physical_mode_density,
        coupling_squared=coupling,
        quadrature_omega=tcl2.quad_omega,
        quadrature_weights=tcl2.quad_weights,
        quadrature_spectral_density=tcl2.J(
            tcl2.quad_omega, 0.0, density=spectral_density
        ),
        quadrature_initial_occupation=quadrature_n0,
        final_occupation=final_occupation,
    )
    table = np.column_stack(
        (physical_omega, initial_j, initial_n, physical_mode_density, coupling)
    )
    np.savetxt(
        path / "bath_initial.csv",
        table,
        delimiter=",",
        header="omega,J,N0,p,g_squared",
        comments="",
    )


def _write_summary(path: Path, metadata: dict[str, Any]) -> None:
    timing = metadata.get("timing_seconds", {})
    lines = [
        f"Mode: {metadata['mode']}",
        f"Configuration: {metadata['config_name']}",
        f"Occupation mode: {metadata.get('occupation_mode') or 'not applicable'}",
        "",
    ]
    if "tcl2_solver" in timing:
        lines.extend(
            [
                "TCL2:",
                f"  equation solver: {timing['tcl2_solver']:.6f} s",
                f"  complete TCL2 stage: {timing['tcl2_total']:.6f} s",
                "",
            ]
        )
    if "heom_solver" in timing:
        lines.extend(
            [
                "HEOM:",
                f"  setup: {timing['heom_setup']:.6f} s",
                f"  equation solver: {timing['heom_solver']:.6f} s",
                f"  complete HEOM stage: {timing['heom_total']:.6f} s",
                "",
            ]
        )
    if "plotting" in timing:
        lines.append(f"Plotting and saving: {timing['plotting']:.6f} s")
    if "total" in timing:
        lines.append(f"Complete run: {timing['total']:.6f} s")
    (path / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_metadata(mode: str, occupation_mode: str | None = None) -> dict[str, Any]:
    config = get_config()
    return {
        "mode": mode,
        "occupation_mode": occupation_mode,
        "config_name": config.CONFIG_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }


def _comparison_figure(
    path: Path,
    tcl_data,
    heom_result,
    occupation_mode: str,
    density: str,
) -> tuple[Path, dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numerical as tcl2

    t, sigma, purity, _ = tcl_data
    if len(t) != len(heom_result.t) or not np.allclose(t, heom_result.t):
        raise ValueError("TCL2 and HEOM time grids do not match.")
    sigma_tcl = sigma[:, 0]
    differences = sigma_tcl - heom_result.sigma
    norm_difference = np.linalg.norm(differences, axis=1)
    purity_difference = purity[:, 0] - heom_result.purity
    tcl_label = "TCL2"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    labels = (
        r"$\langle\sigma_x\rangle$",
        r"$\langle\sigma_y\rangle$",
        r"$\langle\sigma_z\rangle$",
    )
    for index, (label, color) in enumerate(zip(labels, ("C0", "C1", "C2"))):
        ax = axes[0, index]
        ax.plot(t, heom_result.sigma[:, index], color=color, lw=2.0, label="HEOM")
        ax.plot(t, sigma_tcl[:, index], "--", color=color, lw=1.7, label=tcl_label)
        ax.set(xlabel="$t$", ylabel=label, xlim=(0, float(t[-1])))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[1, 0].plot(t, heom_result.purity, color="C3", lw=2.0, label="HEOM")
    axes[1, 0].plot(t, purity[:, 0], "--", color="C3", lw=1.7, label=tcl_label)
    axes[1, 0].set(
        xlabel="$t$",
        ylabel=r"$\mathrm{Tr}(\rho^2)$",
        xlim=(0, float(t[-1])),
        ylim=(0, None),
    )
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, norm_difference, color="C7", lw=1.8)
    axes[1, 1].set(
        xlabel="$t$",
        ylabel=r"$\|\boldsymbol{\sigma}_{\mathrm{TCL2}}-\boldsymbol{\sigma}_{\mathrm{HEOM}}\|$",
        xlim=(0, float(t[-1])),
        ylim=(0, None),
    )
    axes[1, 1].grid(alpha=0.3)

    w = np.linspace(tcl2.pm.OMEGA_MIN, tcl2.pm.OMEGA_MAX, 600)
    full_j = tcl2.pm.LAMBDA**2 * tcl2.J(w, 0.0, density=density)
    axes[1, 2].plot(w, full_j, color="C5", lw=2.0)
    axes[1, 2].set(
        xlabel=r"$\omega$",
        ylabel=r"$\lambda^2J(\omega)$",
        xlim=(0, float(w[-1])),
        ylim=(0, None),
    )
    axes[1, 2].grid(alpha=0.3)
    fig.tight_layout()
    file_name = "compare_dynamic_N.png" if occupation_mode == "dynamic" else "compare_frozen_N.png"
    figure_path = path / file_name
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "max_abs_sigma_x": float(np.max(np.abs(differences[:, 0]))),
        "max_abs_sigma_y": float(np.max(np.abs(differences[:, 1]))),
        "max_abs_sigma_z": float(np.max(np.abs(differences[:, 2]))),
        "max_bloch_difference": float(np.max(norm_difference)),
        "max_abs_purity_difference": float(np.max(np.abs(purity_difference))),
    }
    return figure_path, metrics


def main() -> None:
    run_start = perf_counter()
    arguments = parse_arguments()
    compare = arguments.mode != "numeric"
    config = set_active_config(arguments.config, comparison=compare)
    import numerical as tcl2

    density = config.SPECTRAL_DENSITY
    occupation_mode = "frozen" if arguments.mode == "compare-frozen" else "dynamic"
    path = output_directory(arguments.mode)
    print(f"[run] Mode: {arguments.mode}; results: {path}")
    if compare:
        print("[compare] Both methods use stationary Drude-Lorentz J and the same spin coupling.")

    tcl_start = perf_counter()
    t, sigma, purity, occupations, diagnostics = tcl2.solve(
        spectral_density=density, occupation_mode=occupation_mode,
        return_diagnostics=True,
    )
    timing = {"tcl2_solver": diagnostics.solver_seconds,
              "tcl2_total": perf_counter() - tcl_start}
    metadata = _base_metadata(arguments.mode, occupation_mode)
    effective = asdict(config)
    effective.pop("SOURCE_PATH")
    effective["SPECTRAL_PARAMETERS"] = config.spectral_parameters()
    metadata.update({
        "parameters": effective,
        "diagnostics": asdict(diagnostics),
        "perturbative_lambda": config.LAMBDA,
        "spectral_density": tcl2.spectral_description(),
        "quadrature": tcl2.quadrature_diagnostics(),
        "solver_defaults": {
            "max_iterations": tcl2.MAX_ITERATIONS,
            "absolute_tolerance": tcl2.ABSOLUTE_TOLERANCE,
            "relative_tolerance": tcl2.RELATIVE_TOLERANCE,
            "physicality_tolerance": tcl2.PHYSICALITY_TOLERANCE,
            "adaptive_strength": tcl2.ADAPTIVE_STRENGTH,
            "max_phase_increment": tcl2.MAX_PHASE_INCREMENT,
        },
    })

    heom_result = None
    if compare:
        from heom import solve_heom
        heom_result = solve_heom()
        metadata["heom"] = {
            key: value for key, value in vars(heom_result).items()
            if key not in {"t", "sigma", "purity"}
        }
        metadata["comparison_note"] = (
            "The TCL2 integral uses the configured frequency interval; HEOM uses the full Drude bath. "
            "Both start from a thermal bath. Dynamic N is evolved only in compare, not compare-frozen. "
            "Fixed HEOM defaults do not constitute a convergence check for arbitrary parameters."
        )
        timing.update(heom_setup=heom_result.setup_seconds,
                      heom_solver=heom_result.solver_seconds,
                      heom_total=heom_result.total_seconds)

    plot_start = perf_counter()
    if compare:
        figure_path, metrics = _comparison_figure(
            path, (t, sigma, purity, occupations), heom_result, occupation_mode, density,
        )
        metadata["comparison_metrics"] = metrics
        np.savez_compressed(
            path / "compare_solution.npz", t=t, omega=tcl2.omega_arr,
            sigma_tcl=sigma[:, 0], sigma_heom=heom_result.sigma,
            purity_tcl=purity[:, 0], purity_heom=heom_result.purity,
            occupations=occupations,
        )
    else:
        t_unitary, sigma_unitary = tcl2.solve_unitary()
        figure_path = tcl2.plot_results(
            t, sigma, purity, occupations, t_unitary, sigma_unitary, str(path),
        )
        np.savez_compressed(path / "solution.npz", t=t, omega=tcl2.omega_arr,
                            sigma=sigma, purity=purity, occupations=occupations)
    _save_bath_data(path, tcl2, density, occupations)
    _save_config(path)
    timing["plotting"] = perf_counter() - plot_start
    timing["total"] = perf_counter() - run_start
    metadata["timing_seconds"] = timing
    _write_json(path / "metadata.json", metadata)
    _write_summary(path, metadata)
    print(f"[{arguments.mode}] Plot: {figure_path}")


if __name__ == "__main__":
    main()
