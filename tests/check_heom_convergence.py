from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import set_active_config

set_active_config(ROOT / "config.toml", comparison=True)

from heom import solve_heom

DEPTHS = (3, 4, 5, 6)
EXPANSIONS = (10, 12, 14, 16, 18)
OUTPUT_DIR = ROOT / "heom_check"


def save_plot(parameters, differences, xlabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for index, component in enumerate(("x", "y", "z")):
        ax.plot(parameters, differences[:, index], "o-", label=rf"$\sigma_{component}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\max_t |\Delta\langle\sigma_\alpha\rangle|$")
    ax.set_xticks(parameters)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Saving convergence plots to: {OUTPUT_DIR}", flush=True)
    results = {}
    total_runs = len(EXPANSIONS) + len(DEPTHS) - 1
    completed = 0

    for n_exp in EXPANSIONS:
        print(f"[{completed + 1}/{total_runs}] HEOM: depth=5, Pade terms={n_exp}", flush=True)
        results[(5, n_exp)] = solve_heom(depth=5, n_exp=n_exp)
        completed += 1

    pade_diff = []
    for n_exp in EXPANSIONS[:-1]:
        difference = results[(5, n_exp)].sigma - results[(5, 18)].sigma
        pade_diff.append(np.max(np.abs(difference), axis=0))
    save_plot(EXPANSIONS[:-1], np.array(pade_diff), "Number of Padé terms", "pade_convergence.png")
    print(f"Saved: {OUTPUT_DIR / 'pade_convergence.png'}", flush=True)

    for depth in DEPTHS:
        if (depth, 16) not in results:
            print(f"[{completed + 1}/{total_runs}] HEOM: depth={depth}, Pade terms=16", flush=True)
            results[(depth, 16)] = solve_heom(depth=depth, n_exp=16)
            completed += 1

    depth_diff = []
    for depth in DEPTHS[:-1]:
        difference = results[(depth, 16)].sigma - results[(6, 16)].sigma
        depth_diff.append(np.max(np.abs(difference), axis=0))

    save_plot(DEPTHS[:-1], np.array(depth_diff), "HEOM hierarchy depth", "depth_convergence.png")
    print(f"Saved: {OUTPUT_DIR / 'depth_convergence.png'}", flush=True)


if __name__ == "__main__":
    main()
