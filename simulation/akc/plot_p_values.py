#!/usr/bin/env python3
"""4×K histograms: AKC pairwise (acor) vs three Zou estimators. See ``../README.md``."""
from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _prefix(results_dir: str, discrete_str: str, alt_str: str, n: int) -> str:
    return os.path.join(results_dir, f"akc_zou_{discrete_str}_{alt_str}_n{n}")


def load_akc_zou(
    n_values: list[int],
    discrete: bool,
    alternative: str,
    results_dir: str,
) -> dict[str, list[Optional[np.ndarray]]]:
    """
    For each ``n``, loads all four ``.npy`` files if *all* are present; otherwise
    appends ``None`` for that column (plot shows an empty panel).
    """
    discrete_str = "discrete" if discrete else "continuous"
    alt_str = alternative.replace(".", "_")
    out: dict[str, list[Optional[np.ndarray]]] = {
        "our": [],
        "zou_simple": [],
        "zou_unbiased": [],
        "zou_consist": [],
    }
    for n in n_values:
        base = _prefix(results_dir, discrete_str, alt_str, n)
        paths = {
            "our": f"{base}_our.npy",
            "zou_simple": f"{base}_zou_simple.npy",
            "zou_unbiased": f"{base}_zou_unbiased.npy",
            "zou_consist": f"{base}_zou_consistent.npy",
        }
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            for k in out:
                out[k].append(None)
            print(f"Skipping n={n} (incomplete or missing: {len(missing)} of 4 .npy files; empty panels).")
            continue
        for k, path in paths.items():
            arr = np.load(path)
            out[k].append(arr)
            print(f"Loaded n={n}  T={arr.shape[0]}  {k!r}  {path!r}")
    return out


def create_histograms_akc_zou(
    n_values: list[int],
    pvals: dict[str, list[Optional[np.ndarray]]],
    *,
    discrete: bool,
    alternative: str,
    output_dir: str,
) -> str:
    num_n = len(n_values)
    num_rows = 4
    method_keys = ("our", "zou_simple", "zou_unbiased", "zou_consist")
    labels = ("AKC", "Zou simple", "Zou unbiased", "Zou consistent")

    fig, axes = plt.subplots(num_rows, num_n, figsize=(20, 4 * num_rows), squeeze=False)
    fs = 20
    plt.rcParams.update({"font.size": 16})
    axes = np.atleast_2d(axes)

    bins = np.linspace(0, 1, 20)
    for r, (key, lab) in enumerate(zip(method_keys, labels)):
        for c, n in enumerate(n_values):
            ax = axes[r, c]
            series = pvals[key][c]
            if series is None or series.size == 0:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                    color="gray",
                )
            else:
                ax.hist(
                    series,
                    bins=bins,
                    density=True,
                    color="darkgrey",
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax.axhline(1, color="black", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1.2)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=fs)
            ax.tick_params(axis="both", which="major", labelsize=fs)
            if c == 0:
                ax.text(
                    -0.03,
                    1.05,
                    lab,
                    transform=ax.transAxes,
                    fontsize=fs,
                    verticalalignment="bottom",
                )
    os.makedirs(output_dir, exist_ok=True)
    dstr = "discrete" if discrete else "continuous"
    fn = f"pvalue_histograms_akc_zou_{dstr}_{alternative.replace('.', '_')}.pdf"
    path = os.path.join(output_dir, fn)
    plt.tight_layout()
    plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")
    return path


def _default_results_dir() -> str:
    _here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(_here, "results"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AKC+Zou p-value histograms. Missing n: gray 'No data' panel (all four .npy per n required to plot that column)."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory of akc_zou_*.npy (default: akc/results/)",
    )
    ap.add_argument("--output_dir", type=str, default=None, help="PDF dir (default: results_dir)")
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[50, 100, 500, 1000, 5000],
    )
    args = ap.parse_args()
    res_dir = args.results_dir or _default_results_dir()
    out_dir = args.output_dir or res_dir
    n_list = list(args.n)

    if not os.path.isdir(res_dir):
        raise FileNotFoundError(f"Results directory not found: {res_dir}")

    # File stems: akc_zou_{discrete|continuous}_{two_sided|one_sided}_n{n}_*.npy
    configs: list[tuple[bool, str, str]] = [
        (True, "two.sided", "discrete + two.sided"),
        (True, "one.sided", "discrete + one.sided"),
        (False, "two.sided", "continuous + two.sided"),
        (False, "one.sided", "continuous + one.sided"),
    ]
    for discrete, alt, _label in configs:
        p = load_akc_zou(n_list, discrete, alt, res_dir)
        n_ok = sum(1 for c in range(len(n_list)) if p["our"][c] is not None)
        if n_ok == 0:
            print(f"Skip plot ({_label}): no complete .npy groups for any n in {n_list!r}.")
            continue
        if n_ok < len(n_list):
            print(
                f"({_label}): plotting {n_ok} of {len(n_list)} n with data; other columns show empty panels."
            )
        dd = {k: p[k] for k in p}
        create_histograms_akc_zou(
            n_list, dd, discrete=discrete, alternative=alt, output_dir=out_dir
        )


if __name__ == "__main__":
    main()
