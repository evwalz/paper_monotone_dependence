#!/usr/bin/env python3
"""Histograms: Meng (Spearman) vs AGC ``acor_test`` contrast (``pairwise_results``). See ``../README.md``."""
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _agc_npy_path(results_dir: str, discrete_str: str, alt_str: str, n: int) -> str:
    """Path to ``pvals_agc_{discrete|continuous}_{alt}_n{n}.npy``."""
    return os.path.join(results_dir, f"pvals_agc_{discrete_str}_{alt_str}_n{n}.npy")


def load_results(
    n_values: list[int],
    discrete: bool,
    alternative: str,
    results_dir: str = "./results",
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    """
    Load saved p-value arrays per ``n``. Missing ``.npy`` → ``None`` for that column
    (histogram uses an empty panel). Returns ``(meng_per_n, agc_per_n)``; for discrete,
    ``meng_per_n`` is all ``None``.
    """
    discrete_str = "discrete" if discrete else "continuous"
    alt_str = alternative.replace(".", "_")
    list_meng: list[Optional[np.ndarray]] = []
    list_agc: list[Optional[np.ndarray]] = []

    for n in n_values:
        f_agc = _agc_npy_path(results_dir, discrete_str, alt_str, n)
        if not os.path.exists(f_agc):
            list_agc.append(None)
            print(f"Missing AGC n={n}: {f_agc!r} (empty panel).")
        else:
            p_agc = np.load(f_agc)
            list_agc.append(p_agc)
            print(f"Loaded AGC: n={n}, shape={p_agc.shape} from {f_agc!r}")

        if not discrete:
            f_meng = os.path.join(
                results_dir, f"pvals_meng_{discrete_str}_{alt_str}_n{n}.npy"
            )
            if not os.path.exists(f_meng):
                list_meng.append(None)
                print(f"Missing Meng n={n}: {f_meng!r} (empty panel).")
            else:
                p_meng = np.load(f_meng)
                list_meng.append(p_meng)
                print(f"Loaded Meng: n={n}, shape={p_meng.shape}")
        else:
            list_meng.append(None)

    return list_meng, list_agc


def _has_series(p: Optional[np.ndarray]) -> bool:
    return p is not None and getattr(p, "size", 0) > 0


def _empty_panel(ax: Axes) -> None:
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


def _style_pval_axis(ax: Axes, n: int, fs: int, *, show_title: bool = True) -> None:
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(True, alpha=0.3)
    if show_title:
        ax.set_title(f"n = {n}", fontsize=fs)
    ax.tick_params(axis="both", which="major", labelsize=fs)


def create_pvalue_histograms(
    n_values: list[int],
    list_of_pvals_meng: list[Optional[np.ndarray]],
    list_of_pvals_agc: list[Optional[np.ndarray]],
    discrete: bool = True,
    alternative: str = "two.sided",
    output_dir: str = "./plots",
) -> None:
    num_n = len(n_values)
    num_rows = 1 if discrete else 2
    fig, axes = plt.subplots(
        num_rows, num_n, figsize=(20, 4 * num_rows), squeeze=False, sharey="row"
    )
    fs = 20
    plt.rcParams.update({"font.size": 16})
    axes = np.atleast_2d(axes)

    for i, n in enumerate(n_values):
        if discrete:
            ax_a = axes[0, i]
            pa = list_of_pvals_agc[i]
            if _has_series(pa):
                ax_a.hist(
                    pa,
                    bins=20,
                    density=True,
                    alpha=1,
                    color="darkgrey",
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax_a.axhline(y=1, color="black", linestyle="--", linewidth=1)
            else:
                _empty_panel(ax_a)
            _style_pval_axis(ax_a, n, fs)
        else:
            # Continuous: (a) Our, (b) Meng
            ax_a = axes[0, i]
            pa = list_of_pvals_agc[i]
            if _has_series(pa):
                ax_a.hist(
                    pa,
                    bins=20,
                    density=True,
                    alpha=1,
                    color="darkgrey",
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax_a.axhline(y=1, color="black", linestyle="--", linewidth=1)
            else:
                _empty_panel(ax_a)
            _style_pval_axis(ax_a, n, fs)

            ax_m = axes[1, i]
            pm = list_of_pvals_meng[i]
            if _has_series(pm):
                ax_m.hist(
                    pm,
                    bins=20,
                    density=True,
                    alpha=1,
                    color="darkgrey",
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax_m.axhline(y=1, color="black", linestyle="--", linewidth=1)
            else:
                _empty_panel(ax_m)
            _style_pval_axis(ax_m, n, fs, show_title=False)

    if not discrete:
        axes[0, 0].text(
            -0.03,
            1.1,
            "a) Our",
            transform=axes[0, 0].transAxes,
            fontsize=fs,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        axes[1, 0].text(
            -0.03,
            1.05,
            "b) Meng",
            transform=axes[1, 0].transAxes,
            fontsize=fs,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
    else:
        axes[0, 0].text(
            -0.03,
            1.1,
            "a) Our",
            transform=axes[0, 0].transAxes,
            fontsize=fs,
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    for r in range(num_rows):
        axes[r, 0].set_ylim(0, 1.2)
        axes[r, 0].set_yticks([0, 0.5, 1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    os.makedirs(output_dir, exist_ok=True)
    discrete_str = "discrete" if discrete else "continuous"
    fname = f"pvalue_histograms_{discrete_str.lower()}_{alternative.replace('.', '_')}.pdf"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close()


def main() -> None:
    _here = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.normpath(os.path.join(_here, "results"))
    default_plots = os.path.normpath(os.path.join(_here, "plots"))

    ap = argparse.ArgumentParser(
        description="AGC/Meng p-value histograms from saved .npy (see simulation_p_values.py)."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default=default_results,
        help="Directory with pvals_agc_*.npy / pvals_meng_*.npy (default: agc/results/)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=default_plots,
        help="Where to write PDFs (default: agc/plots/)",
    )
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[50, 100, 500, 1000, 5000],
        help="Sample sizes (columns); missing .npy → gray 'No data' panel for that column",
    )
    args = ap.parse_args()

    results_dir = os.path.normpath(args.results_dir)
    plots_dir = os.path.normpath(args.output_dir)
    n_values = list(args.n)

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    n_files = len(glob.glob(os.path.join(results_dir, "*.npy")))
    print(f"Found {n_files} result files in {results_dir}")
    print(f"Plots will be written under: {plots_dir}/")

    configs: list[tuple[bool, str, str]] = [
        (True, "two.sided", "discrete + two.sided"),
        (True, "one.sided", "discrete + one.sided"),
        (False, "two.sided", "continuous + two.sided"),
        (False, "one.sided", "continuous + one.sided"),
    ]

    for discrete, alt, label in configs:
        print(f"\n--- {label} ---")
        m, a = load_results(n_values, discrete, alt, results_dir=results_dir)
        if discrete:
            n_ok = sum(1 for x in a if _has_series(x))
            if n_ok == 0:
                print(
                    f"Skip plot ({label}): no AGC .npy for any n in {n_values!r}."
                )
                continue
            if n_ok < len(n_values):
                print(
                    f"({label}): {n_ok} of {len(n_values)} columns have data; others show empty panels."
                )
        else:
            n_ok = sum(
                1 for mi, ai in zip(m, a) if _has_series(mi) or _has_series(ai)
            )
            if n_ok == 0:
                print(
                    f"Skip plot ({label}): no Meng or AGC .npy for any n in {n_values!r}."
                )
                continue
            if n_ok < len(n_values):
                print(
                    f"({label}): {n_ok} of {len(n_values)} columns have at least one series; missing cells show empty panels."
                )
        create_pvalue_histograms(
            n_values, m, a, discrete=discrete, alternative=alt, output_dir=plots_dir
        )

    print(f"Done. Plots under: {plots_dir}/")


if __name__ == "__main__":
    main()
