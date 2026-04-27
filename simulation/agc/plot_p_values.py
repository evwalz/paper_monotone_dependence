#!/usr/bin/env python3
"""Histograms: Meng (Spearman) vs global AGC (``acor_test``, CMA/AGC line). See ``../README.md``."""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def _agc_npy_path(results_dir: str, discrete_str: str, alt_str: str, n: int) -> str:
    """Path to ``pvals_agc_{discrete|continuous}_{alt}_n{n}.npy``."""
    return os.path.join(results_dir, f"pvals_agc_{discrete_str}_{alt_str}_n{n}.npy")


def load_results(
    n_values: list,
    discrete: bool,
    alternative: str,
    results_dir: str = "./results",
):
    """
    Load saved p-value arrays. Returns (Meng, AGC) as num_n x T (Meng empty for discrete).
    """
    discrete_str = "discrete" if discrete else "continuous"
    alt_str = alternative.replace(".", "_")
    num_n = len(n_values)
    list_meng: list[np.ndarray] = []
    list_agc: list[np.ndarray] = []

    for n in n_values:
        f_agc = _agc_npy_path(results_dir, discrete_str, alt_str, n)
        if not os.path.exists(f_agc):
            raise FileNotFoundError(f"Missing AGC p-value file: {f_agc!r}")
        p_agc = np.load(f_agc)
        list_agc.append(p_agc)
        print(f"Loaded AGC: n={n}, shape={p_agc.shape} from {f_agc!r}")

        if not discrete:
            f_meng = os.path.join(
                results_dir, f"pvals_meng_{discrete_str}_{alt_str}_n{n}.npy"
            )
            if not os.path.exists(f_meng):
                raise FileNotFoundError(f"Missing file: {f_meng}")
            p_meng = np.load(f_meng)
            list_meng.append(p_meng)
            print(f"Loaded Meng: n={n}, shape={p_meng.shape}")
        else:
            list_meng.append(np.array([]))

    if discrete:
        arr_meng = np.array([np.array([]) for _ in range(num_n)], dtype=object)
    else:
        arr_meng = np.array(list_meng)
    arr_agc = np.array(list_agc)
    return arr_meng, arr_agc


def create_pvalue_histograms(
    n_values: list,
    list_of_pvals_meng: np.ndarray,
    list_of_pvals_agc: np.ndarray,
    discrete: bool = True,
    alternative: str = "two.sided",
    output_dir: str = "./plots",
) -> None:
    num_n = len(n_values)
    num_rows = 1 if discrete else 2
    fig, axes = plt.subplots(num_rows, num_n, figsize=(20, 4 * num_rows))
    fs = 20
    plt.rcParams.update({"font.size": 16})

    if num_rows == 1 and num_n == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_n == 1:
        axes = axes.reshape(-1, 1)

    for i, n in enumerate(n_values):
        if not discrete:
            axes[0, i].hist(
                list_of_pvals_meng[i, :],
                bins=20,
                density=True,
                alpha=1,
                color="darkgrey",
                edgecolor="black",
                linewidth=1.2,
            )
            axes[0, i].axhline(y=1, color="black", linestyle="--", linewidth=1)
            axes[0, i].set_ylim(0, 1.2)
            axes[0, i].set_xticks([0, 0.5, 1])
            axes[0, i].set_yticks([0, 0.5, 1])
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_title(f"n = {n}", fontsize=fs)
            axes[0, i].tick_params(axis="both", which="major", labelsize=fs)

        row_idx = 1 if not discrete else 0
        axes[row_idx, i].hist(
            list_of_pvals_agc[i, :],
            bins=20,
            density=True,
            alpha=1,
            color="darkgrey",
            edgecolor="black",
            linewidth=1.2,
        )
        axes[row_idx, i].axhline(y=1, color="black", linestyle="--", linewidth=1)
        axes[row_idx, i].set_ylim(0, 1.2)
        axes[row_idx, i].set_xticks([0, 0.5, 1])
        axes[row_idx, i].set_yticks([0, 0.5, 1])
        axes[row_idx, i].grid(True, alpha=0.3)
        axes[row_idx, i].set_title(f"n = {n}", fontsize=fs)
        axes[row_idx, i].tick_params(axis="both", which="major", labelsize=fs)

    if not discrete:
        axes[0, 0].text(
            -0.03,
            1.05,
            "Meng",
            transform=axes[0, 0].transAxes,
            fontsize=fs,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        axes[1, 0].text(
            -0.03,
            1.05,
            "AGC",
            transform=axes[1, 0].transAxes,
            fontsize=fs,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
    else:
        axes[0, 0].text(
            -0.03,
            1.05,
            "AGC",
            transform=axes[0, 0].transAxes,
            fontsize=fs,
            verticalalignment="bottom",
            horizontalalignment="left",
        )

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
    n_values = [50, 100, 500, 1000, 5000]
    _here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(_here, "results"))
    plots_dir = results_dir
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    n_files = len(glob.glob(os.path.join(results_dir, "*.npy")))
    print(f"Found {n_files} result files in {results_dir}")

    print("\n--- discrete two-sided (AGC only) ---")
    try:
        m, a = load_results(
            n_values, discrete=True, alternative="two.sided", results_dir=results_dir
        )
        create_pvalue_histograms(
            n_values, m, a, discrete=True, alternative="two.sided", output_dir=plots_dir
        )
    except FileNotFoundError as e:
        print(f"Skip: {e}")

    print("\n--- discrete one-sided (AGC only) ---")
    try:
        m, a = load_results(
            n_values, discrete=True, alternative="one.sided", results_dir=results_dir
        )
        create_pvalue_histograms(
            n_values, m, a, discrete=True, alternative="one.sided", output_dir=plots_dir
        )
    except FileNotFoundError as e:
        print(f"Skip: {e}")

    print("\n--- continuous two-sided (Meng + AGC) ---")
    try:
        m, a = load_results(
            n_values, discrete=False, alternative="two.sided", results_dir=results_dir
        )
        create_pvalue_histograms(
            n_values, m, a, discrete=False, alternative="two.sided", output_dir=plots_dir
        )
    except FileNotFoundError as e:
        print(f"Skip: {e}")

    print("\n--- continuous one-sided (Meng + AGC) ---")
    try:
        m, a = load_results(
            n_values, discrete=False, alternative="one.sided", results_dir=results_dir
        )
        create_pvalue_histograms(
            n_values, m, a, discrete=False, alternative="one.sided", output_dir=plots_dir
        )
    except FileNotFoundError as e:
        print(f"Skip: {e}")

    print(f"Done. Plots under: {plots_dir}/")


if __name__ == "__main__":
    main()
