#!/usr/bin/env python3
"""4×K histograms: AKC vs three Zou estimators. See ``../README.md``."""
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Row order in 4×K figures: (a)–(d) top to bottom.
_AKC_ROW_KEYS = ("our", "zou_unbiased", "zou_consist", "zou_simple")
_AKC_ROW_LABELS = (
    "Our",
    "Zou unbiased",
    "Zou consistent",
    "Zou simple",
)
_AKC_PANEL_LETTERS = ("a", "b", "c", "d")


def _prefix(results_dir: str, discrete_str: str, alt_str: str, n: int) -> str:
    return os.path.join(results_dir, f"akc_zou_{discrete_str}_{alt_str}_n{n}")


def _has_series(p: Optional[np.ndarray]) -> bool:
    return p is not None and getattr(p, "size", 0) > 0


def load_akc_zou(
    n_values: list[int],
    discrete: bool,
    alternative: str,
    results_dir: str,
) -> dict[str, list[Optional[np.ndarray]]]:
    """
    For each ``n``, loads each of the four ``.npy`` files independently. Missing
    file → ``None`` for that subplot (gray ``No data`` panel), same layout for
    all ``n`` in ``n_values``.
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
        for k, path in paths.items():
            if not os.path.exists(path):
                out[k].append(None)
                print(f"Missing n={n}  {k!r}  {path!r} (empty panel).")
            else:
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
    method_keys = _AKC_ROW_KEYS
    labels = _AKC_ROW_LABELS

    fig, axes = plt.subplots(
        num_rows, num_n, figsize=(20, 4.4 * num_rows), squeeze=False, sharey="row"
    )
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
                letter = _AKC_PANEL_LETTERS[r]
                label_y = 1.1 if r == 0 else 1.1
                ax.text(
                    -0.03,
                    label_y,
                    f"{letter}) {lab}",
                    transform=ax.transAxes,
                    fontsize=fs,
                    verticalalignment="bottom",
                )
    for r in range(num_rows):
        axes[r, 0].set_ylim(0, 1.2)
        axes[r, 0].set_yticks([0, 0.5, 1])

    os.makedirs(output_dir, exist_ok=True)
    dstr = "discrete" if discrete else "continuous"
    fn = f"pvalue_histograms_akc_zou_{dstr}_{alternative.replace('.', '_')}.pdf"
    path = os.path.join(output_dir, fn)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")
    return path


def main() -> None:
    _here = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.normpath(os.path.join(_here, "results"))
    default_plots = os.path.normpath(os.path.join(_here, "plots"))

    ap = argparse.ArgumentParser(
        description="AKC+Zou p-value histograms from saved .npy (see simulation_p_values.py)."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default=default_results,
        help="Directory of akc_zou_*.npy (default: akc/results/)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=default_plots,
        help="Where to write PDFs (default: akc/plots/)",
    )
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[50, 100, 500, 1000, 5000],
        help="Sample sizes (columns); missing .npy → gray 'No data' in that cell",
    )
    args = ap.parse_args()
    res_dir = os.path.normpath(args.results_dir)
    out_dir = os.path.normpath(args.output_dir)
    n_list = list(args.n)

    if not os.path.isdir(res_dir):
        raise FileNotFoundError(f"Results directory not found: {res_dir}")
    n_files = len(glob.glob(os.path.join(res_dir, "*.npy")))
    print(f"Found {n_files} result files in {res_dir}")
    print(f"Plots will be written under: {out_dir}/")

    method_keys = _AKC_ROW_KEYS

    # File stems: akc_zou_{discrete|continuous}_{two_sided|one_sided}_n{n}_*.npy
    configs: list[tuple[bool, str, str]] = [
        (True, "two.sided", "discrete + two.sided"),
        (True, "one.sided", "discrete + one.sided"),
        (False, "two.sided", "continuous + two.sided"),
        (False, "one.sided", "continuous + one.sided"),
    ]
    for discrete, alt, _label in configs:
        print(f"\n--- {_label} ---")
        p = load_akc_zou(n_list, discrete, alt, res_dir)
        any_cell = any(
            _has_series(p[k][c]) for k in method_keys for c in range(len(n_list))
        )
        if not any_cell:
            print(
                f"Skip plot ({_label}): no .npy data for any method/n in {n_list!r}."
            )
            continue
        n_cols_with_data = sum(
            1
            for c in range(len(n_list))
            if any(_has_series(p[k][c]) for k in method_keys)
        )
        if n_cols_with_data < len(n_list):
            print(
                f"({_label}): {n_cols_with_data} of {len(n_list)} columns have at least one series; missing cells show empty panels."
            )
        dd = {k: p[k] for k in p}
        create_histograms_akc_zou(
            n_list, dd, discrete=discrete, alternative=alt, output_dir=out_dir
        )

    print(f"Done. Plots under: {out_dir}/")


if __name__ == "__main__":
    main()
