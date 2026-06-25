#!/usr/bin/env python3
"""
AGC and AKC across discretization level k (equal-probability bins on a synthetic DGP).

Equal-probability bins = consecutive equal-size blocks in the rank-sorted outcome vector.
Uses ``acor(..., method="agc")`` and ``acor(..., method="akc")`` explicitly.
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from acor import acor

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PDF = os.path.join(_HERE, "agc_akc_discretization_k.pdf")

_FEAT_RHO = {"X": 0.9, "X'": 0.6, "X''": 0.3}
_FEAT_ORDER = ["X", "X'", "X''"]
_LINE_STYLES = {"AGC": "-", "AKC": "--"}

_FONT_BASE = 16
_FONT_TICK = 14
_FONT_LEGEND = 14


def _rho_S_bvn(rho: float) -> float:
    return (6.0 / np.pi) * np.arcsin(rho / 2.0)


def _tau_bvn(rho: float) -> float:
    return (2.0 / np.pi) * np.arcsin(rho)


def _palette() -> dict[str, str]:
    cb = sns.color_palette("colorblind", 6)
    return {"X": cb[0], "X'": cb[1], "X''": cb[2]}


def compute_scores(*, n_exp: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2**n_exp
    ks = list(range(1, 21))

    zy = rng.standard_normal(n)

    def make_feature(rho: float) -> np.ndarray:
        return np.ascontiguousarray(
            rho * zy + np.sqrt(1 - rho**2) * rng.standard_normal(n),
            dtype=np.float64,
        )

    feats = {
        name: make_feature(rho) for name, rho in _FEAT_RHO.items()
    }

    order = np.argsort(zy, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)

    def bins_for_k(k: int) -> np.ndarray:
        k_bins = 2**k
        if k_bins >= n:
            return np.ascontiguousarray(zy, dtype=np.float64)
        return np.ascontiguousarray(ranks * k_bins // n, dtype=np.float64)

    ys = {k: bins_for_k(k) for k in ks}

    rows: list[dict] = []
    total = len(_FEAT_RHO) * len(ks)
    done = 0
    t0 = time.time()

    for fname, rho in _FEAT_RHO.items():
        x = feats[fname]
        for k in ks:
            y = ys[k]
            agc = float(acor(x, y, method="agc").statistic)
            akc = float(acor(x, y, method="akc").statistic)
            rows.append(
                {"feature": fname, "rho": rho, "k": k, "metric": "AGC", "value": agc}
            )
            rows.append(
                {"feature": fname, "rho": rho, "k": k, "metric": "AKC", "value": akc}
            )
            done += 1
            print(
                f"\r[{done:>3}/{total}] {fname:<3} k={k:>2}  "
                f"elapsed {time.time() - t0:6.1f}s",
                end="",
                flush=True,
            )
    print()
    return pd.DataFrame(rows, columns=["feature", "rho", "k", "metric", "value"])


def plot_scores(df: pd.DataFrame, output_pdf: str) -> None:
    feat_rho = df.drop_duplicates("feature").set_index("feature")["rho"].to_dict()
    pal = _palette()

    plt.rcParams.update(
        {
            "font.size": _FONT_BASE,
            "axes.labelsize": _FONT_BASE,
            "xtick.labelsize": _FONT_TICK,
            "ytick.labelsize": _FONT_TICK,
            "legend.fontsize": _FONT_LEGEND,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for f in _FEAT_ORDER:
        rho = feat_rho[f]
        for ref in (_rho_S_bvn(rho), _tau_bvn(rho)):
            ax.axhline(
                ref,
                color=pal[f],
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                zorder=1,
            )

    for f in _FEAT_ORDER:
        sub = df[df["feature"] == f]
        for metric in ("AGC", "AKC"):
            s = sub[sub["metric"] == metric].sort_values("k")
            ax.plot(
                s["k"],
                s["value"],
                color=pal[f],
                linestyle=_LINE_STYLES[metric],
                linewidth=2,
                zorder=2,
            )

    ax.set_xlabel("Discretization level k")
    ax.set_ylabel("Metric value")
    ax.set_xticks([1, 5, 10, 15, 20])
    span = 20 - 1
    ax.set_xlim(1 - 0.02 * span, 20 + 0.06 * span)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    ax.minorticks_off()

    metric_handles = [
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="AGC"),
        Line2D([0], [0], color="black", linestyle="--", lw=2, label="AKC"),
    ]
    color_handles = [
        Line2D([0], [0], color=pal["X"], linestyle="-", lw=2, label="r = 0.9"),
        Line2D([0], [0], color=pal["X'"], linestyle="-", lw=2, label="r = 0.6"),
        Line2D([0], [0], color=pal["X''"], linestyle="-", lw=2, label="r = 0.3"),
    ]
    leg1 = ax.legend(
        handles=metric_handles,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.52),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.48),
        frameon=False,
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AGC/AKC vs discretization level k (synthetic bivariate normal features)."
    )
    ap.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_PDF,
        help=f"Output PDF (default: {_DEFAULT_PDF!r})",
    )
    ap.add_argument(
        "--n_exp",
        type=int,
        default=20,
        help="Sample size n = 2**n_exp (default: 20 → n=1_048_576)",
    )
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()

    print(f"Computing scores (n=2**{args.n_exp}, seed={args.seed})...")
    df = compute_scores(n_exp=args.n_exp, seed=args.seed)
    plot_scores(df, args.output)


if __name__ == "__main__":
    main()
