#!/usr/bin/env python3
"""
Monte Carlo p-value calibration: Meng (1992) on **Spearman** corrs vs. global **AGC**
test from ``acor.acor_test`` (``method="agc"``; same line as **CMA** in ``acor``, not AKC).

Saves one ``.npy`` per run under ``--output_dir``; see :mod:`plot_p_values` for histograms.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .helpers import run_simulation_agc, run_simulation_meng_agc


def _default_output_dir() -> str:
    _here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(_here, "results"))


def run_single_job(
    n: int,
    T: int,
    discrete: bool,
    alternative: str,
    output_dir: str,
    *,
    variance: str = "delta",
    iid: bool = True,
    fisher: bool = False,
) -> None:
    eff_variance = variance
    print(
        f"Starting simulation: n={n}, T={T}, discrete={discrete}, "
        f"alternative={alternative}, acor (AGC) variance={eff_variance!r}, iid={iid}"
    )

    if discrete:
        pvals_agc = run_simulation_agc(
            n,
            T=T,
            discrete=True,
            alternative=alternative,
            variance=eff_variance,
            iid=iid,
            fisher=fisher,
        )
        pvals_meng = np.array([])
    else:
        pvals_meng, pvals_agc = run_simulation_meng_agc(
            n,
            T=T,
            alternative=alternative,
            variance=eff_variance,
            iid=iid,
            fisher=fisher,
        )

    os.makedirs(output_dir, exist_ok=True)
    discrete_str = "discrete" if discrete else "continuous"
    alt_str = alternative.replace(".", "_")

    f_meng = os.path.join(output_dir, f"pvals_meng_{discrete_str}_{alt_str}_n{n}.npy")
    f_agc = os.path.join(output_dir, f"pvals_agc_{discrete_str}_{alt_str}_n{n}.npy")

    if not discrete:
        np.save(f_meng, pvals_meng)
    np.save(f_agc, pvals_agc)
    print(f"Saved: {f_agc}" + (f" and {f_meng}" if not discrete else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P-value simulation: Meng (Spearman) vs global AGC (acor_test) for a single n"
    )
    parser.add_argument("--n", type=int, required=True, help="Sample size")
    parser.add_argument("--T", type=int, default=100_000, help="Number of Monte Carlo replicates")
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="Round DGP (AGC p-values only; no Meng .npy)",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="two.sided",
        choices=["two.sided", "one.sided"],
        help="Meng only. AGC p-values (acor) always use two.sided, matching R acor.test defaults.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for pvals_*.npy (default: agc/results/)",
    )
    parser.add_argument(
        "--variance",
        type=str,
        default="delta",
        choices=["delta", "ij"],
        help="acor_test variance (default: delta, R acor default).",
    )
    parser.add_argument(
        "--hac",
        action="store_true",
        help="Use HAC / time-series mode (iid=False). Default matches R acor IID=TRUE (iid=True).",
    )
    parser.add_argument(
        "--fisher",
        action="store_true",
        help="Pass fisher=True to acor_test",
    )
    args = parser.parse_args()
    out = args.output_dir or _default_output_dir()

    np.random.seed(42)
    run_single_job(
        args.n,
        args.T,
        args.discrete,
        args.alternative,
        out,
        variance=args.variance,
        iid=not args.hac,
        fisher=args.fisher,
    )


if __name__ == "__main__":
    main()
