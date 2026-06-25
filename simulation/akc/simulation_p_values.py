#!/usr/bin/env python3
"""
CLI: AKC ``acor_test`` contrast (``pairwise_results``) p-values vs. Zou simple / unbiased / consistent z-tests.

Mirrors ``akc_pvals_sim.R``; outputs ``.npy`` files under ``--output_dir`` (see
:mod:`plot_p_values` in this package for PDFs). Naming parallels :mod:`simulation.agc.simulation_p_values`.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .helpers import run_simulation_akc_zou


def _default_output_dir() -> str:
    _here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(_here, "results"))


def _akc_prefix(output_dir: str, discrete_str: str, alt_str: str, n: int) -> str:
    return os.path.join(output_dir, f"akc_zou_{discrete_str}_{alt_str}_n{n}")


def run_single_job(
    n: int,
    T: int,
    discrete: bool,
    alternative: str,
    output_dir: str,
    *,
    variance: str = "plugin",
    iid: bool = True,
    fisher: bool = False,
) -> None:
    print(
        f"Starting AKC+Zou simulation: n={n}, T={T}, discrete={discrete}, "
        f"alternative={alternative}, variance={variance!r}, iid={iid}"
    )
    res = run_simulation_akc_zou(
        n,
        T=T,
        discrete=discrete,
        alternative=alternative,
        variance=variance,
        iid=iid,
        fisher=fisher,
    )
    os.makedirs(output_dir, exist_ok=True)
    discrete_str = "discrete" if discrete else "continuous"
    alt_str = alternative.replace(".", "_")
    base = _akc_prefix(output_dir, discrete_str, alt_str, n)
    for key, suffix in (
        ("ps_our", "our"),
        ("ps_zou_simple", "zou_simple"),
        ("ps_zou_unbiased", "zou_unbiased"),
        ("ps_zou_consist", "zou_consistent"),
    ):
        path = f"{base}_{suffix}.npy"
        np.save(path, res[key])
        print(f"Saved: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="AKC + Zou p-value Monte Carlo (one n per run)")
    p.add_argument("--n", type=int, required=True, help="Sample size")
    p.add_argument("--T", type=int, default=100_000, help="Replicates")
    p.add_argument("--discrete", action="store_true", help="Rounded DGP")
    p.add_argument(
        "--alternative",
        type=str,
        default="two.sided",
        choices=["two.sided", "one.sided"],
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: akc/results/",
    )
    p.add_argument(
        "--variance",
        type=str,
        default="plugin",
        choices=["ij", "plugin"],
        help="acor_test variance (default: plugin; acor package default is ij if omitted)",
    )
    p.add_argument(
        "--hac",
        action="store_true",
        help="Time-series / HAC (iid=False); default is iid=True",
    )
    p.add_argument("--fisher", action="store_true", help="fisher=True for acor_test")
    args = p.parse_args()
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
