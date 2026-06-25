"""
Monte Carlo: **AKC** ``acor_test`` contrast p-value (first ``pairwise_results`` row,
``method="akc"``) vs. three **Zou** variance-based z-tests for the same concordance
difference.

Mirrors the role of :mod:`simulation.agc.helpers` for the AGC + Meng path.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from ..calibration_dgp import (
    _alternative_for_acor_test,
    acor_first_pairwise_entry,
    sample_calibration_dgp,
)
from .zou_concordance import compute_zou_variances_diff

try:
    from acor import acor_test
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "AKC simulation requires the acor package, e.g.\n"
        '  pip install "git+https://github.com/evwalz/acor-python.git"\n'
        "See the repository root README for details."
    ) from e


def pvalue_from_z(z: float, alternative: str) -> float:
    if alternative == "two.sided":
        return float(2.0 * (1.0 - norm.cdf(abs(z))))
    if alternative == "one.sided":
        return float(1.0 - norm.cdf(z))
    raise ValueError("alternative must be 'two.sided' or 'one.sided'")


def pvalue_acor_akc(
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "two.sided",
    variance: str = "ij",
    conf_level: float = 0.95,
    iid: bool = True,
    fisher: bool = False,
) -> float:
    """``acor_test`` ``method='akc'``: p-value from the first ``pairwise_results`` row."""
    y = np.asarray(y, dtype=np.float64).ravel()
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    n = y.size
    if n < 3 or x1.size != n or x2.size != n:
        raise ValueError("y, x1, x2 must be same length and n >= 3 for acor_test")
    x_mat = np.column_stack([x1, x2])
    alt_acor = _alternative_for_acor_test(alternative)
    res = acor_test(
        x_mat,
        y,
        method="akc",
        alternative=alt_acor,
        conf_level=conf_level,
        iid=iid,
        fisher=fisher,
        variance=variance,
    )
    return float(acor_first_pairwise_entry(res)["pvalue"])


def run_simulation_akc_zou(
    n: int,
    T: int = 10_000,
    *,
    discrete: bool = False,
    alternative: str = "two.sided",
    variance: str = "ij",
    conf_level: float = 0.95,
    iid: bool = True,
    fisher: bool = False,
) -> dict[str, np.ndarray]:
    """
    Returns ``ps_our`` = first ``pairwise_results`` ``pvalue`` from ``acor_test``
    (``method='akc'``), and three Zou p-value streams from the same row's
    ``difference`` and Zou variance estimates.
    """
    alt_acor = _alternative_for_acor_test(alternative)

    ps_our: list[float] = []
    ps_zou_simple: list[float] = []
    ps_zou_unbiased: list[float] = []
    ps_zou_consist: list[float] = []
    acor_kw = dict(
        method="akc",
        alternative=alt_acor,
        conf_level=conf_level,
        iid=iid,
        fisher=fisher,
        variance=variance,
    )

    for _ in tqdm(range(T), desc=f"n={n} akc+zou"):
        y0, x1, x2 = sample_calibration_dgp(n, discrete=discrete)
        x_mat = np.column_stack([x1, x2])
        res = acor_test(x_mat, y0, **acor_kw)
        p0 = acor_first_pairwise_entry(res)
        ps_our.append(float(p0["pvalue"]))
        akc_diff = float(p0["difference"])

        zv = compute_zou_variances_diff(x_mat, y0)
        se_s = float(np.sqrt(zv["var_zou_simple"] / n))
        se_u = float(np.sqrt(zv["var_zou_unbiased"] / n))
        se_c = float(np.sqrt(zv["var_zou_consistent"] / n))
        if se_s > 0:
            ps_zou_simple.append(pvalue_from_z(akc_diff / se_s, alternative))
        else:
            ps_zou_simple.append(1.0)
        if se_u > 0:
            ps_zou_unbiased.append(pvalue_from_z(akc_diff / se_u, alternative))
        else:
            ps_zou_unbiased.append(1.0)
        if se_c > 0:
            ps_zou_consist.append(pvalue_from_z(akc_diff / se_c, alternative))
        else:
            ps_zou_consist.append(1.0)

    return {
        "ps_our": np.array(ps_our),
        "ps_zou_simple": np.array(ps_zou_simple),
        "ps_zou_unbiased": np.array(ps_zou_unbiased),
        "ps_zou_consist": np.array(ps_zou_consist),
    }
