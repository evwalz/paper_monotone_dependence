"""
Data-generating process for **Meng (1992)** on **Spearman** correlations and the
**global** test from ``acor.acor_test`` with ``method="agc"``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from tqdm import tqdm

from ..calibration_dgp import (
    _one_sample_continuous,
    _one_sample_discrete,
    _alternative_for_acor_test,
)

try:
    from acor import acor_test
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Simulation requires the acor package, e.g.\n"
        '  pip install "git+https://github.com/evwalz/acor-python.git"\n'
        "See the repository root README for details."
    ) from e


def meng_test_corr(r1, r2, r12, n, alternative: str = "two.sided") -> float:
    """
    Meng, Rosenthal & Rubin (1992) test for two dependent (overlapping) Spearman/correlation
    comparisons (implemented as in the prior cocor-style setup).
    """
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    r_squared_avg = 0.5 * (r1**2 + r2**2)
    f = (1 - r12) / (2 * (1 - r_squared_avg))
    if not np.isnan(f) and f > 1:
        f = 1

    h = 1 + r_squared_avg / (1 - r_squared_avg) * (1 - f)
    z_stat = (z1 - z2) * np.sqrt((n - 3) / (2 * (1 - r12) * h))

    if alternative == "two.sided":
        return float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    if alternative == "one.sided":
        return float(1 - stats.norm.cdf(z_stat))
    raise ValueError(f"alternative must be 'two.sided' or 'one.sided', got {alternative!r}")


def meng_pvalue_spearman(
    y: np.ndarray, x1: np.ndarray, x2: np.ndarray, *, alternative: str = "two.sided"
) -> float:
    """Meng test on Spearman correlations among y, x1, x2."""
    r1, _ = spearmanr(y, x1)
    r2, _ = spearmanr(y, x2)
    r12, _ = spearmanr(x1, x2)
    n = len(y)
    return meng_test_corr(r1, r2, r12, n, alternative=alternative)


def pvalue_acor_agc_global(
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alternative: str = "two.sided",
    variance: str = "delta",
    conf_level: float = 0.95,
    iid: bool = True,
    fisher: bool = False,
) -> float:
    """
    Global ``acor_test`` p-value with ``method='agc'`` (two predictors vs. one outcome).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    n = y.size
    if n < 3 or x1.size != n or x2.size != n:
        raise ValueError("y, x1, x2 must be same length and n >= 3 for acor_test")
    X = np.column_stack([x1, x2])
    alt_acor = _alternative_for_acor_test(alternative)
    res = acor_test(
        X,
        y,
        method="agc",
        alternative=alt_acor,
        conf_level=conf_level,
        iid=iid,
        fisher=fisher,
        variance=variance,
    )
    return float(res.pvalue)


def run_simulation_meng_agc(
    n: int,
    T: int = 10_000,
    alternative: str = "two.sided",
    *,
    variance: str = "delta",
    conf_level: float = 0.95,
    iid: bool = True,
    fisher: bool = False,
):
    """
    Continuous DGP: each replicate = Meng p-value (Spearman) + global ``acor_test`` p-value
    (AGC, two predictors).
    """
    sigma_1 = sigma_2 = 1.0
    ps_meng: list[float] = []
    ps_agc: list[float] = []
    acor_kw = dict(
        alternative="two.sided",
        variance=variance,
        conf_level=conf_level,
        iid=iid,
        fisher=fisher,
    )

    for _ in tqdm(range(T), desc=f"n={n}"):
        y0, x1, x2 = _one_sample_continuous(n, sigma_1, sigma_2)
        p_agc = pvalue_acor_agc_global(y0, x1, x2, **acor_kw)
        p_meng = meng_pvalue_spearman(y0, x1, x2, alternative=alternative)
        ps_meng.append(p_meng)
        ps_agc.append(p_agc)

    return np.array(ps_meng), np.array(ps_agc)


def run_simulation_agc(
    n: int,
    T: int = 10_000,
    discrete: bool = False,
    alternative: str = "two.sided",
    *,
    variance: str | None = None,
    conf_level: float = 0.95,
    iid: bool = True,
    fisher: bool = False,
) -> np.ndarray:
    _ = alternative
    sigma_1 = sigma_2 = 1.0
    v = variance if variance is not None else "delta"
    acor_kw = dict(
        alternative="two.sided",
        variance=v,
        conf_level=conf_level,
        iid=iid,
        fisher=fisher,
    )
    out: list[float] = []
    for _ in tqdm(range(T), desc=f"n={n}"):
        if discrete:
            y0, x1, x2 = _one_sample_discrete(n, sigma_1, sigma_2)
        else:
            y0, x1, x2 = _one_sample_continuous(n, sigma_1, sigma_2)
        out.append(pvalue_acor_agc_global(y0, x1, x2, **acor_kw))
    return np.array(out)
