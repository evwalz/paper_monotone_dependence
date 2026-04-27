"""
Zou (2000)-style concordance variance estimators for two predictors vs. one standard
(outcome), and standard errors for the **concordance difference** ``c1 - c2``.

Pure-NumPy port of ``zou_concordance`` in ``akc_pvals_sim.R``. The default path is
**O(n log n)** (``zou_fenwick``, aligned with ``zou_concordance2_fast_cpp``); a
dense O(n^2) reference is ``zou_concordance_numpy_dense``. The R pipeline's
``compute_zou_variances_diff`` uses ``4 * sed^2 * n`` on the AKC-difference scale.
"""
from __future__ import annotations

import numpy as np


def _cov2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.cov(np.vstack((a, b)), ddof=1)[0, 1])


def _v_ij_cxx(sum_x: float, sum_x2: float, npairs: float) -> float:
    """C++ `var_from_sums`."""
    if npairs <= 1.0:
        return 0.0
    return (sum_x2 - sum_x**2 / npairs) / (npairs - 1.0)


def _cov_ij_cxx(sx: float, sy: float, sxy: float, npairs: float) -> float:
    """C++ `cov_from_sums`."""
    if npairs <= 1.0:
        return 0.0
    return (sxy - sx * sy / npairs) / (npairs - 1.0)


def zou_seds_from_row_stats(
    n: int,
    rsum_XX: np.ndarray,
    rsum_XY1: np.ndarray,
    rsum_XY2: np.ndarray,
    rsum_Xd: np.ndarray,
    rsum_XXsq: np.ndarray,
    rsum_XY1sq: np.ndarray,
    rsum_XY2sq: np.ndarray,
    rsum_Xdsq: np.ndarray,
    r_cross1: np.ndarray,
    r_cross2: np.ndarray,
    r_crossd: np.ndarray,
) -> dict:
    """
    Difference SEDs from 1D rowwise totals, matching the dense `outer` path and
    (with Numba) `zou_numba` without storing `n^2` matrices in Python.
    ij-block uses C++ scalar moment formulas (``zou_concordance2_fast_cpp`` 715–732).
    """
    tiXX_m = rsum_XX / (n - 1)
    tiXY1_m = rsum_XY1 / (n - 1)
    tiXY2_m = rsum_XY2 / (n - 1)
    tiXd_m = rsum_Xd / (n - 1)

    pairs = float(n * (n - 1))
    tXX = float(rsum_XX.sum() / pairs)
    tXY1 = float(rsum_XY1.sum() / pairs)
    tXY2 = float(rsum_XY2.sum() / pairs)
    sXX = float(rsum_XX.sum())
    sXY1 = float(rsum_XY1.sum())
    sXY2 = float(rsum_XY2.sum())
    sXd = float(rsum_Xd.sum())

    d1 = tXY1 / tXX
    d2 = tXY2 / tXX
    dd = d1 - d2
    c1 = d1 / 2.0 + 0.5
    c2 = d2 / 2.0 + 0.5
    _ = c1, c2

    denom_unb = float(n * (n - 1) * (n - 2) * (n - 3))
    tadj = 2.0 * (2 * n - 3) / (n * (n - 1))

    v_XX_u = (4.0 * np.sum(rsum_XX**2) - 2.0 * np.sum(rsum_XXsq) - tadj * sXX**2) / denom_unb
    v_XY1_u = (4.0 * np.sum(rsum_XY1**2) - 2.0 * np.sum(rsum_XY1sq) - tadj * sXY1**2) / denom_unb
    v_XY2_u = (4.0 * np.sum(rsum_XY2**2) - 2.0 * np.sum(rsum_XY2sq) - tadj * sXY2**2) / denom_unb
    v_Xd_u = (4.0 * np.sum(rsum_Xd**2) - 2.0 * np.sum(rsum_Xdsq) - tadj * sXd**2) / denom_unb

    cov_XY1_u = (4.0 * np.sum(rsum_XY1 * rsum_XX) - 2.0 * np.sum(r_cross1) - tadj * sXY1 * sXX) / denom_unb
    cov_XY2_u = (4.0 * np.sum(rsum_XY2 * rsum_XX) - 2.0 * np.sum(r_cross2) - tadj * sXY2 * sXX) / denom_unb
    cov_Xd_u = (4.0 * np.sum(rsum_Xd * rsum_XX) - 2.0 * np.sum(r_crossd) - tadj * sXd * sXX) / denom_unb

    _ = v_XY1_u, v_XY2_u, cov_XY1_u, cov_XY2_u

    unb_vard = (v_Xd_u - 2.0 * dd * cov_Xd_u + dd**2 * v_XX_u) / tXX**2 / 4.0
    unb_sed = float(np.sqrt(max(unb_vard, 0.0)))
    if not np.isfinite(unb_sed):
        unb_sed = 0.0

    v_iXX = float(np.var(tiXX_m, ddof=1))
    v_iXY1 = float(np.var(tiXY1_m, ddof=1))
    v_iXY2 = float(np.var(tiXY2_m, ddof=1))
    v_iXd = float(np.var(tiXd_m, ddof=1))
    cov_iXY1 = _cov2(tiXY1_m, tiXX_m)
    cov_iXY2 = _cov2(tiXY2_m, tiXX_m)
    cov_iXd = _cov2(tiXd_m, tiXX_m)
    _ = v_iXY1, cov_iXY1, v_iXY2, cov_iXY2

    s_ti1sqf = float(np.sum(rsum_XY1sq))
    s_ti2sqf = float(np.sum(rsum_XY2sq))
    s_tidsqf = float(np.sum(rsum_Xdsq))
    v_ijXX = _v_ij_cxx(sXX, sXX, pairs)
    v_ijXY1 = _v_ij_cxx(sXY1, s_ti1sqf, pairs)
    v_ijXY2 = _v_ij_cxx(sXY2, s_ti2sqf, pairs)
    v_ijXd = _v_ij_cxx(sXd, s_tidsqf, pairs)
    cov_ijXY1 = _cov_ij_cxx(sXY1, sXX, sXY1, pairs)
    cov_ijXY2 = _cov_ij_cxx(sXY2, sXX, sXY2, pairs)
    cov_ijXd = _cov_ij_cxx(sXd, sXX, sXd, pairs)
    _ = v_ijXY1, v_ijXY2, cov_ijXY1, cov_ijXY2

    UV_XX = (4.0 * (n - 2) * v_iXX + 2.0 * v_ijXX) / pairs
    UV_Xd = (4.0 * (n - 2) * v_iXd + 2.0 * v_ijXd) / pairs
    Ucov_Xd = (4.0 * (n - 2) * cov_iXd + 2.0 * cov_ijXd) / pairs
    cons_vard = (UV_Xd - 2.0 * dd * Ucov_Xd + dd**2 * UV_XX) / tXX**2 / 4.0
    cons_sed = float(np.sqrt(max(cons_vard, 0.0)))
    if not np.isfinite(cons_sed):
        cons_sed = 0.0

    SUV_XX = v_iXX / n
    SUV_Xd = v_iXd / n
    SUcov_Xd = cov_iXd / n
    simple_vard = (SUV_Xd - 2.0 * dd * SUcov_Xd + dd**2 * SUV_XX) / tXX**2
    simple_sed = float(np.sqrt(max(simple_vard, 0.0)))
    if not np.isfinite(simple_sed):
        simple_sed = 0.0

    return {
        "n": n,
        "unbiased_sed": unb_sed,
        "consistent_sed": cons_sed,
        "simple_sed": simple_sed,
    }


def zou_concordance_numpy_dense(
    standard: np.ndarray,
    test1: np.ndarray,
    test2: np.ndarray,
) -> dict:
    """O(n^2) reference using full ``outer`` matrices (for checks only)."""
    s = np.asarray(standard, dtype=np.float64).ravel()
    t1 = np.asarray(test1, dtype=np.float64).ravel()
    t2 = np.asarray(test2, dtype=np.float64).ravel()
    n = int(s.size)
    sign_T = np.sign(np.subtract.outer(s, s))
    sign_Y1 = np.sign(np.subtract.outer(t1, t1))
    sign_Y2 = np.sign(np.subtract.outer(t2, t2))
    tij_XX = sign_T**2
    tijXY1 = sign_T * sign_Y1
    tijXY2 = sign_T * sign_Y2
    tij_Xd = tijXY1 - tijXY2
    for m in (tij_XX, tijXY1, tijXY2, tij_Xd):
        np.fill_diagonal(m, 0.0)
    rsum_XX = tij_XX.sum(axis=1)
    rsum_XY1 = tijXY1.sum(axis=1)
    rsum_XY2 = tijXY2.sum(axis=1)
    rsum_Xd = tij_Xd.sum(axis=1)
    rsum_XXsq = (tij_XX**2).sum(axis=1)
    rsum_XY1sq = (tijXY1**2).sum(axis=1)
    rsum_XY2sq = (tijXY2**2).sum(axis=1)
    rsum_Xdsq = (tij_Xd**2).sum(axis=1)
    r_cross1 = (tijXY1 * tij_XX).sum(axis=1)
    r_cross2 = (tijXY2 * tij_XX).sum(axis=1)
    r_crossd = (tij_Xd * tij_XX).sum(axis=1)
    return zou_seds_from_row_stats(
        n,
        rsum_XX,
        rsum_XY1,
        rsum_XY2,
        rsum_Xd,
        rsum_XXsq,
        rsum_XY1sq,
        rsum_XY2sq,
        rsum_Xdsq,
        r_cross1,
        r_cross2,
        r_crossd,
    )


def zou_concordance(
    standard: np.ndarray,
    test1: np.ndarray,
    test2: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Return **difference**-scale SEDs used in the R script: ``simple_sed``,
    ``unbiased_sed``, ``consistent_sed`` (standard errors of ``c1 - c2`` on the
    concordance scale). ``alpha`` is kept for API parity with R (CIs); only SEDs
    are used for the simulation.

    Default implementation is **O(n log n)** (Fenwick + frequency tables, matching
    ``zou_concordance2_fast_cpp``). For regression checks see
    ``zou_concordance_numpy_dense`` and :func:`zou_concordance_numba`.
    """
    _ = alpha
    s = np.asarray(standard, dtype=np.float64).ravel()
    t1 = np.asarray(test1, dtype=np.float64).ravel()
    t2 = np.asarray(test2, dtype=np.float64).ravel()
    ok = np.isfinite(s) & np.isfinite(t1) & np.isfinite(t2)
    s, t1, t2 = s[ok], t1[ok], t2[ok]
    n = int(s.size)
    if n < 4:
        raise ValueError("Zou unbiased variance needs n >= 4")

    from .zou_fenwick import zou_concordance_fenwick

    return zou_concordance_fenwick(s, t1, t2)


def compute_zou_variances_diff(
    x_mat: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """``V = 4 * zou_sed^2 * n`` for each Zou SED, matching the R script."""
    n = int(y.size)
    zou = zou_concordance(
        y,
        x_mat[:, 0].copy(),
        x_mat[:, 1].copy(),
        alpha=alpha,
    )
    return {
        "var_zou_simple": 4.0 * zou["simple_sed"] ** 2 * n,
        "var_zou_unbiased": 4.0 * zou["unbiased_sed"] ** 2 * n,
        "var_zou_consistent": 4.0 * zou["consistent_sed"] ** 2 * n,
    }
