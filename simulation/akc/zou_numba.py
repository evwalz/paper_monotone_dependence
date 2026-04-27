"""O(n^2) time, O(n) memory row-stat build for Zou; avoids n×n Python allocations."""
from __future__ import annotations

import numpy as np
from numba import njit

from .zou_concordance import zou_seds_from_row_stats


@njit(cache=True)
def _fill_zou_row_stats(
    s: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    n: int,
) -> tuple:
    rsum_XX = np.zeros(n, dtype=np.float64)
    rsum_XY1 = np.zeros(n, dtype=np.float64)
    rsum_XY2 = np.zeros(n, dtype=np.float64)
    rsum_Xd = np.zeros(n, dtype=np.float64)
    rsum_XXsq = np.zeros(n, dtype=np.float64)
    rsum_XY1sq = np.zeros(n, dtype=np.float64)
    rsum_XY2sq = np.zeros(n, dtype=np.float64)
    rsum_Xdsq = np.zeros(n, dtype=np.float64)
    r_cross1 = np.zeros(n, dtype=np.float64)
    r_cross2 = np.zeros(n, dtype=np.float64)
    r_crossd = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = s[i] - s[j]
            st = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
            d1 = t1[i] - t1[j]
            sy1 = 1.0 if d1 > 0 else (-1.0 if d1 < 0 else 0.0)
            d2 = t2[i] - t2[j]
            sy2 = 1.0 if d2 > 0 else (-1.0 if d2 < 0 else 0.0)
            txx = st * st
            txy1 = st * sy1
            txy2 = st * sy2
            txd = txy1 - txy2
            rsum_XX[i] += txx
            rsum_XY1[i] += txy1
            rsum_XY2[i] += txy2
            rsum_Xd[i] += txd
            rsum_XXsq[i] += txx * txx
            rsum_XY1sq[i] += txy1 * txy1
            rsum_XY2sq[i] += txy2 * txy2
            rsum_Xdsq[i] += txd * txd
            r_cross1[i] += txy1 * txx
            r_cross2[i] += txy2 * txx
            r_crossd[i] += txd * txx
    return (
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


def zou_concordance_numba(
    s: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> dict:
    """Same as ``zou_concordance`` after filtering; no ``n^2`` dense arrays."""
    s = np.asarray(s, dtype=np.float64).ravel()
    t1 = np.asarray(t1, dtype=np.float64).ravel()
    t2 = np.asarray(t2, dtype=np.float64).ravel()
    n = int(s.size)
    rsums = _fill_zou_row_stats(s, t1, t2, n)
    return zou_seds_from_row_stats(n, *rsums)
