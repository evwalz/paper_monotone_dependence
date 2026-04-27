"""
O(n log n) Zou concordance row statistics via Fenwick trees and frequency tables,
port of ``zou_concordance2_fast_cpp`` in ``zou_concordance_fast.cpp``.
"""
from __future__ import annotations

import numpy as np

from .zou_concordance import zou_seds_from_row_stats


class FenwickTree:
    __slots__ = ("_n", "tree")

    def __init__(self, n: int) -> None:
        self._n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int = 1) -> None:
        n = self._n
        while i <= n:
            self.tree[i] += delta
            i += i & -i

    def query(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def reset(self) -> None:
        t = self.tree
        for i in range(len(t)):
            t[i] = 0


def _compress_1based(vals: np.ndarray) -> tuple[np.ndarray, int]:
    u = np.unique(np.sort(np.asarray(vals, dtype=np.float64)))
    m = int(u.size)
    if m == 0:
        return np.zeros(len(vals), dtype=np.int32), 0
    c = np.searchsorted(u, np.asarray(vals, dtype=np.float64), side="left")
    c = c.astype(np.int32) + 1
    return c, m


def compute_tiXY_fenwick(T: np.ndarray, Y: np.ndarray, n: int) -> np.ndarray:
    """``tiXY[i] = sum_{j != i} sign(T_i - T_j) * sign(Y_i - Y_j)``."""
    t = T.astype(np.float64, copy=False)
    y = Y.astype(np.float64, copy=False)
    yc, my = _compress_1based(y)
    ord_ = np.argsort(t, kind="mergesort")
    ti_xy = np.zeros(n, dtype=np.float64)
    t_sorted = t[ord_]
    yc_s = yc[ord_]

    # Forward pass
    tree = FenwickTree(my)
    i = 0
    while i < n:
        j = i
        while j < n and t_sorted[j] == t_sorted[i]:
            j += 1
        for k in range(i, j):
            idx = int(ord_[k])
            c = int(yc_s[k])
            below_leq = tree.query(c)
            below_lt = tree.query(c - 1)
            total = tree.query(my)
            below_gt = total - below_leq
            ti_xy[idx] += float(below_lt - below_gt)
        for k in range(i, j):
            tree.update(int(yc_s[k]))
        i = j

    # Reverse pass
    tree = FenwickTree(my)
    i = n - 1
    while i >= 0:
        j = i
        while j >= 0 and t_sorted[j] == t_sorted[i]:
            j -= 1
        for k in range(j + 1, i + 1):
            idx = int(ord_[k])
            c = int(yc_s[k])
            above_leq = tree.query(c)
            above_lt = tree.query(c - 1)
            total = tree.query(my)
            above_gt = total - above_leq
            ti_xy[idx] += float(above_gt - above_lt)
        for k in range(j + 1, i + 1):
            tree.update(int(yc_s[k]))
        i = j
    return ti_xy


def compute_tiXX_freq(t: np.ndarray, n: int) -> np.ndarray:
    u, cnt = np.unique(t.astype(np.float64, copy=False), return_counts=True)
    fmap: dict[float, int] = {float(uu): int(c) for uu, c in zip(u, cnt)}
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(n - fmap[float(t[i])])
    return out


def compute_tiXYsq_freq(
    t: np.ndarray,
    y: np.ndarray,
    n: int,
    ti_xx: np.ndarray,
) -> np.ndarray:
    t64 = t.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    u, cnt = np.unique(y64, return_counts=True)
    f_y: dict[float, int] = {float(uu): int(c) for uu, c in zip(u, cnt)}

    keys = np.column_stack((t64, y64))
    uniq, ic = np.unique(keys, axis=0, return_inverse=True)
    c_ty = np.bincount(ic)
    f_ty: dict[tuple[float, float], int] = {
        (float(uniq[i, 0]), float(uniq[i, 1])): int(c_ty[i]) for i in range(len(uniq))
    }
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        ft = f_ty.get((float(t64[i]), float(y64[i])), 0)
        out[i] = float(ti_xx[i] - (f_y[float(y64[i])] - ft))
    return out


def compute_pairwise_concordance_subset(
    y1: np.ndarray,
    y2: np.ndarray,
    indices: np.ndarray,
    out: np.ndarray,
) -> None:
    """In-place: ``out[idx] +=`` pairwise concordance of (Y1,Y2) for idx in ``indices``."""
    m = int(indices.size)
    if m <= 1:
        return
    y1a = y1.astype(np.float64, copy=False)
    y2a = y2.astype(np.float64, copy=False)
    y2sub = y2a[indices]
    u2, inv = np.unique(y2sub, return_inverse=True)
    m2 = int(u2.size)
    y2c = inv.astype(np.int32) + 1

    ord_ = np.argsort(y1a[indices], kind="mergesort")
    y1g = y1a[indices]

    # Forward
    tree = FenwickTree(m2)
    i = 0
    while i < m:
        j = i
        y1v = y1g[ord_[i]]
        while j < m and y1g[ord_[j]] == y1v:
            j += 1
        for k in range(i, j):
            local_k = int(ord_[k])
            idx0 = int(indices[local_k])
            yc = int(y2c[local_k])
            below_leq = tree.query(yc)
            below_lt = tree.query(yc - 1)
            tot = tree.query(m2)
            below_gt = tot - below_leq
            out[idx0] += float(below_lt - below_gt)
        for k in range(i, j):
            tree.update(int(y2c[ord_[k]]))
        i = j

    # Reverse
    tree = FenwickTree(m2)
    i2 = m - 1
    while i2 >= 0:
        j = i2
        y1v = y1g[ord_[i2]]
        while j >= 0 and y1g[ord_[j]] == y1v:
            j -= 1
        for k in range(j + 1, i2 + 1):
            local_k = int(ord_[k])
            idx0 = int(indices[local_k])
            yc = int(y2c[local_k])
            above_leq = tree.query(yc)
            above_lt = tree.query(yc - 1)
            tot = tree.query(m2)
            above_gt = tot - above_leq
            out[idx0] += float(above_gt - above_lt)
        for k in range(j + 1, i2 + 1):
            tree.update(int(y2c[ord_[k]]))
        i2 = j


def compute_tiY1Y2(
    t: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    n: int,
) -> np.ndarray:
    ti = np.zeros(n, dtype=np.float64)
    all_idx = np.arange(n, dtype=np.int32)
    compute_pairwise_concordance_subset(y1, y2, all_idx, ti)

    t64 = t.astype(np.float64, copy=False)
    u, inv, cnt = np.unique(t64, return_inverse=True, return_counts=True)
    Ttied = np.zeros(n, dtype=np.float64)
    for g in range(len(u)):
        if int(cnt[g]) <= 1:
            continue
        msk = inv == g
        idx = np.nonzero(msk)[0].astype(np.int32)
        compute_pairwise_concordance_subset(y1, y2, idx, Ttied)

    return ti - Ttied


def zou_row_stats_fenwick(
    t: np.ndarray, y1: np.ndarray, y2: np.ndarray, n: int
) -> tuple[np.ndarray, ...]:
    """The ten 1D row summaries matching ``zou_seds_from_row_stats`` / dense / Numba."""
    ti_xy1 = compute_tiXY_fenwick(t, y1, n)
    ti_xy2 = compute_tiXY_fenwick(t, y2, n)
    ti_xx = compute_tiXX_freq(t, n)
    ti_xd = ti_xy1 - ti_xy2
    ti_xy1sq = compute_tiXYsq_freq(t, y1, n, ti_xx)
    ti_xy2sq = compute_tiXYsq_freq(t, y2, n, ti_xx)
    ti_y1y2 = compute_tiY1Y2(t, y1, y2, n)
    ti_xdsq = ti_xy1sq + ti_xy2sq - 2.0 * ti_y1y2

    rsum_XX = ti_xx
    rsum_XY1 = ti_xy1
    rsum_XY2 = ti_xy2
    rsum_Xd = ti_xd
    rsum_XXsq = rsum_XX
    rsum_XY1sq = ti_xy1sq
    rsum_XY2sq = ti_xy2sq
    rsum_Xdsq = ti_xdsq
    r_cross1 = rsum_XY1
    r_cross2 = rsum_XY2
    r_crossd = rsum_Xd
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


def zou_concordance_fenwick(
    s: np.ndarray, t1: np.ndarray, t2: np.ndarray
) -> dict:
    s = np.asarray(s, dtype=np.float64).ravel()
    t1 = np.asarray(t1, dtype=np.float64).ravel()
    t2 = np.asarray(t2, dtype=np.float64).ravel()
    n = int(s.size)
    rsums = zou_row_stats_fenwick(s, t1, t2, n)
    return zou_seds_from_row_stats(n, *rsums)
