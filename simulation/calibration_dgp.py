"""
Shared DGP and CLI→``acor_test`` alternative mapping for AGC/AKC calibration.
"""
from __future__ import annotations

import numpy as np


def _alternative_for_acor_test(simulation_alternative: str) -> str:
    """
    ``acor_test`` only accepts ``two.sided``, ``greater``, and ``less``.

    Our CLIs and Meng helper use the older labels ``one.sided`` / ``two.sided``.
    We map ``one.sided`` → ``greater`` to match the same upper-tail flavour as
    the Meng one-sided p-value in ``simulation.agc.helpers.meng_test_corr``.
    """
    if simulation_alternative == "two.sided":
        return "two.sided"
    if simulation_alternative == "one.sided":
        return "greater"
    if simulation_alternative in ("greater", "less"):
        return simulation_alternative
    raise ValueError(
        f"alternative must be 'two.sided', 'one.sided', 'greater', or 'less'; "
        f"got {simulation_alternative!r}"
    )


def _one_sample_discrete(n: int, sigma_1: float, sigma_2: float):
    x0 = np.random.normal(0, 1, n)
    z1 = np.random.normal(0, sigma_1, n)
    z2 = np.random.normal(0, sigma_2, n)
    y0 = np.round(np.random.normal(x0, 1, n))
    x1 = np.round(x0 + z1)
    x2 = np.round(x0 + z2)
    return y0, x1, x2


def _one_sample_continuous(n: int, sigma_1: float, sigma_2: float):
    x0 = np.random.normal(0, 1, n)
    z1 = np.random.normal(0, sigma_1, n)
    z2 = np.random.normal(0, sigma_2, n)
    y0 = np.random.normal(x0, 1, n)
    x1 = x0 + z1
    x2 = x0 + z2
    return y0, x1, x2


def sample_calibration_dgp(
    n: int, *, discrete: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single draw from the shared p-value simulation DGP: ``Y, X1, X2`` with
    i.i.d. structure (``sigma_1 = sigma_2 = 1`` as in the R scripts). Used
    by the AGC and AKC+Zou calibration drivers.
    """
    s1 = s2 = 1.0
    if discrete:
        return _one_sample_discrete(n, s1, s2)
    return _one_sample_continuous(n, s1, s2)
