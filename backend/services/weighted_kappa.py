"""Asymptotic standard errors for (weighted) Cohen's kappa (pure functions).

Fleiss, Cohen and Everitt (1969), "Large sample standard errors of kappa and
weighted kappa", Psychological Bulletin 72, 323-327. With joint proportions
p_ij, margins p_i. and p_.j and agreement weights w_ij (1 on the diagonal):

    w_i. = sum_j p_.j w_ij        w_.j = sum_i p_i. w_ij
    po_w = sum_ij w_ij p_ij       pe_w = sum_ij p_i. p_.j w_ij
    kappa_w = (po_w - pe_w) / (1 - pe_w)

    var(kappa_w)  = { sum_ij p_ij [w_ij (1 - pe_w) - (w_.j + w_i.)(1 - po_w)]^2
                      - (po_w pe_w - 2 pe_w + po_w)^2 } / (n (1 - pe_w)^4)
    var0(kappa_w) = { sum_ij p_i. p_.j [w_ij - (w_i. + w_.j)]^2 - pe_w^2 }
                    / (n (1 - pe_w)^2)

var is the variance around the estimate (confidence interval); var0 is the
variance under H0: kappa_w = 0 (z test). With identity weights var0 equals the
Fleiss (1981) null variance and var equals the unweighted FCE variance used by
R's psych::cohen.kappa.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

METHOD_NOTE = (
    "SE, CI and z test from the Fleiss, Cohen and Everitt (1969) large-sample variance "
    "of weighted kappa (CI around the estimate, normal approximation clipped to [-1, 1]; "
    "z test uses the variance under H0 of no agreement)."
)


def agreement_weights(k: int, kind: Optional[str]) -> np.ndarray:
    """k x k agreement weight matrix (1 = full agreement).

    ``None`` / ``"identity"`` give unweighted kappa, ``"linear"`` gives
    1 - |i - j| / (k - 1) and ``"quadratic"`` gives 1 - ((i - j) / (k - 1))^2.
    """
    idx = np.arange(k, dtype=float)
    dist = np.abs(idx[:, None] - idx[None, :])
    if kind in (None, "none", "identity"):
        return np.eye(k)
    if k < 2:
        return np.ones((k, k))
    if kind == "linear":
        return 1.0 - dist / (k - 1)
    if kind == "quadratic":
        return 1.0 - (dist / (k - 1)) ** 2
    raise ValueError(f"unknown kappa weights: {kind!r}")


def weighted_kappa_stats(cm, weights: Optional[str] = None, conf_level: float = 0.95) -> dict:
    """Weighted kappa with FCE standard errors from a k x k confusion matrix.

    Rows are rater 1 and columns are rater 2. Returns kappa, po_w, pe_w, the two
    variances / standard errors, the clipped normal CI, z and the two-sided p.
    Interval and test fields are ``None`` when chance agreement is perfect
    (pe_w >= 1, kappa undefined) or the null variance is not positive.
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    n = float(cm.sum())
    if n <= 0:
        raise ValueError("empty confusion matrix")
    k = cm.shape[0]
    w = agreement_weights(k, weights)
    p = cm / n
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    w_i = (w * col[None, :]).sum(axis=1)   # w_i. = sum_j p_.j w_ij
    w_j = (w * row[:, None]).sum(axis=0)   # w_.j = sum_i p_i. w_ij
    po_w = float((w * p).sum())
    pe_w = float((w * np.outer(row, col)).sum())
    out: dict = {
        "po": po_w, "pe": pe_w, "n": int(n),
        "kappa": None, "var": None, "se": None, "var_null": None, "se_null": None,
        "ci_low": None, "ci_high": None, "z": None, "p": None,
    }
    if 1.0 - pe_w <= 1e-12:
        return out
    kappa = (po_w - pe_w) / (1.0 - pe_w)
    out["kappa"] = float(kappa)

    a = w * (1.0 - pe_w) - (w_j[None, :] + w_i[:, None]) * (1.0 - po_w)
    var = (float((p * a ** 2).sum()) - (po_w * pe_w - 2.0 * pe_w + po_w) ** 2) / (n * (1.0 - pe_w) ** 4)
    var = max(var, 0.0)
    b = w - (w_i[:, None] + w_j[None, :])
    var0 = (float((np.outer(row, col) * b ** 2).sum()) - pe_w ** 2) / (n * (1.0 - pe_w) ** 2)

    crit = float(scipy_stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
    se = float(np.sqrt(var))
    out.update(
        var=float(var), se=se,
        ci_low=float(np.clip(kappa - crit * se, -1.0, 1.0)),
        ci_high=float(np.clip(kappa + crit * se, -1.0, 1.0)),
    )
    if var0 > 0:
        se0 = float(np.sqrt(var0))
        z = float(kappa / se0)
        out.update(var_null=float(var0), se_null=se0, z=z, p=float(2.0 * scipy_stats.norm.sf(abs(z))))
    return out
