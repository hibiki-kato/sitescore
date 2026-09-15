"""Original code by Chirag Adwani (https://github.com/divide-by-zer0).
Platt scaling with Platt's Bayes/Laplace target priors.

Recalibrates a model's positive-class probabilities ``p`` in (0, 1) by fitting a
two-parameter logistic map on the recovered model logit ``z = log(p / (1 - p))``:

    p_cal = sigmoid(a * z + b)

The slope ``a`` corrects over/under-confidence (sharpness); the intercept ``b``
absorbs base-rate / prior shift. The map is monotonic for ``a > 0``, so it never
changes the candidate ranking (PR-AUC, F1-at-best-threshold are preserved) -- it
only relabels the score axis so a value reads as a probability.

Targets follow Platt (1999). Instead of hard 0/1 labels, positives use
``t+ = (N+ + 1) / (N+ + 2)`` and negatives use ``t- = 1 / (N- + 2)``. These
Bayes/Laplace-smoothed targets keep the fit from saturating at 0/1 and matter
most for the rare start/stop heads where N+ is small.
"""

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


def prob_to_logit(p, eps=_EPS):
    """Recover the model logit from a stored positive-class probability."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    """Numerically stable elementwise logistic sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def laplace_targets(y):
    """Platt's Bayes/Laplace soft targets for a boolean label vector."""
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    targets = np.where(y, t_pos, t_neg)
    return targets, n_pos, n_neg, t_pos, t_neg


def fit_platt(z, y):
    """Fit ``p_cal = sigmoid(a * z + b)`` to Bayes/Laplace targets.

    Parameters
    ----------
    z : array of recovered model logits (see :func:`prob_to_logit`).
    y : boolean array of truth labels aligned with ``z``.

    Returns ``(a, b, info)``. The objective is convex in ``(a, b)`` (logistic
    regression with soft labels), so the L-BFGS solution is the global optimum.
    """
    z = np.asarray(z, dtype=np.float64)
    targets, n_pos, n_neg, t_pos, t_neg = laplace_targets(y)
    if n_pos == 0 or n_neg == 0:
        # Degenerate head (all one class); identity map is the safe fallback.
        return (
            1.0,
            0.0,
            {
                "n_pos": n_pos,
                "n_neg": n_neg,
                "t_pos": t_pos,
                "t_neg": t_neg,
                "base_rate": n_pos / max(n_pos + n_neg, 1),
                "converged": False,
                "degenerate": True,
                "loss": float("nan"),
                "n_iter": 0,
            },
        )

    def objective(params):
        a, b = params
        f = a * z + b
        prob = sigmoid(f)
        # Cross-entropy to soft targets via stable softplus:
        #   CE = t * softplus(-f) + (1 - t) * softplus(f)
        loss = np.mean(targets * np.logaddexp(0.0, -f) + (1.0 - targets) * np.logaddexp(0.0, f))
        residual = prob - targets
        grad = np.array([np.mean(residual * z), np.mean(residual)])
        return loss, grad

    result = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    a, b = float(result.x[0]), float(result.x[1])
    info = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "t_pos": t_pos,
        "t_neg": t_neg,
        "base_rate": n_pos / max(n_pos + n_neg, 1),
        "converged": bool(result.success),
        "degenerate": False,
        "loss": float(result.fun),
        "n_iter": int(result.nit),
    }
    return a, b, info


def apply_platt(p, a, b):
    """Apply a fitted Platt map to stored probabilities ``p``."""
    return sigmoid(a * prob_to_logit(p) + b)


def expected_calibration_error(p, y, n_bins=15):
    """Equal-width-bin ECE between predicted probabilities and empirical frequency."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / p.size) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def reliability_table(p, y, n_bins=15):
    """Per-bin reliability rows for plotting and JSON export."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": count,
                "mean_pred": float(p[mask].mean()) if count else None,
                "empirical": float(y[mask].mean()) if count else None,
            }
        )
    return rows
