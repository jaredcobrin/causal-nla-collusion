"""Statistics: group-clustered bootstrap CIs, Spearman correlation, rank AUROC."""

from __future__ import annotations

import numpy as np


def boot_ci(values, groups=None, n_boot: int = 2000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    """95% CI of stat(values), resampling whole groups (e.g. all samples of one question)."""
    values = np.asarray(values, float)
    groups = np.arange(len(values)) if groups is None else np.asarray(groups)
    keys, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(keys))]
    rng = np.random.default_rng(seed)
    stats = [stat(values[np.concatenate([members[i] for i in rng.integers(0, len(keys), len(keys))])])
             for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def spearman(x, y) -> float:
    rx, ry = (np.argsort(np.argsort(np.asarray(a, float))) for a in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(scores, labels) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    ranks = np.argsort(np.argsort(scores)) + 1
    n1, n0 = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
