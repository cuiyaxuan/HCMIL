from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    return {
        "n": len(y), "spearman": float(spearmanr(y, prediction).statistic),
        "r2": float(r2_score(y, prediction)), "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
    }


def permutation_pvalue(y, prediction, repeats, seed):
    observed = spearmanr(y, prediction).statistic; rng = np.random.default_rng(seed)
    null = np.asarray([spearmanr(rng.permutation(y), prediction).statistic for _ in range(repeats)])
    return float((1 + np.sum(null >= observed)) / (repeats + 1))


def paired_bootstrap(y, first, second, repeats, seed):
    y, first, second = map(np.asarray, (y, first, second)); rng = np.random.default_rng(seed); values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size > 1:
            values.append(spearmanr(y[idx], first[idx]).statistic - spearmanr(y[idx], second[idx]).statistic)
    values = np.asarray(values, float)
    return float(np.nanmean(values)), float(np.nanquantile(values, .025)), float(np.nanquantile(values, .975))
