from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)


CLASS_NAMES = np.asarray(["Braak_0_2", "Braak_3_4", "Braak_5_6"], dtype="U")


def braak_to_class(y):
    y = np.asarray(y, dtype=float)
    result = np.full(y.shape, -1, dtype=np.int64)
    result[np.isfinite(y) & (y <= 2)] = 0
    result[np.isfinite(y) & (y >= 3) & (y <= 4)] = 1
    result[np.isfinite(y) & (y >= 5)] = 2
    return result


def normalize_probabilities(probability):
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-7, None)
    return probability / probability.sum(1, keepdims=True)


def classification_metrics(y, probability):
    y = np.asarray(y, dtype=np.int64); probability = normalize_probabilities(probability)
    prediction = probability.argmax(1)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "macro_auroc_ovr": float(roc_auc_score(y, probability, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y, probability, labels=[0, 1, 2])),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, prediction, weights="quadratic")),
        "ordinal_class_mae": float(np.mean(np.abs(y - prediction))),
    }


def bootstrap_intervals(y, probability, seed, repeats):
    y = np.asarray(y, dtype=np.int64); probability = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(seed); values = {"macro_f1": [], "balanced_accuracy": [], "macro_auroc_ovr": []}
    for _ in range(int(repeats)):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size != 3:
            continue
        metric = classification_metrics(y[index], probability[index])
        for name in values: values[name].append(metric[name])
    result = {}
    for name, value in values.items():
        result[f"{name}_ci_low"] = float(np.quantile(value, .025))
        result[f"{name}_ci_high"] = float(np.quantile(value, .975))
    return result


def permutation_pvalue(y, probability, seed, repeats):
    y = np.asarray(y, dtype=np.int64); prediction = np.asarray(probability).argmax(1)
    observed = balanced_accuracy_score(y, prediction); rng = np.random.default_rng(seed)
    null = np.asarray([balanced_accuracy_score(rng.permutation(y), prediction) for _ in range(int(repeats))])
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def confusion_rows(y, probability, model, seed="ensemble"):
    matrix = confusion_matrix(np.asarray(y, dtype=np.int64), np.asarray(probability).argmax(1), labels=[0, 1, 2])
    rows = []
    for observed in range(3):
        for predicted in range(3):
            rows.append({"model": model, "seed": seed, "observed_class": observed,
                         "predicted_class": predicted, "n": int(matrix[observed, predicted])})
    return rows
