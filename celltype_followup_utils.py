from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from hcmil_stack.classification import classification_metrics, normalize_probabilities


BRAAK_VALUES = np.asarray([1.0, 3.5, 5.5], dtype=float)
CONTRIBUTION_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "macro_auroc_ovr",
    "raw_braak_spearman",
    "raw_braak_r2",
)


def evaluate(y_class, y_braak, probability):
    probability = normalize_probabilities(probability)
    result = classification_metrics(y_class, probability)
    continuous = probability @ BRAAK_VALUES
    result["raw_braak_spearman"] = float(spearmanr(y_braak, continuous).statistic)
    result["raw_braak_r2"] = float(r2_score(y_braak, continuous))
    return result


def expected_braak(probability):
    return normalize_probabilities(probability) @ BRAAK_VALUES


def intervene(state, counts, type_indices, mode, permutation=None):
    """Return copied arrays with a pre-specified inference-only intervention."""
    type_indices = np.atleast_1d(type_indices).astype(int)
    value, number = state.copy(), counts.copy()
    if mode == "leave_out":
        value[:, type_indices] = 0
        number[:, type_indices] = 0
    elif mode == "only":
        keep = np.zeros(state.shape[1], dtype=bool)
        keep[type_indices] = True
        value[:, ~keep] = 0
        number[:, ~keep] = 0
    elif mode == "state_permutation":
        if permutation is None:
            raise ValueError("state_permutation requires permutation")
        value[:, type_indices] = value[permutation][:, type_indices]
    elif mode == "count_permutation":
        if permutation is None:
            raise ValueError("count_permutation requires permutation")
        number[:, type_indices] = number[permutation][:, type_indices]
    elif mode == "joint_permutation":
        if permutation is None:
            raise ValueError("joint_permutation requires permutation")
        value[:, type_indices] = value[permutation][:, type_indices]
        number[:, type_indices] = number[permutation][:, type_indices]
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")
    return value, number


def metric_delta(y_class, y_braak, perturbed, baseline):
    a = evaluate(y_class, y_braak, perturbed)
    b = evaluate(y_class, y_braak, baseline)
    return {key: a[key] - b[key] for key in CONTRIBUTION_METRICS}


def paired_bootstrap_delta(y_class, y_braak, perturbed, baseline, repeats, seed):
    rng = np.random.default_rng(seed)
    observed = metric_delta(y_class, y_braak, perturbed, baseline)
    values = {key: [] for key in CONTRIBUTION_METRICS}
    for _ in range(int(repeats)):
        index = rng.integers(0, len(y_class), len(y_class))
        if np.unique(y_class[index]).size != 3:
            continue
        delta = metric_delta(y_class[index], y_braak[index], perturbed[index], baseline[index])
        for key in values:
            values[key].append(delta[key])
    result = {}
    for key, observed_value in observed.items():
        sample = np.asarray(values[key], dtype=float)
        result[f"delta_{key}"] = observed_value
        result[f"delta_{key}_ci_low"] = float(np.quantile(sample, 0.025))
        result[f"delta_{key}_ci_high"] = float(np.quantile(sample, 0.975))
    result["bootstrap_valid_repeats"] = len(next(iter(values.values())))
    return result

