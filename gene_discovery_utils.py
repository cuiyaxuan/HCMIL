#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjustment over every finite p-value supplied."""
    pvalues = np.asarray(pvalues, dtype=float)
    output = np.full(len(pvalues), np.nan)
    keep = np.isfinite(pvalues)
    if not np.any(keep):
        return output
    values = pvalues[keep]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    output[keep] = restored
    return output


def covariate_transform(frame, fit_rows, numeric, categorical):
    work = frame.copy()
    for name in numeric:
        work[name] = pd.to_numeric(work[name], errors="coerce")
    for name in categorical:
        work[name] = work[name].astype("string").fillna("NA")
    number_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    category_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transform = ColumnTransformer([
        ("numeric", number_pipe, numeric),
        ("categorical", category_pipe, categorical),
    ], remainder="drop")
    transform.fit(work.iloc[fit_rows])
    return transform, work


def percentile(series):
    return series.rank(pct=True, method="average").fillna(0.0)


def write_markdown_table(frame, columns, maximum=30):
    if frame.empty:
        return "No genes passed."
    shown = frame.loc[:, columns].head(maximum).copy()
    return shown.to_markdown(index=False)

