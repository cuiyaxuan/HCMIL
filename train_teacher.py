#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import sparse
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hcmil_stack.metrics import metrics, paired_bootstrap, permutation_pvalue
from hcmil_stack.model import PathwayOnlyNet
from hcmil_stack.classification import (bootstrap_intervals, braak_to_class, classification_metrics,
                                        confusion_rows, permutation_pvalue as classification_permutation_pvalue)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "student_gene_attribution.yaml"
GENE_MODELS = ["pca50_pls2", "direct_pls2", "supervised_screen_pls2"]
MODEL_NAMES = GENE_MODELS + ["best_gene_nested", "pathway_only_deep", "biological_ridge", "technical_only_ridge",
                             "stack_gene_pathway", "stack_gene_pathway_biological", "stack_full_with_technical"]


def load_config(path):
    whole = yaml.safe_load(Path(path).read_text(encoding="utf-8")); os.chdir(ROOT)
    teacher = whole["teacher"]
    return {
        "seed": whole["seed"],
        "data": {"linear_cache": whole["data"]["linear_cache"], "pathway_cache": whole["data"]["pathway_cache"],
                 "expected_development_donors": whole["data"].get("expected_development_donors"),
                 "expected_test_donors": whole["data"].get("expected_test_donors")},
        "gene_model": teacher["gene_model"], "pathway_model": teacher["pathway_model"],
        "clinical": whole["clinical"], "training": teacher["training"],
        "stacking": teacher["stacking"], "classification": teacher["classification"], "evaluation": teacher["evaluation"],
        "output_dir": teacher["output_dir"],
    }


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def strata(y, folds=5):
    rounded = np.rint(y).astype(int)
    if np.unique(rounded, return_counts=True)[1].min() >= folds:
        return rounded
    return np.asarray(pd.qcut(y, 4, labels=False, duplicates="drop"))


def select_hvgs(cells, cell_donor, cell_type, fit_donors, n_hvg, flavor):
    import anndata as ad
    import scanpy as sc

    rows = np.flatnonzero(np.isin(cell_donor, fit_donors)); a = ad.AnnData(X=cells[rows].copy())
    a.obs["cell_type"] = pd.Categorical(cell_type[rows].astype(str))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor=flavor, batch_key="cell_type", inplace=True)
    var = a.var
    if "highly_variable_rank" in var:
        order = np.argsort(var["highly_variable_rank"].fillna(np.inf).to_numpy())
    else:
        order = np.lexsort((-var["dispersions_norm"].fillna(-np.inf).to_numpy(),
                            -var["highly_variable_nbatches"].fillna(0).to_numpy()))
    chosen = order[:n_hvg].astype(np.int64)
    if len(chosen) != n_hvg:
        raise RuntimeError("Scanpy returned fewer HVGs than requested")
    return chosen


def logcpm(matrix, genes):
    library = np.asarray(matrix.sum(1)).ravel(); value = matrix[:, genes].toarray().astype(np.float32)
    return np.log1p(np.divide(value * 1e6, library[:, None], out=np.zeros_like(value), where=library[:, None] > 0))


def screen_order(x, y, fit_idx):
    block = x[fit_idx].astype(np.float64); target = y[fit_idx].astype(np.float64)
    block -= block.mean(0); target -= target.mean()
    denominator = np.sqrt((block * block).sum(0) * (target * target).sum())
    association = np.divide(block.T @ target, denominator, out=np.zeros(block.shape[1]), where=denominator > 0)
    return np.argsort(-np.abs(association))


def gene_predict(x, y, fit_idx, held_idx, kind, cfg, screen_n=None):
    scaler = StandardScaler().fit(x[fit_idx]); standardized = scaler.transform(x)
    if kind == "pca50_pls2":
        pca = PCA(n_components=min(int(cfg["pca_components"]), len(fit_idx) - 1, x.shape[1]), whiten=True, random_state=0)
        embedded = pca.fit(standardized[fit_idx]).transform(standardized)
        variance = float(pca.explained_variance_ratio_.sum()); selected = embedded.shape[1]
    elif kind == "direct_pls2":
        embedded, variance, selected = standardized, np.nan, standardized.shape[1]
    elif kind == "supervised_screen_pls2":
        if screen_n is None:
            raise ValueError("screen_n is required")
        columns = screen_order(standardized, y, fit_idx)[:int(screen_n)]
        embedded, variance, selected = standardized[:, columns], np.nan, len(columns)
    else:
        raise ValueError(kind)
    model = make_pipeline(StandardScaler(), PLSRegression(
        n_components=min(int(cfg["pls_components"]), len(fit_idx) - 1, embedded.shape[1]), scale=False, max_iter=1000
    )).fit(embedded[fit_idx], y[fit_idx])
    return model.predict(embedded[held_idx]).ravel(), variance, selected


class ClinicalEncoder:
    def __init__(self, names, kinds, numeric, categorical):
        self.names = list(map(str, names)); self.kinds = list(map(str, kinds))
        self.numeric = [x for x in numeric if x in self.names]
        self.categorical = [x for x in categorical if x in self.names]

    def transform_pair(self, raw, fit_idx, held_idx):
        frame = pd.DataFrame(raw.astype(str), columns=self.names); train_parts, held_parts = [], []
        for name in self.numeric:
            train = pd.to_numeric(frame.loc[fit_idx, name], errors="coerce"); held = pd.to_numeric(frame.loc[held_idx, name], errors="coerce")
            median = float(train.median()) if train.notna().any() else 0.0
            train_parts.append(train.fillna(median).to_numpy(float)[:, None]); held_parts.append(held.fillna(median).to_numpy(float)[:, None])
        for name in self.categorical:
            train_values = frame.loc[fit_idx, name].replace({"nan": "NA", "None": "NA"}).astype(str)
            held_values = frame.loc[held_idx, name].replace({"nan": "NA", "None": "NA"}).astype(str)
            levels = sorted(train_values.unique())
            train_parts.append(np.stack([(train_values == level).to_numpy(float) for level in levels], axis=1))
            held_parts.append(np.stack([(held_values == level).to_numpy(float) for level in levels], axis=1))
        if not train_parts:
            return np.zeros((len(fit_idx), 0)), np.zeros((len(held_idx), 0))
        return np.concatenate(train_parts, 1), np.concatenate(held_parts, 1)


def clinical_predict(raw, encoder, y, fit_idx, held_idx, alpha):
    train, held = encoder.transform_pair(raw, fit_idx, held_idx)
    if train.shape[1] == 0:
        return np.full(len(held_idx), y[fit_idx].mean(), np.float32)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha))).fit(train, y[fit_idx])
    return model.predict(held)


def pathway_statistics(value, counts, indices):
    mean = np.zeros(value.shape[1:], np.float32); std = np.ones(value.shape[1:], np.float32)
    for cell_type in range(value.shape[1]):
        rows = indices[counts[indices, cell_type] > 0]
        if len(rows):
            mean[cell_type] = value[rows, cell_type].mean(0); std[cell_type] = value[rows, cell_type].std(0)
            std[cell_type, std[cell_type] < 1e-5] = 1
    return mean, std


def pathway_transform(value, counts, mean, std):
    return ((value - mean[None]) / std[None]) * (counts > 0)[:, :, None]


def pathway_loader(pathway, counts, y, indices, batch, shuffle, seed):
    dataset = TensorDataset(torch.from_numpy(pathway[indices]), torch.from_numpy(counts[indices]), torch.from_numpy(y[indices]))
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


def fit_epochs(model, data, epochs, y_mean, y_std, cfg, device, validation=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["pathway_learning_rate"]),
                                  weight_decay=float(cfg["pathway_weight_decay"]))
    best, best_epoch, best_state, stale = float("inf"), 1, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for pathway, counts, target in data:
            pathway, counts, target = pathway.to(device), counts.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(pathway, counts), (target - y_mean) / y_std,
                                    beta=float(cfg["pathway_huber_beta"]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["pathway_gradient_clip"])); optimizer.step()
        if validation is None:
            continue
        model.eval(); total = n = 0
        with torch.no_grad():
            for pathway, counts, target in validation:
                target = target.to(device)
                loss = F.smooth_l1_loss(model(pathway.to(device), counts.to(device)), (target - y_mean) / y_std,
                                        beta=float(cfg["pathway_huber_beta"]))
                total += float(loss) * len(target); n += len(target)
        score = total / n
        if score < best - float(cfg["pathway_minimum_improvement"]):
            best, best_epoch, best_state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= int(cfg["pathway_patience"]):
            break
    if validation is not None:
        if best_state is None:
            raise RuntimeError("No pathway checkpoint selected")
        model.load_state_dict(best_state)
    return best_epoch


@torch.no_grad()
def pathway_infer(model, data, y_mean, y_std, device):
    model.eval(); result = []
    for pathway, counts, _ in data:
        result.append((model(pathway.to(device), counts.to(device)) * y_std + y_mean).cpu().numpy())
    return np.concatenate(result)


def pathway_predict(pathway, counts, y, fit_idx, held_idx, cfg, model_cfg, device, seed):
    inner_fit, validation = train_test_split(
        fit_idx, test_size=float(cfg["pathway_validation_fraction"]), random_state=seed,
        # This project predicts three Braak groups.  Stratifying checkpoint
        # selection by the seven raw stages can leave a stage represented by
        # one donor inside a nested fold, which sklearn correctly rejects.
        # The task-level groups remain well populated and match evaluation.
        stratify=braak_to_class(y[fit_idx]),
    )
    mean, std = pathway_statistics(pathway, counts, inner_fit); normalized = pathway_transform(pathway, counts, mean, std)
    y_mean, y_std = float(y[inner_fit].mean()), float(y[inner_fit].std()); seed_all(seed)
    model = PathwayOnlyNet(pathway.shape[2], pathway.shape[1], model_cfg).to(device)
    train_data = pathway_loader(normalized, counts, y, inner_fit, int(cfg["pathway_batch_size"]), True, seed)
    validation_data = pathway_loader(normalized, counts, y, validation, int(cfg["pathway_batch_size"]), False, seed)
    selected_epoch = fit_epochs(model, train_data, int(cfg["pathway_max_epochs"]), y_mean, y_std, cfg, device, validation_data)

    mean, std = pathway_statistics(pathway, counts, fit_idx); normalized = pathway_transform(pathway, counts, mean, std)
    y_mean, y_std = float(y[fit_idx].mean()), float(y[fit_idx].std()); seed_all(seed)
    model = PathwayOnlyNet(pathway.shape[2], pathway.shape[1], model_cfg).to(device)
    full_data = pathway_loader(normalized, counts, y, fit_idx, int(cfg["pathway_batch_size"]), True, seed)
    fit_epochs(model, full_data, selected_epoch, y_mean, y_std, cfg, device)
    held_data = pathway_loader(normalized, counts, y, held_idx, int(cfg["pathway_batch_size"]), False, seed)
    return pathway_infer(model, held_data, y_mean, y_std, device), selected_epoch


class NonnegativeRidgeStacker:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def fit(self, x, y):
        self.mean = x.mean(0); self.std = x.std(0); self.std[self.std < 1e-8] = 1
        z = (x - self.mean) / self.std; self.y_mean = float(y.mean())
        augmented_x = np.vstack((z, np.sqrt(self.alpha) * np.eye(z.shape[1])))
        augmented_y = np.r_[y - self.y_mean, np.zeros(z.shape[1])]
        self.coef, _ = nnls(augmented_x, augmented_y)
        return self

    def predict(self, x):
        return self.y_mean + ((x - self.mean) / self.std) @ self.coef


def fit_stacker(oof, outer_values, columns, y_fit, alpha):
    train = np.column_stack([oof[name] for name in columns]); held = np.column_stack([outer_values[name] for name in columns])
    model = NonnegativeRidgeStacker(alpha).fit(train, y_fit)
    return model.predict(held), model.coef


def fit_classification_teacher(oof, outer_values, columns, y_class, cfg):
    train = np.column_stack([oof[name] for name in columns]); held = np.column_stack([outer_values[name] for name in columns])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=float(cfg["logistic_c"]), class_weight=str(cfg["class_weight"]), max_iter=2000, solver="lbfgs",
    ))
    model.fit(train, y_class)
    probability = model.predict_proba(held)
    classes = model.named_steps["logisticregression"].classes_
    if not np.array_equal(classes, np.arange(3)): raise ValueError(f"Teacher classes mismatch: {classes}")
    return probability.astype(np.float32)


def check_alignment(meta, pathway):
    checks = [np.array_equal(meta["donor_names"].astype(str), pathway["donor_names"].astype(str)),
              np.array_equal(meta["split"].astype(str), pathway["split"].astype(str)),
              np.allclose(meta["braak"], pathway["braak"], equal_nan=True),
              np.array_equal(meta["braak_class"].astype(np.int64), pathway["braak_class"].astype(np.int64)),
              np.array_equal(meta["cell_type_names"].astype(str), pathway["cell_type_names"].astype(str))]
    if not all(checks):
        raise ValueError(f"Linear/pathway cache alignment failed: {checks}")


def run(cfg, force):
    linear_cache, pathway_cache = Path(cfg["data"]["linear_cache"]), Path(cfg["data"]["pathway_cache"])
    output = Path(cfg["output_dir"])
    if output.exists():
        if not force: raise FileExistsError(f"Output exists: {output}; use --force")
        shutil.rmtree(output)
    (output / "predictions").mkdir(parents=True)
    meta = np.load(linear_cache / "metadata.npz", allow_pickle=False)
    pathway_file = np.load(pathway_cache / "features.npz", allow_pickle=False); check_alignment(meta, pathway_file)
    cells = sparse.load_npz(linear_cache / "balanced_cells.npz"); donor_counts = sparse.load_npz(linear_cache / "donor_counts.npz")
    y = meta["braak"].astype(np.float32); y_class = meta["braak_class"].astype(np.int64)
    split = meta["split"].astype(str); donors = meta["donor_names"].astype(str)
    gene_names = meta["gene_names"].astype(str); pathway = pathway_file["pathway"].astype(np.float32)
    counts = pathway_file["counts"].astype(np.int64); clinical_raw = pathway_file["clinical_values"].astype(str)
    clinical_names = pathway_file["clinical_names"].astype(str); clinical_kinds = pathway_file["clinical_kinds"].astype(str)
    dev = np.flatnonzero((split == "train") & np.isfinite(y)); sealed = np.flatnonzero(split != "train")
    zero_cell = np.flatnonzero(counts.sum(1) == 0)
    if len(zero_cell): raise ValueError(f"Found {len(zero_cell)} donors with zero eligible cells: {donors[zero_cell].tolist()}")
    expected_dev, expected_test = cfg["data"].get("expected_development_donors"), cfg["data"].get("expected_test_donors")
    if expected_dev is not None and expected_test is not None and (len(dev), len(sealed)) != (int(expected_dev), int(expected_test)):
        raise ValueError(f"Expected development/test={expected_dev}/{expected_test}, found {len(dev)}/{len(sealed)}")
    if len(dev) == 0 or len(sealed) == 0: raise ValueError(f"Empty development or sealed split: {len(dev)}/{len(sealed)}")
    if len(dev) + len(sealed) != len(y):
        raise ValueError(
            f"Split/label inconsistency: development={len(dev)}, sealed={len(sealed)}, total={len(y)}"
        )

    ccfg = cfg["clinical"]
    biological_encoder = ClinicalEncoder(clinical_names, clinical_kinds, ccfg["biological_numeric"], ccfg["biological_categorical"])
    technical_encoder = ClinicalEncoder(clinical_names, clinical_kinds, ccfg["technical_numeric"], ccfg["technical_categorical"])
    tr, gcfg = cfg["training"], cfg["gene_model"]; seeds = list(map(int, tr["seeds"])); k = int(tr["outer_folds"])
    predictions = {name: np.zeros((len(seeds), len(y)), np.float32) for name in MODEL_NAMES}
    teacher_probability = np.zeros((len(seeds), len(y), 3), np.float32)
    fold_rows, hvg_rows, weight_rows = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; development={len(dev)}; sealed_test={len(sealed)}", flush=True)

    for seed_index, cv_seed in enumerate(seeds):
        outer = StratifiedKFold(k, shuffle=True, random_state=cv_seed)
        for fold, (fit_local, held_local) in enumerate(outer.split(dev, y_class[dev]), 1):
            fit_idx, held_idx = dev[fit_local], dev[held_local]
            genes = select_hvgs(cells, meta["cell_donor"], meta["cell_type"], fit_idx,
                                int(gcfg["n_hvg"]), gcfg["hvg_flavor"])
            for rank, gene_index in enumerate(genes, 1):
                hvg_rows.append({"seed": cv_seed, "fold": fold, "rank": rank, "gene": gene_names[gene_index]})
            expression = logcpm(donor_counts, genes)

            inner = StratifiedKFold(int(tr["inner_folds"]), shuffle=True, random_state=cv_seed * 100 + fold)
            oof = {name: np.full(len(fit_idx), np.nan, np.float32) for name in
                   GENE_MODELS + ["pathway_only_deep", "biological_ridge", "technical_only_ridge"]}
            screen_candidates = {int(n): np.full(len(fit_idx), np.nan, np.float32) for n in gcfg["supervised_screen_counts"]}
            inner_epochs = []
            for inner_fold, (inner_fit_local, inner_held_local) in enumerate(
                    inner.split(fit_idx, y_class[fit_idx]), 1):
                inner_fit, inner_held = fit_idx[inner_fit_local], fit_idx[inner_held_local]
                for name in ("pca50_pls2", "direct_pls2"):
                    value, _, _ = gene_predict(expression, y, inner_fit, inner_held, name, gcfg)
                    oof[name][inner_held_local] = value
                for n in screen_candidates:
                    value, _, _ = gene_predict(expression, y, inner_fit, inner_held, "supervised_screen_pls2", gcfg, n)
                    screen_candidates[n][inner_held_local] = value
                path_value, epoch = pathway_predict(pathway, counts, y, inner_fit, inner_held, tr, cfg["pathway_model"], device,
                                                    cv_seed * 100000 + fold * 1000 + inner_fold)
                oof["pathway_only_deep"][inner_held_local] = path_value; inner_epochs.append(epoch)
                oof["biological_ridge"][inner_held_local] = clinical_predict(
                    clinical_raw, biological_encoder, y, inner_fit, inner_held, ccfg["ridge_alpha"])
                oof["technical_only_ridge"][inner_held_local] = clinical_predict(
                    clinical_raw, technical_encoder, y, inner_fit, inner_held, ccfg["ridge_alpha"])
                print(f"seed={cv_seed} outer_fold={fold}/{k} inner_fold={inner_fold}/{int(tr['inner_folds'])} "
                      f"pathway_epoch={epoch}", flush=True)

            screen_scores = {n: spearmanr(y[fit_idx], value).statistic for n, value in screen_candidates.items()}
            selected_screen = max(screen_scores, key=screen_scores.get)
            oof["supervised_screen_pls2"] = screen_candidates[selected_screen]
            gene_scores = {name: spearmanr(y[fit_idx], oof[name]).statistic for name in GENE_MODELS}
            selected_gene = max(gene_scores, key=gene_scores.get)

            outer_values = {}
            pca_variance = np.nan
            for name in ("pca50_pls2", "direct_pls2"):
                value, variance, _ = gene_predict(expression, y, fit_idx, held_idx, name, gcfg)
                outer_values[name] = value
                if name == "pca50_pls2": pca_variance = variance
            outer_values["supervised_screen_pls2"], _, _ = gene_predict(
                expression, y, fit_idx, held_idx, "supervised_screen_pls2", gcfg, selected_screen)
            outer_values["best_gene_nested"] = outer_values[selected_gene]
            outer_values["pathway_only_deep"], outer_epoch = pathway_predict(
                pathway, counts, y, fit_idx, held_idx, tr, cfg["pathway_model"], device, cv_seed * 100000 + fold * 1000 + 99)
            outer_values["biological_ridge"] = clinical_predict(
                clinical_raw, biological_encoder, y, fit_idx, held_idx, ccfg["ridge_alpha"])
            outer_values["technical_only_ridge"] = clinical_predict(
                clinical_raw, technical_encoder, y, fit_idx, held_idx, ccfg["ridge_alpha"])

            stack_specs = {
                "stack_gene_pathway": [selected_gene, "pathway_only_deep"],
                "stack_gene_pathway_biological": [selected_gene, "pathway_only_deep", "biological_ridge"],
                "stack_full_with_technical": [selected_gene, "pathway_only_deep", "biological_ridge", "technical_only_ridge"],
            }
            for stack_name, columns in stack_specs.items():
                value, weights = fit_stacker(oof, outer_values, columns, y[fit_idx], cfg["stacking"]["nonnegative_ridge_alpha"])
                outer_values[stack_name] = value
                for column, weight in zip(columns, weights):
                    weight_rows.append({"seed": cv_seed, "fold": fold, "stack": stack_name, "base_model": column,
                                        "standardized_nonnegative_weight": float(weight)})
            teacher_columns = [selected_gene, "pathway_only_deep", "biological_ridge"]
            class_probability = fit_classification_teacher(
                oof, outer_values, teacher_columns, y_class[fit_idx], cfg["classification"])
            teacher_probability[seed_index, held_idx] = class_probability
            for name, value in outer_values.items(): predictions[name][seed_index, held_idx] = value
            row = {"seed": cv_seed, "fold": fold, "selected_screen_genes": selected_screen,
                   "selected_gene_model": selected_gene, "pca_variance": pca_variance,
                   "pathway_outer_epoch": outer_epoch, "pathway_inner_epoch_mean": float(np.mean(inner_epochs))}
            for name in MODEL_NAMES:
                row[f"{name}_spearman"] = float(spearmanr(y[held_idx], outer_values[name]).statistic)
            class_fold = classification_metrics(y_class[held_idx], class_probability)
            for name, value in class_fold.items(): row[f"braak3_teacher_{name}"] = value
            fold_rows.append(row)
            print(f"seed={cv_seed} fold={fold}/{k}; screen={selected_screen}; gene={selected_gene}; epoch={outer_epoch}", flush=True)

    rows, seed_rows, ensembles = [], [], {}
    for model_index, (name, values) in enumerate(predictions.items()):
        ensemble = values[:, dev].mean(0); ensembles[name] = ensemble; per_seed = []
        result = {"model": name, **metrics(y[dev], ensemble)}
        for seed_index, cv_seed in enumerate(seeds):
            one = {"model": name, "seed": cv_seed, **metrics(y[dev], values[seed_index, dev])}
            seed_rows.append(one); per_seed.append(one["spearman"])
        result.update(seed_spearman_mean=float(np.mean(per_seed)), seed_spearman_std=float(np.std(per_seed)),
                      seed_spearman_min=float(np.min(per_seed)), seed_spearman_max=float(np.max(per_seed)))
        result["permutation_p_one_sided"] = permutation_pvalue(
            y[dev], ensemble, int(cfg["evaluation"]["permutation_repeats"]), int(cfg["seed"]) + model_index)
        rows.append(result)
        frame = pd.DataFrame({"donor_id": donors[dev], "observed_braak": y[dev], "ensemble_oof": ensemble})
        for seed_index, cv_seed in enumerate(seeds): frame[f"seed_{cv_seed}_oof"] = values[seed_index, dev]
        frame.to_csv(output / "predictions" / f"{name}.csv", index=False)

    class_ensemble = teacher_probability[:, dev].mean(0)
    class_result = {"model": "teacher_braak3", **classification_metrics(y_class[dev], class_ensemble),
                    **bootstrap_intervals(y_class[dev], class_ensemble, int(cfg["seed"]) + 700,
                                          int(cfg["evaluation"]["bootstrap_repeats"]))}
    class_result["balanced_accuracy_permutation_p"] = classification_permutation_pvalue(
        y_class[dev], class_ensemble, int(cfg["seed"]) + 701, int(cfg["evaluation"]["permutation_repeats"]))
    class_seed_rows, class_confusion = [], confusion_rows(y_class[dev], class_ensemble, "teacher_braak3")
    for seed_index, cv_seed in enumerate(seeds):
        one = {"model": "teacher_braak3", "seed": cv_seed,
               **classification_metrics(y_class[dev], teacher_probability[seed_index, dev])}
        class_seed_rows.append(one)
        class_confusion.extend(confusion_rows(y_class[dev], teacher_probability[seed_index, dev], "teacher_braak3", cv_seed))
    class_frame = pd.DataFrame({"donor_id": donors[dev], "observed_braak": y[dev], "observed_class": y_class[dev],
                                "ensemble_predicted_class": class_ensemble.argmax(1)})
    for class_index in range(3): class_frame[f"ensemble_probability_class{class_index}"] = class_ensemble[:, class_index]
    for seed_index, cv_seed in enumerate(seeds):
        for class_index in range(3):
            class_frame[f"seed_{cv_seed}_probability_class{class_index}"] = teacher_probability[seed_index, dev, class_index]
    class_frame.to_csv(output / "predictions" / "teacher_braak3.csv", index=False)
    pd.DataFrame([class_result]).to_csv(output / "braak3_oof_metrics.csv", index=False)
    pd.DataFrame(class_seed_rows).to_csv(output / "braak3_seed_metrics.csv", index=False)
    pd.DataFrame(class_confusion).to_csv(output / "braak3_confusion_matrix.csv", index=False)

    pairs = [("direct_pls2", "pca50_pls2"), ("supervised_screen_pls2", "pca50_pls2"),
             ("stack_gene_pathway", "best_gene_nested"),
             ("stack_gene_pathway_biological", "stack_gene_pathway"),
             ("stack_full_with_technical", "stack_gene_pathway_biological")]
    bootstrap_rows = []
    for pair_index, (first, second) in enumerate(pairs):
        mean, low, high = paired_bootstrap(y[dev], ensembles[first], ensembles[second],
                                           int(cfg["evaluation"]["bootstrap_repeats"]), int(cfg["seed"]) + 100 + pair_index)
        bootstrap_rows.append({"first_model": first, "second_model": second,
                               "spearman_difference_first_minus_second": metrics(y[dev], ensembles[first])["spearman"] - metrics(y[dev], ensembles[second])["spearman"],
                               "bootstrap_mean_difference": mean, "ci_95_low": low, "ci_95_high": high})

    pd.DataFrame(rows).sort_values("spearman", ascending=False).to_csv(output / "oof_model_comparison.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "seed_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(hvg_rows).to_csv(output / "selected_hvgs.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "stacking_weights.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "paired_bootstrap_comparisons.csv", index=False)
    manifest = {"development_donors": len(dev), "untouched_test_donors": len(sealed), "test_metrics_computed": False,
                "outer_folds": k, "seeds": seeds,
                "stacking_training": "base-model predictions are inner OOF within each outer-training fold",
                "hvg_scope": "selected from each complete outer-training fold without labels; all supervised transforms refit in inner folds",
                "technical_covariates_reported_separately": True,
                "braak_3class_teacher": "multinomial logistic stack fit on inner-OOF gene/pathway/biological predictions",
                "braak_3class_ranges": [[0, 2], [3, 4], [5, 6]], "config": cfg}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).sort_values("spearman", ascending=False).to_string(index=False))
    print(pd.DataFrame([class_result]).to_string(index=False)); print(f"Wrote results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(DEFAULT_CONFIG)); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); run(load_config(args.config), args.force)
