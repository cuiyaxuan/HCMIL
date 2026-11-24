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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hcmil_stack.classification import (bootstrap_intervals, classification_metrics, confusion_rows,
                                        normalize_probabilities, permutation_pvalue)
from student_model import CellScaleTeacherStudent

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def load(path):
    cfg = yaml.safe_load(Path(path).read_text()); os.chdir(ROOT); return cfg


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class BiologicalEncoder:
    def __init__(self, names, kinds, cfg):
        self.names, self.kinds, self.cfg = list(map(str, names)), list(map(str, kinds)), cfg

    def fit_transform(self, raw, fit, target):
        frame = pd.DataFrame(raw.astype(str), columns=self.names); pieces = []
        for name in self.cfg["biological_numeric"]:
            if name not in frame: continue
            value = pd.to_numeric(frame[name], errors="coerce"); median = float(value.iloc[fit].median()); value = value.fillna(median)
            mean, std = float(value.iloc[fit].mean()), float(value.iloc[fit].std())
            pieces.append(((value.iloc[target].to_numpy(float) - mean) / max(std, 1e-6))[:, None])
        for name in self.cfg["biological_categorical"]:
            if name not in frame: continue
            value = frame[name].replace({"nan": "NA", "None": "NA"}).astype(str); levels = sorted(value.iloc[fit].unique())
            pieces.extend([(value.iloc[target].to_numpy() == level).astype(np.float32)[:, None] for level in levels])
        return np.concatenate(pieces, 1).astype(np.float32) if pieces else np.zeros((len(target), 0), np.float32)


def normalize_state(state, counts, fit, target):
    present = counts[fit] > 0; fit_values = state[fit]; denominator = present.sum((0, 1)).clip(min=1)
    mean = (fit_values * present[..., None]).sum((0, 1)) / denominator
    var = (((fit_values - mean) ** 2) * present[..., None]).sum((0, 1)) / denominator
    value = (state[target] - mean) / np.sqrt(var + 1e-5); value[counts[target] == 0] = 0
    return value.astype(np.float32)


def class_weights(y):
    count = np.bincount(np.asarray(y, dtype=np.int64), minlength=3).astype(np.float32)
    return (len(y) / (3.0 * np.maximum(count, 1))).astype(np.float32)


def make_loader(state, counts, bio, y_class, teacher_probability, batch, shuffle, seed):
    data = TensorDataset(torch.from_numpy(state), torch.from_numpy(counts.astype(np.float32)), torch.from_numpy(bio),
                         torch.from_numpy(y_class.astype(np.int64)), torch.from_numpy(teacher_probability.astype(np.float32)))
    return DataLoader(data, batch_size=batch, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


def loss_value(model, batch, stats, cfg, device):
    state, counts, bio, y_class, teacher_probability = [x.to(device) for x in batch]
    student, cell_only, _, prototype = model(state, counts, bio)
    target_mean = state[..., :32]; present = counts > 0
    reconstruction = (((prototype - target_mean) ** 2).mean(-1) * present).sum() / present.sum().clamp_min(1)
    weight = torch.as_tensor(stats["class_weight"], dtype=torch.float32, device=device)
    temperature = float(cfg["distillation_temperature"])
    soft_teacher = teacher_probability.clamp_min(1e-7).pow(1.0 / temperature)
    soft_teacher = soft_teacher / soft_teacher.sum(1, keepdim=True)
    distillation = F.kl_div(F.log_softmax(student / temperature, dim=1), soft_teacher,
                            reduction="batchmean") * temperature ** 2
    return (float(cfg["reconstruction_weight"]) * reconstruction +
            float(cfg["teacher_weight"]) * distillation +
            float(cfg["student_braak_weight"]) * F.cross_entropy(student, y_class, weight=weight) +
            float(cfg["cell_only_braak_weight"]) * F.cross_entropy(cell_only, y_class, weight=weight))


def fit_epochs(model, train, cfg, stats, device, validation=None, epochs=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    best, best_state, best_epoch, stale = np.inf, None, 1, 0; history = []
    for epoch in range(1, int(epochs or cfg["max_epochs"]) + 1):
        model.train(); total = n = 0
        for batch in train:
            optimizer.zero_grad(set_to_none=True); loss = loss_value(model, batch, stats, cfg, device); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"])); optimizer.step()
            total += float(loss) * len(batch[0]); n += len(batch[0])
        row = {"epoch": epoch, "train_loss": total / n}
        if validation is not None:
            model.eval(); value = vn = 0
            with torch.no_grad():
                for batch in validation:
                    loss = loss_value(model, batch, stats, cfg, device); value += float(loss) * len(batch[0]); vn += len(batch[0])
            score = value / vn; row["validation_loss"] = score
            if score < best - float(cfg["minimum_improvement"]):
                best, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
            else: stale += 1
            if stale >= int(cfg["patience"]): history.append(row); break
        history.append(row)
    if validation is not None:
        if best_state is None: raise RuntimeError("No student checkpoint selected")
        model.load_state_dict(best_state)
    return best_epoch, history


@torch.no_grad()
def infer(model, loader, device):
    model.eval(); student, cell, donor, proto = [], [], [], []
    for state, counts, bio, _, _ in loader:
        s, c, d, p = model(state.to(device), counts.to(device), bio.to(device))
        student.append(torch.softmax(s, 1).cpu().numpy()); cell.append(torch.softmax(c, 1).cpu().numpy())
        donor.append(d.cpu().numpy()); proto.append(p.cpu().numpy())
    return np.concatenate(student), np.concatenate(cell), np.concatenate(donor), np.concatenate(proto)


def prediction_frame(donors, braak, y_class, ensemble, values, seeds):
    frame = pd.DataFrame({"donor_id": donors, "observed_braak": braak, "observed_class": y_class,
                          "ensemble_predicted_class": ensemble.argmax(1)})
    for class_index in range(3): frame[f"ensemble_probability_class{class_index}"] = ensemble[:, class_index]
    for seed_index, seed in enumerate(seeds):
        for class_index in range(3): frame[f"seed_{seed}_probability_class{class_index}"] = values[seed_index, :, class_index]
    return frame


def run_mode(cfg, mode, force=False):
    scvi_dir, student_cfg = Path(cfg["data"]["scvi_cache"]), cfg["student"]
    z = np.load(scvi_dir / f"features_{mode}_cells.npz", allow_pickle=False)
    output = Path(student_cfg["output_dir"]) / mode; baseline_output = Path("artifacts/non_teacher_student/scvi_summary_baselines") / mode
    for path in (output, baseline_output):
        if path.exists():
            if not force: raise FileExistsError(f"Output exists: {path}; use --force")
            shutil.rmtree(path)
        path.mkdir(parents=True)
    state, counts = z["state"].astype(np.float32), z["exact_cell_counts"].astype(np.int64)
    braak, y_class = z["braak"].astype(np.float32), z["braak_class"].astype(np.int64)
    split, donors = z["split"].astype(str), z["donor_names"].astype(str)
    dev = np.flatnonzero((split == "train") & (y_class >= 0)); sealed = np.flatnonzero(split != "train")
    expected = (int(cfg["data"]["expected_development_donors"]), int(cfg["data"]["expected_test_donors"]))
    if (len(dev), len(sealed)) != expected: raise ValueError(f"Expected development/test={expected}, found {len(dev)}/{len(sealed)}")
    zero_cell = np.flatnonzero(counts.sum(1) == 0)
    if len(zero_cell): raise ValueError(f"Found zero-cell donors: {donors[zero_cell].tolist()}")
    teacher_file = Path(cfg["teacher"]["output_dir"]) / "predictions" / "teacher_braak3.csv"
    teacher_frame = pd.read_csv(teacher_file, dtype={"donor_id": str}).set_index("donor_id")
    teacher = np.full((len(donors), 3), np.nan, np.float32)
    probability_columns = [f"ensemble_probability_class{x}" for x in range(3)]
    for i in dev: teacher[i] = teacher_frame.loc[donors[i], probability_columns].to_numpy(float)
    teacher[dev] = normalize_probabilities(teacher[dev])
    names, kinds, raw = z["biological_names"], z["biological_kinds"], z["biological_values"]
    bio_encoder = BiologicalEncoder(names, kinds, cfg["clinical"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds, folds = list(map(int, student_cfg["seeds"])), int(student_cfg["outer_folds"])
    predictions = {"teacher": np.tile(teacher[dev][None], (len(seeds), 1, 1)),
                   "cell_only": np.zeros((len(seeds), len(dev), 3), np.float32),
                   "student": np.zeros((len(seeds), len(dev), 3), np.float32),
                   "summary_logistic": np.zeros((len(seeds), len(dev), 3), np.float32)}
    histories, fold_rows = [], []
    representative = {"donor_embedding": np.zeros((len(dev), int(student_cfg["donor_dim"])), np.float32),
                      "prototype": np.zeros((len(dev), len(z["cell_type_names"]), 32), np.float32), "fold": np.zeros(len(dev), int)}
    for seed_index, seed in enumerate(seeds):
        outer = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for fold, (fit_local, held_local) in enumerate(outer.split(dev, y_class[dev]), 1):
            fit_idx, held_idx = dev[fit_local], dev[held_local]
            inner_fit, inner_val = train_test_split(fit_idx, test_size=float(student_cfg["validation_fraction"]),
                                                    random_state=seed + fold, stratify=y_class[fit_idx])
            inner_state = normalize_state(state, counts, inner_fit, inner_fit); val_state = normalize_state(state, counts, inner_fit, inner_val)
            inner_bio = bio_encoder.fit_transform(raw, inner_fit, inner_fit); val_bio = bio_encoder.fit_transform(raw, inner_fit, inner_val)
            stats = {"class_weight": class_weights(y_class[inner_fit])}; seed_all(seed * 100 + fold)
            model = CellScaleTeacherStudent(state.shape[2], state.shape[1], inner_bio.shape[1], student_cfg).to(device)
            selected, history = fit_epochs(model,
                make_loader(inner_state, counts[inner_fit], inner_bio, y_class[inner_fit], teacher[inner_fit], int(student_cfg["batch_size"]), True, seed),
                student_cfg, stats, device,
                make_loader(val_state, counts[inner_val], val_bio, y_class[inner_val], teacher[inner_val], int(student_cfg["batch_size"]), False, seed))
            for row in history: histories.append({"mode": mode, "seed": seed, "fold": fold, "phase": "selection", **row})
            fit_state = normalize_state(state, counts, fit_idx, fit_idx); held_state = normalize_state(state, counts, fit_idx, held_idx)
            fit_bio = bio_encoder.fit_transform(raw, fit_idx, fit_idx); held_bio = bio_encoder.fit_transform(raw, fit_idx, held_idx)
            stats = {"class_weight": class_weights(y_class[fit_idx])}; seed_all(seed * 100 + fold)
            model = CellScaleTeacherStudent(state.shape[2], state.shape[1], fit_bio.shape[1], student_cfg).to(device)
            _, history = fit_epochs(model,
                make_loader(fit_state, counts[fit_idx], fit_bio, y_class[fit_idx], teacher[fit_idx], int(student_cfg["batch_size"]), True, seed),
                student_cfg, stats, device, epochs=selected)
            for row in history: histories.append({"mode": mode, "seed": seed, "fold": fold, "phase": "outer_train", **row})
            held_loader = make_loader(held_state, counts[held_idx], held_bio, y_class[held_idx], teacher[held_idx], int(student_cfg["batch_size"]), False, seed)
            student_probability, cell_probability, donor_embedding, prototype = infer(model, held_loader, device)
            predictions["student"][seed_index, held_local] = student_probability; predictions["cell_only"][seed_index, held_local] = cell_probability
            flat_fit, flat_held = fit_state.reshape(len(fit_idx), -1), held_state.reshape(len(held_idx), -1)
            logistic = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, solver="lbfgs"))
            logistic.fit(flat_fit, y_class[fit_idx]); predictions["summary_logistic"][seed_index, held_local] = logistic.predict_proba(flat_held)
            if seed_index == 0:
                representative["donor_embedding"][held_local] = donor_embedding; representative["prototype"][held_local] = prototype
                representative["fold"][held_local] = fold
            row = {"mode": mode, "seed": seed, "fold": fold, "selected_epoch": selected}
            for name, probability in (("student", student_probability), ("cell_only", cell_probability)):
                for metric, value in classification_metrics(y_class[held_idx], probability).items(): row[f"{name}_{metric}"] = value
            fold_rows.append(row); print(f"mode={mode} seed={seed} fold={fold}/{folds} epoch={selected}", flush=True)

    rows, seed_rows, confusion = [], [], []
    for model_index, (name, values) in enumerate(predictions.items()):
        ensemble = normalize_probabilities(values.mean(0)); family = "teacher_student" if name in ("student", "cell_only") else "non_teacher_student"
        row = {"result_family": family, "model": name, **classification_metrics(y_class[dev], ensemble),
               **bootstrap_intervals(y_class[dev], ensemble, int(cfg["seed"]) + model_index, int(student_cfg["bootstrap_repeats"]))}
        row["balanced_accuracy_permutation_p"] = permutation_pvalue(
            y_class[dev], ensemble, int(cfg["seed"]) + 100 + model_index, int(student_cfg["permutation_repeats"]))
        per_seed_f1, per_seed_balanced = [], []
        for seed_index, seed in enumerate(seeds):
            metric = classification_metrics(y_class[dev], values[seed_index]); per_seed_f1.append(metric["macro_f1"]); per_seed_balanced.append(metric["balanced_accuracy"])
            if name != "teacher": seed_rows.append({"result_family": family, "model": name, "seed": seed, **metric})
            confusion.extend(confusion_rows(y_class[dev], values[seed_index], name, seed))
        row.update(seed_macro_f1_mean=float(np.mean(per_seed_f1)), seed_macro_f1_std=float(np.std(per_seed_f1)),
                   seed_macro_f1_min=float(np.min(per_seed_f1)), seed_macro_f1_max=float(np.max(per_seed_f1)),
                   seed_balanced_accuracy_mean=float(np.mean(per_seed_balanced)), seed_balanced_accuracy_std=float(np.std(per_seed_balanced)))
        rows.append(row); confusion.extend(confusion_rows(y_class[dev], ensemble, name))
        target = output if family == "teacher_student" else baseline_output
        prediction_frame(donors[dev], braak[dev], y_class[dev], ensemble, values, seeds).to_csv(target / f"{name}_predictions.csv", index=False)
    result = pd.DataFrame(rows)
    result[result.result_family == "teacher_student"].to_csv(output / "oof_metrics.csv", index=False)
    result[result.result_family == "non_teacher_student"].to_csv(baseline_output / "oof_metrics.csv", index=False)
    seed_result = pd.DataFrame(seed_rows)
    seed_result[seed_result.result_family == "teacher_student"].to_csv(output / "seed_metrics.csv", index=False)
    seed_result[seed_result.result_family == "non_teacher_student"].to_csv(baseline_output / "seed_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(output / "confusion_matrices.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False); pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    np.savez_compressed(output / f"crossfit_representations_seed{seeds[0]}.npz", donor_names=donors[dev].astype("U"),
                        observed_braak=braak[dev], observed_class=y_class[dev], cell_type_names=z["cell_type_names"], **representative)
    manifest = {"task": "braak_3class", "class_ranges": [[0, 2], [3, 4], [5, 6]], "mode": mode,
                "teacher_student_results": ["student", "cell_only"], "non_teacher_student_results": ["teacher", "summary_logistic"],
                "sampled_cells_per_donor_type": int(cfg["data"]["sampled_cells_per_donor_type"]) if mode == "sampled" else None,
                "all_cells_used_for_distribution_summary": mode == "all", "outer_folds": folds, "seeds": seeds,
                "representative_embedding_seed": seeds[0], "sealed_test_used": False, "teacher_target_is_inner_oof_class_probability": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("sampled", "all", "both"))
    parser.add_argument("--config", default=str(DEFAULT)); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); config = load(args.config)
    for selected_mode in (("sampled", "all") if args.mode == "both" else (args.mode,)): run_mode(config, selected_mode, args.force)
