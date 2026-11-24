#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from celltype_followup_utils import evaluate, expected_braak
from hcmil_stack.classification import normalize_probabilities
from student_model import CellScaleTeacherStudent
from train_students import (BiologicalEncoder, class_weights, fit_epochs, infer,
                            make_loader, normalize_state, seed_all)


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def state_statistics(state, counts, fit):
    present = counts[fit] > 0
    values = state[fit]
    denominator = present.sum((0, 1)).clip(min=1)
    mean = (values * present[..., None]).sum((0, 1)) / denominator
    variance = (((values - mean) ** 2) * present[..., None]).sum((0, 1)) / denominator
    return mean.astype(np.float32), np.sqrt(variance + 1e-5).astype(np.float32)


@torch.no_grad()
def predict(model, state, counts, bio, labels, batch_size, device):
    dummy = np.full((len(state), 3), 1 / 3, np.float32)
    loader = make_loader(state, counts, bio, labels, dummy, batch_size, False, 0)
    probability, _, donor, _ = infer(model, loader, device)
    return normalize_probabilities(probability).astype(np.float32), donor


def main(config, seed_index, force=False):
    cfg = yaml.safe_load(Path(config).read_text())
    os.chdir(ROOT)
    acfg = cfg["attribution"]
    seeds = list(map(int, acfg["student_seeds"]))
    if not 0 <= seed_index < len(seeds):
        raise ValueError(f"seed-index must be in [0,{len(seeds)-1}]")
    seed = seeds[seed_index]
    output = Path(acfg["model_dir"]) / f"seed_{seed}"
    success = output / "_SUCCESS.json"
    if output.exists():
        expected = [output / f"fold_{fold}.pt" for fold in range(1, int(acfg["outer_folds"]) + 1)]
        expected += [output / "oof_metrics.csv", output / "oof_predictions.csv",
                     output / "oof_representations.npz", output / "fold_manifest.csv"]
        if not force and success.is_file() and all(path.is_file() and path.stat().st_size > 0 for path in expected):
            print(f"Completed Student seed exists: {output}")
            return
        print(f"Removing incomplete Student seed directory before retry: {output}", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    cache = np.load(Path(cfg["data"]["scvi_cache"]) / "features_sampled_cells.npz", allow_pickle=False)
    state = cache["state"].astype(np.float32)
    counts = cache["exact_cell_counts"].astype(np.int64)
    braak = cache["braak"].astype(np.float32)
    labels = cache["braak_class"].astype(np.int64)
    split = cache["split"].astype(str)
    donors = cache["donor_names"].astype(str)
    types = cache["cell_type_names"].astype(str)
    matches = np.flatnonzero(types == str(acfg["cell_type"]))
    if len(matches) != 1:
        raise ValueError(f"Expected one {acfg['cell_type']} type, found {matches.tolist()} in {types.tolist()}")
    ast = int(matches[0])
    state = state[:, ast:ast + 1]
    counts = counts[:, ast:ast + 1]
    dev = np.flatnonzero((split == "train") & (labels >= 0))
    if len(dev) != int(cfg["data"]["expected_development_donors"]):
        raise ValueError("Development donor count mismatch")

    teacher_frame = pd.read_csv(
        Path(cfg["teacher"]["output_dir"]) / "predictions" / "teacher_braak3.csv",
        dtype={"donor_id": str},
    ).set_index("donor_id")
    pcols = [f"ensemble_probability_class{x}" for x in range(3)]
    teacher = normalize_probabilities(teacher_frame.loc[donors[dev], pcols].to_numpy(float)).astype(np.float32)
    encoder = BiologicalEncoder(cache["biological_names"], cache["biological_kinds"], cfg["clinical"])
    raw_bio = cache["biological_values"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["student"]["batch_size"])
    folds = int(acfg["outer_folds"])
    probabilities = np.zeros((len(dev), 3), np.float32)
    embeddings = np.zeros((len(dev), int(cfg["student"]["donor_dim"])), np.float32)
    rows = []

    outer = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fold, (fit_local, held_local) in enumerate(outer.split(dev, labels[dev]), 1):
        fit_idx, held_idx = dev[fit_local], dev[held_local]
        inner_fit, inner_val = train_test_split(
            fit_idx, test_size=float(cfg["student"]["validation_fraction"]),
            random_state=seed + fold, stratify=labels[fit_idx],
        )
        s_fit = normalize_state(state, counts, inner_fit, inner_fit)
        s_val = normalize_state(state, counts, inner_fit, inner_val)
        b_fit = encoder.fit_transform(raw_bio, inner_fit, inner_fit)
        b_val = encoder.fit_transform(raw_bio, inner_fit, inner_val)
        seed_all(seed * 100 + fold)
        selector = CellScaleTeacherStudent(state.shape[2], 1, b_fit.shape[1], cfg["student"]).to(device)
        selected_epoch, _ = fit_epochs(
            selector,
            make_loader(s_fit, counts[inner_fit], b_fit, labels[inner_fit],
                        teacher[np.searchsorted(dev, inner_fit)], batch_size, True, seed),
            cfg["student"], {"class_weight": class_weights(labels[inner_fit])}, device,
            make_loader(s_val, counts[inner_val], b_val, labels[inner_val],
                        teacher[np.searchsorted(dev, inner_val)], batch_size, False, seed),
        )
        del selector

        s_fit = normalize_state(state, counts, fit_idx, fit_idx)
        s_held = normalize_state(state, counts, fit_idx, held_idx)
        b_fit = encoder.fit_transform(raw_bio, fit_idx, fit_idx)
        b_held = encoder.fit_transform(raw_bio, fit_idx, held_idx)
        mean, scale = state_statistics(state, counts, fit_idx)
        seed_all(seed * 100 + fold)
        model = CellScaleTeacherStudent(state.shape[2], 1, b_fit.shape[1], cfg["student"]).to(device)
        fit_epochs(
            model,
            make_loader(s_fit, counts[fit_idx], b_fit, labels[fit_idx],
                        teacher[np.searchsorted(dev, fit_idx)], batch_size, True, seed),
            cfg["student"], {"class_weight": class_weights(labels[fit_idx])}, device,
            epochs=selected_epoch,
        )
        probability, donor_embedding = predict(model, s_held, counts[held_idx], b_held,
                                               labels[held_idx], batch_size, device)
        probabilities[held_local] = probability
        embeddings[held_local] = donor_embedding
        checkpoint = {
            "seed": seed, "fold": fold, "selected_epoch": int(selected_epoch),
            "cell_type": str(acfg["cell_type"]), "ast_type_index": ast,
            "input_dim": int(state.shape[2]), "n_types": 1,
            "biological_dim": int(b_fit.shape[1]),
            "fit_indices": fit_idx, "held_indices": held_idx,
            "fit_donor_ids": donors[fit_idx].tolist(), "held_donor_ids": donors[held_idx].tolist(),
            "state_mean": mean, "state_scale": scale,
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        }
        torch.save(checkpoint, output / f"fold_{fold}.pt")
        rows.append({"seed": seed, "fold": fold, "selected_epoch": selected_epoch,
                     "n_fit": len(fit_idx), "n_held": len(held_idx)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Ast Student seed={seed} fold={fold}/{folds} epoch={selected_epoch}", flush=True)

    metrics = {"seed": seed, **evaluate(labels[dev], braak[dev], probabilities)}
    pd.DataFrame([metrics]).to_csv(output / "oof_metrics.csv", index=False)
    frame = pd.DataFrame({"donor_id": donors[dev], "observed_braak": braak[dev],
                          "observed_class": labels[dev], "predicted_braak": expected_braak(probabilities)})
    for k in range(3):
        frame[f"probability_class{k}"] = probabilities[:, k]
    frame.to_csv(output / "oof_predictions.csv", index=False)
    np.savez_compressed(output / "oof_representations.npz", donor_ids=donors[dev].astype("U"),
                        donor_embedding=embeddings)
    pd.DataFrame(rows).to_csv(output / "fold_manifest.csv", index=False)
    manifest = {
        "model": "original nonlinear CellScaleTeacherStudent",
        "cell_type": str(acfg["cell_type"]), "seed": seed, "outer_folds": folds,
        "checkpoint_per_outer_fold": True, "held_donors_never_used_to_fit_their_checkpoint": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    success_tmp = output / "_SUCCESS.json.tmp"
    success_tmp.write_text(json.dumps({"seed": seed, "folds": folds, "complete": True}, indent=2))
    success_tmp.replace(success)
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.seed_index, args.force)
