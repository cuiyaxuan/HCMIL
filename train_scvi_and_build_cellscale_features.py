#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def load(path):
    cfg = yaml.safe_load(Path(path).read_text()); os.chdir(ROOT); return cfg


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def group_sample(groups, cap, seed, allowed=None):
    rng = np.random.default_rng(seed); base = np.arange(len(groups)) if allowed is None else np.flatnonzero(allowed)
    order = base[np.argsort(groups[base])]; sorted_group = groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]; stops = np.r_[starts[1:], len(order)]; chosen = []
    for start, stop in zip(starts, stops):
        idx = order[start:stop]; chosen.append(idx if cap <= 0 or len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(chosen))


def choose_hvgs(cells, donor_codes, development, n_hvg):
    import anndata as ad
    import scanpy as sc
    rows = np.flatnonzero(np.isin(donor_codes, development)); value = ad.AnnData(X=cells[rows].copy())
    sc.pp.normalize_total(value, target_sum=1e4); sc.pp.log1p(value)
    sc.pp.highly_variable_genes(value, n_top_genes=n_hvg, flavor="seurat", inplace=True)
    if "highly_variable_rank" in value.var:
        order = np.argsort(value.var["highly_variable_rank"].fillna(np.inf).to_numpy())
    else:
        order = np.argsort(-value.var["dispersions_norm"].fillna(-np.inf).to_numpy())
    return order[:n_hvg].astype(np.int64)


def summarize(latent, groups, n_donors, n_types, rows):
    dim, ng = latent.shape[1], n_donors * n_types
    state = np.zeros((ng, dim * 5), np.float32); used = np.zeros(ng, np.int64)
    order = rows[np.argsort(groups[rows])]; sorted_group = groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]; stops = np.r_[starts[1:], len(order)]
    for number, (start, stop) in enumerate(zip(starts, stops), 1):
        idx = order[start:stop]; value = latent[idx].astype(np.float32, copy=False); group = int(sorted_group[start])
        state[group] = np.concatenate((value.mean(0), value.std(0), np.quantile(value, .25, axis=0),
                                       np.quantile(value, .5, axis=0), np.quantile(value, .75, axis=0))).astype(np.float32)
        used[group] = len(idx)
        if number % 500 == 0: print(f"Summarized groups {number}/{len(starts)}", flush=True)
    return state.reshape(n_donors, n_types, -1), used.reshape(n_donors, n_types)


def prepare(cfg, force=False):
    import anndata as ad
    import scvi
    d, scfg, seed = cfg["data"], cfg["scvi"], int(cfg["seed"])
    output = Path(d["scvi_cache"]); complete = output / "features_all_cells.npz"; model_path = output / "scvi_reference_model"
    if complete.is_file() and not force:
        print(f"Completed standalone scVI cache exists: {output}"); return
    if output.exists():
        if not force: raise FileExistsError(f"Incomplete cache exists: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True); seed_all(seed); scvi.settings.seed = seed

    linear = Path(d["linear_cache"]); metadata = Path(d["metadata_cache"])
    meta = np.load(linear / "metadata.npz", allow_pickle=False); donors, types = meta["donor_names"].astype(str), meta["cell_type_names"].astype(str)
    split, y = meta["split"].astype(str), meta["braak"].astype(np.float32); development = np.flatnonzero(split == "train")
    sealed = np.flatnonzero(split != "train"); expected = (int(d["expected_development_donors"]), int(d["expected_test_donors"]))
    if (len(development), len(sealed)) != expected: raise ValueError(f"Expected development/test={expected}, found {len(development)}/{len(sealed)}")
    balanced = sparse.load_npz(linear / "balanced_cells.npz")
    hvg = choose_hvgs(balanced, meta["cell_donor"], development, int(scfg["n_hvg"])); np.save(output / "hvg_indices.npy", hvg)

    backed = ad.read_h5ad(d["h5ad_path"], backed="r"); rows = np.load(metadata / "eligible_h5ad_rows.npy")
    dc, tc = np.load(metadata / "cell_donor_codes.npy"), np.load(metadata / "cell_type_codes.npy")
    donor_cell_counts = np.bincount(dc, minlength=len(donors)); zero_cell = np.flatnonzero(donor_cell_counts == 0)
    if len(zero_cell): raise ValueError(f"Found {len(zero_cell)} donors with zero eligible cells: {donors[zero_cell].tolist()}")
    source = backed[rows, hvg]; layer = d.get("counts_layer"); layer = None if layer in (None, "", "null") else str(layer)
    matrix = source.X if layer is None else source.layers[layer]
    matrix = matrix.to_memory() if hasattr(matrix, "to_memory") else matrix.copy(); obs, var = source.obs.copy(), source.var.copy(); backed.file.close()
    matrix = sparse.csr_matrix(matrix); sample = matrix[:min(2048, matrix.shape[0])].data
    if len(sample) and (sample.min() < 0 or not np.allclose(sample, np.rint(sample))): raise ValueError("scVI requires raw integer counts")
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    groups = dc.astype(np.int64) * len(types) + tc; training_rows = group_sample(groups, int(d["scvi_train_cells_per_donor_type"]), seed + 101,
                                                                                 allowed=np.isin(dc, development))
    np.save(output / "scvi_training_h5ad_rows.npy", rows[training_rows])
    train = adata[training_rows].copy(); batch_key = d.get("batch_key"); batch_key = None if batch_key in (None, "", "null") else str(batch_key)
    scvi.model.SCVI.setup_anndata(train, batch_key=batch_key)
    model = scvi.model.SCVI(train, n_hidden=int(scfg["n_hidden"]), n_latent=int(scfg["n_latent"]), n_layers=int(scfg["n_layers"]),
                            dropout_rate=float(scfg["dropout_rate"]), gene_likelihood=str(scfg["gene_likelihood"]), dispersion=str(scfg["dispersion"]))
    model.train(max_epochs=int(scfg["max_epochs"]), batch_size=int(scfg["batch_size"]), early_stopping=bool(scfg["early_stopping"]),
                early_stopping_patience=int(scfg["early_stopping_patience"]), accelerator=str(scfg["accelerator"]), devices=scfg["devices"])
    model.save(model_path, overwrite=True, save_anndata=False)
    scvi.model.SCVI.prepare_query_anndata(adata, model); query = scvi.model.SCVI.load_query_data(adata, model); query.is_trained = True
    latent = query.get_latent_representation(adata=adata, batch_size=int(d["query_batch_size"]), give_mean=True).astype(np.float32)
    np.save(output / "all_cell_latents.npy", latent.astype(np.float16)); np.save(output / "cell_donor_codes.npy", dc); np.save(output / "cell_type_codes.npy", tc)

    sampled_rows = group_sample(groups, int(d["sampled_cells_per_donor_type"]), seed + 202)
    np.save(output / "sampled_h5ad_rows.npy", rows[sampled_rows])
    np.save(output / "sampled_cell_donor_codes.npy", dc[sampled_rows])
    np.save(output / "sampled_cell_type_codes.npy", tc[sampled_rows])
    sampled_state, sampled_used = summarize(latent, groups, len(donors), len(types), sampled_rows)
    all_state, all_used = summarize(latent, groups, len(donors), len(types), np.arange(len(groups)))
    pathway = np.load(Path(d["pathway_cache"]) / "features.npz", allow_pickle=False)
    common = dict(braak=y, braak_class=meta["braak_class"].astype(np.int64), split=split.astype("U"),
                  donor_names=donors.astype("U"), cell_type_names=types.astype("U"),
                  biological_values=pathway["clinical_values"], biological_names=pathway["clinical_names"], biological_kinds=pathway["clinical_kinds"],
                  exact_cell_counts=all_used)
    np.savez_compressed(output / "features_sampled_cells.npz", state=sampled_state, representation_cell_counts=sampled_used, **common)
    np.savez_compressed(output / "features_all_cells.npz", state=all_state, representation_cell_counts=all_used, **common)
    manifest = {"standalone_scvi_trained_from_scratch": True, "n_encoded_cells": len(latent), "n_scvi_training_cells": len(training_rows),
                "n_hvg": len(hvg), "n_latent": latent.shape[1], "sampled_cap_per_donor_type": int(d["sampled_cells_per_donor_type"]),
                "all_cell_features_use_every_encoded_cell": True, "test_braak_used": False, "scvi_training_donors": int(len(development)),
                "summary_order": cfg["student"]["summary_order"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default=str(DEFAULT)); p.add_argument("--force", action="store_true")
    a = p.parse_args(); prepare(load(a.config), a.force)
