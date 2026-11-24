#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.model_selection import train_test_split

from hcmil_stack.classification import braak_to_class

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def load(path):
    cfg = yaml.safe_load(Path(path).read_text()); os.chdir(ROOT); return cfg


def first_by_donor(obs, donor_key, columns, donors):
    frame = pd.DataFrame({donor_key: obs[donor_key].astype(str)})
    for name in columns:
        if name in obs: frame[name] = obs[name].astype(str)
    return frame.groupby(donor_key, sort=False).first().reindex(donors)


def split_donors(obs, cfg, eligible_donors):
    d, seed = cfg["data"], int(cfg["seed"])
    dk, bk = d["donor_key"], d["braak_key"]
    frame = first_by_donor(obs, dk, [bk], sorted(obs[dk].dropna().astype(str).unique()))
    y = pd.to_numeric(frame[bk], errors="coerce"); frame = frame[y.notna()].copy(); y = y[y.notna()].astype(float)
    eligible_donors = set(map(str, eligible_donors))
    frame = frame[frame.index.astype(str).isin(eligible_donors)].copy()
    y = y.reindex(frame.index)
    donors = frame.index.astype(str).to_numpy(); stages = braak_to_class(y.to_numpy())
    dev, test = train_test_split(np.arange(len(donors)), train_size=float(d["development_fraction"]),
                                 random_state=seed, stratify=stages)
    split = np.full(len(donors), "test", dtype="U5"); split[dev] = "train"
    return donors, y.to_numpy(np.float32), split


def parse_gmt(paths):
    sets = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip().split("\t")
                if len(fields) >= 3: sets.append((fields[0], fields[1], set(fields[2:])))
    return sets


def choose_symbols(var_names, var, gene_sets):
    universe = set().union(*(x[2] for x in gene_sets)); candidates = {"var_names": np.asarray(var_names).astype(str)}
    for key in ("gene_name", "gene_names", "gene_symbol", "gene_symbols", "feature_name", "features", "symbol"):
        if key in var: candidates[key] = var[key].astype(str).to_numpy()
    name, values = max(candidates.items(), key=lambda x: sum(v in universe for v in x[1]))
    return values, name


def membership(symbols, sets, minimum, maximum):
    lookup = {}
    for i, value in enumerate(symbols):
        if value and value != "nan" and value not in lookup: lookup[value] = i
    rows, cols, names, sizes = [], [], [], []
    for name, _, genes in sets:
        idx = sorted({lookup[x] for x in genes if x in lookup})
        if minimum <= len(idx) <= maximum:
            rows.extend(idx); cols.extend([len(names)] * len(idx)); names.append(name); sizes.append(len(idx))
    if len(names) < 100: raise ValueError(f"Only {len(names)} pathways matched")
    values = np.concatenate([np.full(n, 1.0 / n, np.float32) for n in sizes])
    return sparse.csc_matrix((values, (rows, cols)), shape=(len(symbols), len(names))), np.asarray(names, dtype="U"), sizes


def logcpm_pathway(pooled, member):
    pooled = pooled.tocsr().astype(np.float32); library = np.asarray(pooled.sum(1)).ravel().astype(np.float32)
    row = np.repeat(np.arange(len(library)), np.diff(pooled.indptr)); scale = np.divide(1e6, library, out=np.zeros_like(library), where=library > 0)
    pooled.data = np.log1p(pooled.data * scale[row]); return (pooled @ member).toarray().astype(np.float32)


def prepare(cfg, force=False):
    import anndata as ad
    d, seed = cfg["data"], int(cfg["seed"])
    metadata_dir, linear_dir, pathway_dir = map(Path, (d["metadata_cache"], d["linear_cache"], d["pathway_cache"]))
    complete = pathway_dir / "features.npz"
    if complete.is_file() and not force:
        print(f"Completed standalone raw-data caches exist: {complete}"); return
    for path in (metadata_dir, linear_dir, pathway_dir):
        if path.exists():
            if not force: raise FileExistsError(f"Incomplete cache exists: {path}; use --force")
            shutil.rmtree(path)
        path.mkdir(parents=True)

    backed = ad.read_h5ad(d["h5ad_path"], backed="r"); dk, tk = d["donor_key"], d["cell_type_key"]
    for key in (dk, tk, d["braak_key"]):
        if key not in backed.obs: raise KeyError(f"Required obs field missing: {key}")
    donor_text, type_text = backed.obs[dk].astype("string"), backed.obs[tk].astype("string")
    label_table = first_by_donor(backed.obs, dk, [d["braak_key"]], sorted(donor_text.dropna().astype(str).unique()))
    valid_label = pd.to_numeric(label_table[d["braak_key"]], errors="coerce").notna()
    labeled_donors = set(label_table.index[valid_label].astype(str))
    type_ok = type_text.notna()
    labeled_cell = donor_text.notna() & type_ok & donor_text.astype(str).isin(labeled_donors)
    raw_types = type_text[labeled_cell].astype(str).value_counts()
    type_names = np.sort(raw_types[raw_types >= int(d["minimum_cells_per_type"])].index.to_numpy(str))
    eligible_cell = labeled_cell & type_text.astype(str).isin(set(type_names))
    eligible_donors = set(donor_text[eligible_cell].astype(str))
    excluded = sorted(labeled_donors - eligible_donors)
    donors, y_true, split = split_donors(backed.obs, cfg, eligible_donors)
    donor_ok = donor_text.notna() & donor_text.astype(str).isin(set(donors))
    keep = donor_ok & type_ok & type_text.astype(str).isin(set(type_names)); rows = np.flatnonzero(keep.to_numpy())

    expected_total = d.get("expected_eligible_donors")
    expected_dev = d.get("expected_development_donors")
    expected_test = d.get("expected_test_donors")
    observed = (len(donors), int(np.sum(split == "train")), int(np.sum(split != "train")))
    expected = tuple(None if x is None else int(x) for x in (expected_total, expected_dev, expected_test))
    if any(x is not None for x in expected) and any(e is not None and e != o for e, o in zip(expected, observed)):
        raise ValueError(f"Expected eligible/development/test donors={expected}, found {observed}")
    raw_y = pd.to_numeric(label_table.loc[valid_label, d["braak_key"]], errors="coerce").to_numpy(float)
    raw_class_counts = np.bincount(braak_to_class(raw_y), minlength=3)
    eligible_class = braak_to_class(y_true); eligible_class_counts = np.bincount(eligible_class, minlength=3)
    expected_raw = np.asarray(cfg["task"]["raw_labeled_class_counts"], dtype=int)
    expected_eligible = np.asarray(cfg["task"]["eligible_class_counts"], dtype=int)
    if not np.array_equal(raw_class_counts, expected_raw):
        raise ValueError(f"Expected raw Braak class counts={expected_raw.tolist()}, found {raw_class_counts.tolist()}")
    if not np.array_equal(eligible_class_counts, expected_eligible):
        raise ValueError(f"Expected eligible Braak class counts={expected_eligible.tolist()}, found {eligible_class_counts.tolist()}")
    print(f"Eligible donors={observed[0]} development={observed[1]} sealed_test={observed[2]}", flush=True)
    print(f"Raw Braak classes 0-2/3-4/5-6: {raw_class_counts.tolist()}; eligible: {eligible_class_counts.tolist()}", flush=True)
    if excluded:
        print(f"Excluded {len(excluded)} Braak-labeled donors with zero eligible cells: {excluded}", flush=True)
    view = backed[rows, :]; layer = d.get("counts_layer"); layer = None if layer in (None, "", "null") else str(layer)
    if layer is not None and layer not in backed.layers: raise KeyError(f"counts_layer={layer!r} not found; layers={list(backed.layers)}")
    matrix = view.X if layer is None else view.layers[layer]
    matrix = matrix.to_memory() if hasattr(matrix, "to_memory") else matrix.copy(); matrix = sparse.csr_matrix(matrix)
    obs, var, gene_names = view.obs.copy(), view.var.copy(), view.var_names.astype(str).to_numpy(); backed.file.close()
    sample = matrix[:min(2048, matrix.shape[0])].data
    if len(sample) and (sample.min() < 0 or not np.allclose(sample, np.rint(sample))): raise ValueError("Configured matrix is not raw integer counts")

    dm, tm = {x: i for i, x in enumerate(donors)}, {x: i for i, x in enumerate(type_names)}
    dc = np.fromiter((dm[x] for x in obs[dk].astype(str)), np.int32, count=len(obs)); tc = np.fromiter((tm[x] for x in obs[tk].astype(str)), np.int16, count=len(obs))
    group, ng = dc.astype(np.int64) * len(type_names) + tc, len(donors) * len(type_names)
    selector = sparse.coo_matrix((np.ones(len(group), np.float32), (group, np.arange(len(group)))), shape=(ng, len(group))).tocsr()
    donor_selector = sparse.coo_matrix((np.ones(len(group), np.float32), (dc, np.arange(len(group)))), shape=(len(donors), len(group))).tocsr()
    print(f"Aggregating all {len(group):,} eligible cells x {matrix.shape[1]:,} genes", flush=True)
    by_type, by_donor = (selector @ matrix).tocsr(), (donor_selector @ matrix).tocsr()
    sparse.save_npz(linear_dir / "donor_type_counts.npz", by_type); sparse.save_npz(linear_dir / "donor_counts.npz", by_donor)

    rng, cap = np.random.default_rng(seed + 1), int(d["sampled_cells_per_donor_type"])
    order = np.argsort(group); sorted_group = group[order]; starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]; stops = np.r_[starts[1:], len(order)]
    chosen = []
    for start, stop in zip(starts, stops):
        idx = order[start:stop]; chosen.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    chosen = np.sort(np.concatenate(chosen)); sparse.save_npz(linear_dir / "balanced_cells.npz", matrix[chosen])
    y_sealed = y_true.copy(); y_sealed[split != "train"] = np.nan
    class_sealed = eligible_class.copy(); class_sealed[split != "train"] = -1
    np.savez_compressed(linear_dir / "metadata.npz", cell_donor=dc[chosen], cell_type=tc[chosen], donor_names=donors.astype("U"),
                        cell_type_names=type_names.astype("U"), gene_names=gene_names.astype("U"), braak=y_sealed,
                        braak_class=class_sealed, split=split.astype("U"))

    pathway_cfg = cfg["pathways"]; gene_sets = parse_gmt(pathway_cfg["files"])
    symbols, symbol_source = choose_symbols(gene_names, var, gene_sets)
    member, pathway_names, pathway_sizes = membership(symbols, gene_sets, int(pathway_cfg["minimum_genes"]), int(pathway_cfg["maximum_genes"]))
    pathway = logcpm_pathway(by_type, member).reshape(len(donors), len(type_names), -1)
    counts = np.bincount(group, minlength=ng).reshape(len(donors), len(type_names)).astype(np.int64)
    clinical_cfg = cfg["clinical"]; clinical_columns = clinical_cfg["biological_numeric"] + clinical_cfg["biological_categorical"] + clinical_cfg["technical_numeric"] + clinical_cfg["technical_categorical"]
    clinical = first_by_donor(obs, dk, clinical_columns, donors); available = [x for x in clinical_columns if x in clinical]
    kinds = np.asarray(["numeric" if x in clinical_cfg["biological_numeric"] + clinical_cfg["technical_numeric"] else "categorical" for x in available], dtype="U")
    np.savez_compressed(pathway_dir / "features.npz", pathway=pathway, counts=counts, braak=y_sealed, braak_class=class_sealed, split=split.astype("U"),
                        donor_names=donors.astype("U"), cell_type_names=type_names.astype("U"), pathway_names=pathway_names,
                        clinical_values=clinical[available].fillna("NA").astype(str).to_numpy(dtype="U"), clinical_names=np.asarray(available, dtype="U"), clinical_kinds=kinds)
    pd.DataFrame({"donor_id": donors, "split": split, "braak_development_only": y_sealed}).to_csv(metadata_dir / "patients.csv", index=False)
    np.savez_compressed(metadata_dir / "sealed_test_labels.npz", donor_names=donors[split != "train"].astype("U"),
                        braak=y_true[split != "train"], braak_class=eligible_class[split != "train"])
    np.save(metadata_dir / "eligible_h5ad_rows.npy", rows); np.save(metadata_dir / "cell_donor_codes.npy", dc); np.save(metadata_dir / "cell_type_codes.npy", tc)
    manifest = {"standalone": True, "source_h5ad": d["h5ad_path"], "n_cells": len(rows), "n_donors": len(donors),
                "development_donors": int(np.sum(split == "train")), "sealed_test_donors": int(np.sum(split != "train")),
                "donor_selection": "nonmissing Braak and at least one cell in the retained cell types before splitting",
                "excluded_zero_eligible_cell_donors": excluded,
                "braak_3class_ranges": cfg["task"]["class_ranges"], "raw_class_counts": raw_class_counts.tolist(),
                "eligible_class_counts": eligible_class_counts.tolist(),
                "n_cell_types": len(type_names), "cell_type_names": type_names.tolist(), "n_genes": matrix.shape[1],
                "n_pathways": len(pathway_names), "pathway_symbol_source": symbol_source, "sample_cap": cap,
                "test_labels_removed_from_training_caches": True}
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)); (linear_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (pathway_dir / "manifest.json").write_text(json.dumps({**manifest, "pathway_sizes": {"median": float(np.median(pathway_sizes))}}, indent=2))
    print(json.dumps(manifest, indent=2)); print("Wrote standalone raw-data caches", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default=str(DEFAULT)); p.add_argument("--force", action="store_true")
    a = p.parse_args(); prepare(load(a.config), a.force)
