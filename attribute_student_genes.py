#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import sparse

from celltype_followup_utils import expected_braak
from hcmil_stack.classification import normalize_probabilities
from student_model import CellScaleTeacherStudent
from train_students import BiologicalEncoder


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def make_anndata(backed, rows, genes, layer):
    import anndata as ad
    view = backed[rows, genes]
    matrix = view.X if layer is None else view.layers[layer]
    matrix = matrix.to_memory() if hasattr(matrix, "to_memory") else matrix.copy()
    return ad.AnnData(X=sparse.csr_matrix(matrix), obs=view.obs.copy(), var=view.var.copy())


def summarize_latent(latent, donor_codes, donor_indices):
    dim = latent.shape[1]
    output = np.zeros((len(donor_indices), dim * 5), np.float32)
    counts = np.zeros(len(donor_indices), np.int64)
    for local, donor in enumerate(donor_indices):
        values = latent[donor_codes == donor]
        counts[local] = len(values)
        if len(values):
            output[local] = np.concatenate((values.mean(0), values.std(0),
                                            np.quantile(values, .25, axis=0),
                                            np.quantile(values, .5, axis=0),
                                            np.quantile(values, .75, axis=0)))
    return output[:, None, :], counts[:, None]


@torch.no_grad()
def student_scores(model, state, counts, bio, batch_size, device):
    values, cell_values = [], []
    model.eval()
    for start in range(0, len(state), batch_size):
        stop = min(len(state), start + batch_size)
        s = torch.from_numpy(state[start:stop]).to(device)
        c = torch.from_numpy(counts[start:stop].astype(np.float32)).to(device)
        b = torch.from_numpy(bio[start:stop]).to(device)
        logits, cell_logits, _, _ = model(s, c, b)
        values.append(torch.softmax(logits, 1).cpu().numpy())
        cell_values.append(torch.softmax(cell_logits, 1).cpu().numpy())
    probability = normalize_probabilities(np.concatenate(values))
    cell_probability = normalize_probabilities(np.concatenate(cell_values))
    return expected_braak(probability), probability, expected_braak(cell_probability), cell_probability


def donor_removed_expression(counts, donor_codes, donor_indices, column):
    value = counts[:, column]
    return np.asarray([np.log1p(value[donor_codes == donor].sum()) for donor in donor_indices], dtype=np.float32)


def main(config, task_id, force=False):
    cfg = yaml.safe_load(Path(config).read_text())
    os.chdir(ROOT)
    acfg, dcfg = cfg["attribution"], cfg["data"]
    seeds = list(map(int, acfg["student_seeds"]))
    folds = int(acfg["outer_folds"])
    n_genes = int(cfg["scvi"]["n_hvg"])
    chunk_size = int(acfg["genes_per_task"])
    chunks = int(np.ceil(n_genes / chunk_size))
    total = len(seeds) * folds * chunks
    if not 0 <= task_id < total:
        raise ValueError(f"task-id must be in [0,{total - 1}], received {task_id}")
    model_number, chunk = divmod(task_id, chunks)
    seed_index, fold_zero = divmod(model_number, folds)
    seed, fold = seeds[seed_index], fold_zero + 1
    output = Path(acfg["attribution_dir"]) / f"seed_{seed}" / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    summary_file = output / f"gene_summary_chunk_{chunk:03d}.csv.gz"
    donor_file = output / f"donor_effects_chunk_{chunk:03d}.csv.gz"
    manifest_file = output / f"manifest_chunk_{chunk:03d}.json"
    if summary_file.is_file() and donor_file.is_file() and manifest_file.is_file() and not force:
        try:
            existing = json.loads(manifest_file.read_text())
            if (existing.get("task_id") == task_id and existing.get("seed") == seed and
                    existing.get("fold") == fold and existing.get("chunk") == chunk and
                    summary_file.stat().st_size > 0 and donor_file.stat().st_size > 0):
                print(f"Complete attribution chunk already exists: {summary_file}")
                return
        except Exception:
            pass
    for partial in (summary_file, donor_file, manifest_file,
                    Path(str(summary_file) + ".tmp"), Path(str(donor_file) + ".tmp"),
                    Path(str(manifest_file) + ".tmp")):
        if partial.exists():
            partial.unlink()

    checkpoint_file = Path(acfg["model_dir"]) / f"seed_{seed}" / f"fold_{fold}.pt"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    cache_dir = Path(dcfg["scvi_cache"])
    feature = np.load(cache_dir / "features_sampled_cells.npz", allow_pickle=False)
    donors = feature["donor_names"].astype(str)
    held = np.asarray(checkpoint["held_indices"], dtype=int)
    fit = np.asarray(checkpoint["fit_indices"], dtype=int)
    if donors[held].tolist() != list(checkpoint["held_donor_ids"]):
        raise ValueError("Checkpoint donor order does not match the feature cache")
    types = feature["cell_type_names"].astype(str)
    ast_index = int(checkpoint["ast_type_index"])
    if types[ast_index] != str(acfg["cell_type"]):
        raise ValueError("Ast cell-type index changed")

    biological = BiologicalEncoder(feature["biological_names"], feature["biological_kinds"], cfg["clinical"])
    held_bio = biological.fit_transform(feature["biological_values"], fit, held)
    raw_bio = pd.DataFrame(feature["biological_values"].astype(str), columns=feature["biological_names"].astype(str))
    domain_names = [name for name in acfg["domains"] if name in raw_bio]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellScaleTeacherStudent(int(checkpoint["input_dim"]), 1,
                                    int(checkpoint["biological_dim"]), cfg["student"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["state_scale"], dtype=np.float32)

    hvg = np.load(cache_dir / "hvg_indices.npy").astype(int)
    if len(hvg) != n_genes:
        raise ValueError(f"Expected {n_genes} scVI HVGs, found {len(hvg)}")
    sampled_rows = np.load(cache_dir / "sampled_h5ad_rows.npy").astype(int)
    sampled_donors = np.load(cache_dir / "sampled_cell_donor_codes.npy").astype(int)
    sampled_types = np.load(cache_dir / "sampled_cell_type_codes.npy").astype(int)
    chosen = (sampled_types == ast_index) & np.isin(sampled_donors, held)
    query_rows = sampled_rows[chosen]
    query_donors = sampled_donors[chosen]
    if not len(query_rows):
        raise ValueError("No sampled Ast cells for held donors")

    import anndata as ad
    import scvi
    backed = ad.read_h5ad(dcfg["h5ad_path"], backed="r")
    layer = dcfg.get("counts_layer")
    layer = None if layer in (None, "", "null") else str(layer)
    query_adata = make_anndata(backed, query_rows, hvg, layer)
    backed.file.close()
    gene_var_names = query_adata.var_names.astype(str).to_numpy()
    symbol_key = str(acfg.get("gene_symbol_key", "gene_names"))
    gene_symbols = (query_adata.var[symbol_key].astype(str).to_numpy()
                    if symbol_key in query_adata.var else gene_var_names.copy())
    # Loading the frozen reference weights directly onto held-donor AnnData is
    # inference-only and avoids materializing the large training AnnData.  Some
    # older scvi-tools releases require the reference AnnData; retain a strict
    # fallback that reconstructs it and performs no query training.
    try:
        query = scvi.model.SCVI.load(cache_dir / "scvi_reference_model", adata=query_adata)
        scvi_loading_path = "direct_frozen_reference_on_held_anndata"
    except Exception as direct_error:
        print(f"Direct frozen-model load failed; using reference-AnnData fallback: {direct_error}", flush=True)
        backed = ad.read_h5ad(dcfg["h5ad_path"], backed="r")
        training_rows = np.load(cache_dir / "scvi_training_h5ad_rows.npy").astype(int)
        reference_adata = make_anndata(backed, training_rows, hvg, layer)
        backed.file.close()
        reference = scvi.model.SCVI.load(cache_dir / "scvi_reference_model", adata=reference_adata)
        scvi.model.SCVI.prepare_query_anndata(query_adata, reference)
        query = scvi.model.SCVI.load_query_data(query_adata, reference)
        query.is_trained = True
        del reference, reference_adata
        scvi_loading_path = "frozen_reference_query_fallback_no_training"
    query.to_device(device)
    gc.collect()

    dense = query_adata.X.toarray().astype(np.float32) if sparse.issparse(query_adata.X) else np.asarray(query_adata.X, np.float32)
    query_adata.X = dense
    batch = int(dcfg["query_batch_size"])

    # Baseline is recomputed through the same scVI query encoder used for every
    # knockout, so the only changing quantity is the selected gene column.
    baseline_latent = query.get_latent_representation(adata=query_adata, batch_size=batch, give_mean=True).astype(np.float32)
    baseline_state, held_counts = summarize_latent(baseline_latent, query_donors, held)
    cached_baseline_state = feature["state"].astype(np.float32)[held, ast_index:ast_index + 1]
    present = held_counts[:, 0] > 0
    maximum_state_difference = (float(np.max(np.abs(baseline_state[present] - cached_baseline_state[present])))
                                if np.any(present) else 0.0)
    if maximum_state_difference > float(acfg["maximum_baseline_state_difference"]):
        raise RuntimeError(
            f"Re-encoded baseline differs from the Student cache by {maximum_state_difference:.6g}; "
            "do not interpret knockout effects until the scVI loading path is reconciled"
        )
    baseline_state = (baseline_state - mean[None]) / scale[None]
    baseline_state[held_counts == 0] = 0
    baseline_score, baseline_probability, baseline_cell_score, _ = student_scores(
        model, baseline_state.astype(np.float32), held_counts, held_bio,
        int(cfg["student"]["batch_size"]), device,
    )

    start = chunk * chunk_size
    stop = min(n_genes, start + chunk_size)
    summary_rows, donor_rows = [], []
    for gene_index in range(start, stop):
        original = dense[:, gene_index].copy()
        removed = donor_removed_expression(dense, query_donors, held, gene_index)
        if np.any(original != 0):
            dense[:, gene_index] = 0
            perturbed_latent = query.get_latent_representation(
                adata=query_adata, batch_size=batch, give_mean=True
            ).astype(np.float32)
            dense[:, gene_index] = original
            perturbed_state, perturbed_counts = summarize_latent(perturbed_latent, query_donors, held)
            perturbed_state = (perturbed_state - mean[None]) / scale[None]
            perturbed_state[perturbed_counts == 0] = 0
            perturbed_score, _, perturbed_cell_score, _ = student_scores(
                model, perturbed_state.astype(np.float32), perturbed_counts, held_bio,
                int(cfg["student"]["batch_size"]), device,
            )
            effect = baseline_score - perturbed_score
            cell_effect = baseline_cell_score - perturbed_cell_score
        else:
            effect = np.zeros(len(held), dtype=float)
            cell_effect = np.zeros(len(held), dtype=float)
        expressed = removed > 0
        nonzero_effect = effect[expressed]
        summary_rows.append({
            "seed": seed, "fold": fold, "gene_index": gene_index,
            "var_name": gene_var_names[gene_index], "gene": gene_symbols[gene_index],
            "n_held_donors": len(held), "n_expressed_donors": int(expressed.sum()),
            "mean_knockout_effect_all": float(np.mean(effect)),
            "mean_cell_only_knockout_effect_all": float(np.mean(cell_effect)),
            "median_knockout_effect_all": float(np.median(effect)),
            "mean_knockout_effect_expressed": float(np.mean(nonzero_effect)) if len(nonzero_effect) else 0.0,
            "mean_absolute_effect": float(np.mean(np.abs(effect))),
            "positive_fraction_expressed": float(np.mean(nonzero_effect > 0)) if len(nonzero_effect) else np.nan,
            "mean_removed_log1p_count": float(np.mean(removed)),
            "direction_definition": "positive means gene presence increases Student expected Braak",
        })
        for local, donor_index in enumerate(held):
            row = {
                "seed": seed, "fold": fold, "donor_id": donors[donor_index],
                "gene_index": gene_index, "var_name": gene_var_names[gene_index],
                "gene": gene_symbols[gene_index], "baseline_score": float(baseline_score[local]),
                "knockout_effect": float(effect[local]), "removed_log1p_count": float(removed[local]),
                "cell_only_knockout_effect": float(cell_effect[local]),
                "expressed": bool(expressed[local]),
            }
            for domain in domain_names:
                row[domain] = raw_bio.iloc[donor_index][domain]
            donor_rows.append(row)
        print(f"task={task_id}/{total-1} seed={seed} fold={fold} gene={gene_index+1}/{stop}", flush=True)

    summary_tmp = Path(str(summary_file) + ".tmp")
    donor_tmp = Path(str(donor_file) + ".tmp")
    pd.DataFrame(summary_rows).to_csv(summary_tmp, index=False, compression="gzip")
    pd.DataFrame(donor_rows).to_csv(donor_tmp, index=False, compression="gzip")
    summary_tmp.replace(summary_file)
    donor_tmp.replace(donor_file)
    metadata = {
        "task_id": task_id, "seed": seed, "fold": fold, "chunk": chunk,
        "gene_start_inclusive": start, "gene_stop_exclusive": stop,
        "intervention": "set selected gene raw count to zero in every sampled Ast cell of each held donor",
        "baseline_reencoded": True, "held_donor_attribution": True,
        "maximum_baseline_state_difference": maximum_state_difference,
        "scvi_loading_path": scvi_loading_path,
        "test_labels_used": False,
        "baseline_probability_file_not_required": baseline_probability.shape,
    }
    manifest_tmp = Path(str(manifest_file) + ".tmp")
    manifest_tmp.write_text(json.dumps(metadata, indent=2))
    manifest_tmp.replace(manifest_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.task_id, args.force)
