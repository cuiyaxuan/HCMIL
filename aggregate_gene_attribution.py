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
from scipy import stats

from celltype_followup_utils import evaluate
from hcmil_stack.classification import normalize_probabilities


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def bh_adjust(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def direction_consistency(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return np.nan
    sign = np.sign(np.median(values))
    if sign == 0:
        sign = np.sign(np.mean(values))
    return float(np.mean(np.sign(values) == sign)) if sign != 0 else np.nan


def main(config, force=False):
    cfg = yaml.safe_load(Path(config).read_text())
    os.chdir(ROOT)
    acfg = cfg["attribution"]
    output = Path(acfg["aggregate_dir"])
    if output.exists():
        complete = output / "_SUCCESS.json"
        expected = output / "student_gene_weights_all.csv"
        if not force and complete.is_file() and expected.is_file() and expected.stat().st_size > 0:
            print(f"Completed aggregate exists: {output}")
            return
        print(f"Removing incomplete aggregate before retry: {output}", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    seeds = list(map(int, acfg["student_seeds"]))
    folds = int(acfg["outer_folds"])
    n_genes = int(cfg["scvi"]["n_hvg"])
    chunks = int(np.ceil(n_genes / int(acfg["genes_per_task"])))
    summary_files, donor_files, missing = [], [], []
    for seed in seeds:
        for fold in range(1, folds + 1):
            folder = Path(acfg["attribution_dir"]) / f"seed_{seed}" / f"fold_{fold}"
            for chunk in range(chunks):
                summary = folder / f"gene_summary_chunk_{chunk:03d}.csv.gz"
                donor = folder / f"donor_effects_chunk_{chunk:03d}.csv.gz"
                if not summary.is_file() or not donor.is_file():
                    missing.append(str(summary if not summary.is_file() else donor))
                else:
                    summary_files.append(summary); donor_files.append(donor)
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} attribution outputs. First missing files:\n{preview}")

    model = pd.concat((pd.read_csv(path) for path in summary_files), ignore_index=True)
    model.to_csv(output / "all_model_gene_effects.csv.gz", index=False, compression="gzip")
    model_rows = []
    for (gene_index, var_name, gene), part in model.groupby(["gene_index", "var_name", "gene"], sort=False):
        value = part["mean_knockout_effect_all"].to_numpy(float)
        cell_value = part["mean_cell_only_knockout_effect_all"].to_numpy(float)
        expressed_value = part["mean_knockout_effect_expressed"].to_numpy(float)
        model_rows.append({
            "gene_index": gene_index, "var_name": var_name, "gene": gene,
            "n_models": len(part), "mean_model_effect": float(np.mean(value)),
            "median_model_effect": float(np.median(value)), "sd_model_effect": float(np.std(value)),
            "model_direction_consistency": direction_consistency(value),
            "mean_cell_only_model_effect": float(np.mean(cell_value)),
            "cell_only_model_direction_consistency": direction_consistency(cell_value),
            "mean_model_effect_expressed": float(np.mean(expressed_value)),
            "mean_absolute_model_effect": float(part["mean_absolute_effect"].mean()),
            "mean_expressed_donors_per_model": float(part["n_expressed_donors"].mean()),
        })
    model_stability = pd.DataFrame(model_rows)

    donor = pd.concat((pd.read_csv(path, dtype={"donor_id": str}) for path in donor_files), ignore_index=True)
    identity = ["gene_index", "var_name", "gene", "donor_id"]
    aggregation = {"knockout_effect": "mean", "cell_only_knockout_effect": "mean",
                   "removed_log1p_count": "mean", "expressed": "max"}
    for domain in acfg["domains"]:
        if domain in donor:
            aggregation[domain] = "first"
    donor_ensemble = donor.groupby(identity, as_index=False, sort=False).agg(aggregation)
    donor_ensemble.to_csv(output / "donor_ensemble_gene_effects.csv.gz", index=False, compression="gzip")

    rows = []
    minimum_domain = int(acfg["minimum_domain_donors"])
    for (gene_index, var_name, gene), part in donor_ensemble.groupby(["gene_index", "var_name", "gene"], sort=False):
        effect = part["knockout_effect"].to_numpy(float)
        cell_effect = part["cell_only_knockout_effect"].to_numpy(float)
        expressed = part["expressed"].astype(bool).to_numpy()
        n = len(effect); mean = float(np.mean(effect)); sd = float(np.std(effect, ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(max(n, 1)); tvalue = mean / se if se > 0 else 0.0
        pvalue = float(2 * stats.t.sf(abs(tvalue), max(n - 1, 1))) if se > 0 else 1.0
        domain_values = []
        for domain in acfg["domains"]:
            if domain not in part:
                continue
            for _, group in part.groupby(domain, dropna=False):
                if len(group) >= minimum_domain:
                    domain_values.append(float(group["knockout_effect"].mean()))
        overall_sign = np.sign(mean)
        domain_consistency = (float(np.mean(np.sign(domain_values) == overall_sign))
                              if domain_values and overall_sign != 0 else np.nan)
        expressed_effect = effect[expressed]
        mean_removed = float(part["removed_log1p_count"].mean())
        rows.append({
            "gene_index": gene_index, "var_name": var_name, "gene": gene,
            "n_donors": n, "n_expressed_donors": int(expressed.sum()),
            "expression_prevalence": float(expressed.mean()),
            "student_gene_weight": mean,
            "cell_only_gene_weight": float(np.mean(cell_effect)),
            "student_gene_weight_expressed": float(np.mean(expressed_effect)) if len(expressed_effect) else 0.0,
            "weight_se": se, "weight_t": tvalue, "weight_p": pvalue,
            "positive_donor_fraction_expressed": float(np.mean(expressed_effect > 0)) if len(expressed_effect) else np.nan,
            "mean_removed_log1p_count": mean_removed,
            "weight_per_removed_log1p_count": mean / mean_removed if mean_removed > 0 else 0.0,
            "domain_direction_consistency": domain_consistency,
            "n_tested_domains": len(domain_values),
            "direction": "positive" if mean > 0 else ("negative" if mean < 0 else "zero"),
        })
    result = pd.DataFrame(rows)
    result["weight_fdr"] = bh_adjust(result["weight_p"].to_numpy(float))
    result = result.merge(model_stability, on=["gene_index", "var_name", "gene"], how="left")
    eligible = ((result["n_expressed_donors"] >= int(acfg["minimum_expressed_donors"])) &
                (result["model_direction_consistency"] >= float(acfg["minimum_model_direction_consistency"])))
    result["stable_candidate"] = eligible
    result["absolute_weight_rank"] = result["student_gene_weight"].abs().rank(ascending=False, method="min").astype(int)
    result = result.sort_values(["stable_candidate", "absolute_weight_rank"], ascending=[False, True])
    result.to_csv(output / "student_gene_weights_all.csv", index=False)
    stable = result[result["stable_candidate"]].copy()
    stable.head(int(acfg["top_genes_per_direction"]) * 2).to_csv(output / "top_stable_student_genes.csv", index=False)
    stable[stable.student_gene_weight > 0].nlargest(int(acfg["top_genes_per_direction"]), "student_gene_weight").to_csv(
        output / "top_positive_student_genes.csv", index=False)
    stable[stable.student_gene_weight < 0].nsmallest(int(acfg["top_genes_per_direction"]), "student_gene_weight").to_csv(
        output / "top_negative_student_genes.csv", index=False)
    audit_rows = []
    by_gene = result.drop_duplicates("gene").set_index("gene")
    for gene in map(str, acfg.get("audit_genes", [])):
        if gene in by_gene.index:
            row = by_gene.loc[gene].to_dict()
            audit_rows.append({"requested_gene": gene, "in_student_hvg": True, **row})
        else:
            audit_rows.append({"requested_gene": gene, "in_student_hvg": False})
    pd.DataFrame(audit_rows).to_csv(output / "prior_candidate_gene_audit.csv", index=False)

    # Aggregate the five OOF prediction sets independently of attribution.
    predictions = []
    for seed in seeds:
        frame = pd.read_csv(Path(acfg["model_dir"]) / f"seed_{seed}" / "oof_predictions.csv", dtype={"donor_id": str})
        frame["seed"] = seed
        predictions.append(frame)
    pred = pd.concat(predictions, ignore_index=True)
    probability_columns = [f"probability_class{x}" for x in range(3)]
    ensemble = pred.groupby("donor_id", sort=False)[probability_columns].mean()
    observed = pred.drop_duplicates("donor_id").set_index("donor_id").loc[ensemble.index]
    probability = normalize_probabilities(ensemble.to_numpy(float))
    metrics = evaluate(observed["observed_class"].to_numpy(int), observed["observed_braak"].to_numpy(float), probability)
    pd.DataFrame([{"model": "Ast-only Student OOF probability ensemble", **metrics}]).to_csv(
        output / "student_oof_ensemble_metrics.csv", index=False)

    manifest = {
        "n_genes": len(result), "n_models": len(seeds) * folds,
        "student_seeds": seeds, "outer_folds": folds,
        "primary_weight": "mean over development donors of baseline Student expected Braak minus gene-knockout expected Braak",
        "positive_direction": "gene presence increases predicted Braak severity",
        "negative_direction": "gene presence decreases predicted Braak severity",
        "attribution_is_oof": True, "sealed_test_labels_used": False,
        "complete_expected_files": len(summary_files) + len(donor_files),
        "audit_genes": list(map(str, acfg.get("audit_genes", []))),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    success_tmp = output / "_SUCCESS.json.tmp"
    success_tmp.write_text(json.dumps({"complete": True, "n_genes": len(result),
                                       "n_models": len(seeds) * folds}, indent=2))
    success_tmp.replace(output / "_SUCCESS.json")
    print(pd.DataFrame([{"model": "Ast-only Student OOF ensemble", **metrics}]).to_string(index=False))
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.force)
