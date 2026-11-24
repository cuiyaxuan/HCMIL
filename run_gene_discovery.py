#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from gene_discovery_utils import bh_adjust, covariate_transform, percentile, write_markdown_table


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def sparse_logcpm(counts):
    value = counts.tocsr().astype(np.float32, copy=True)
    library = np.asarray(value.sum(1)).ravel().astype(np.float32)
    row = np.repeat(np.arange(value.shape[0]), np.diff(value.indptr))
    scale = np.divide(1e6, library, out=np.zeros_like(library), where=library > 0)
    value.data = np.log1p(value.data * scale[row])
    return value


def association(expression, braak, rows, covariates, numeric, categorical):
    transform, work = covariate_transform(covariates[numeric + categorical], rows, numeric, categorical)
    nuisance = np.asarray(transform.transform(work.iloc[rows]), dtype=float)
    outcome = braak[rows]
    outcome_z = (outcome - np.mean(outcome)) / max(np.std(outcome), 1e-12)
    design = np.column_stack((np.ones(len(rows)), outcome_z, nuisance))
    inverse = np.linalg.pinv(design)
    beta = inverse @ expression[rows]
    residual = expression[rows] - design @ beta
    degrees = max(1, len(rows) - np.linalg.matrix_rank(design))
    sigma2 = np.sum(residual ** 2, axis=0) / degrees
    variance = np.linalg.pinv(design.T @ design)[1, 1]
    standard_error = np.sqrt(np.maximum(0, sigma2 * variance))
    tvalue = np.divide(beta[1], standard_error, out=np.zeros_like(beta[1]), where=standard_error > 0)
    pvalue = 2 * stats.t.sf(np.abs(tvalue), degrees)
    prevalence = np.mean(expression[rows] > 0, axis=0)
    return beta[1], tvalue, pvalue, bh_adjust(pvalue), prevalence, degrees


def main(config, force=False):
    cfg = load_config(config)
    discovery = cfg["gene_discovery"]
    output = ROOT / discovery["output_dir"]
    complete = output / "locked_20_gene_panel.csv"
    success = output / "_SUCCESS.json"
    if complete.is_file() and complete.stat().st_size > 0 and success.is_file() and not force:
        print(f"Gene discovery already complete: {complete}")
        return
    if output.exists():
        print(f"Removing incomplete gene-discovery output before retry: {output}", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # Everything below is generated inside this project from the source H5AD.
    linear = ROOT / cfg["data"]["linear_cache"]
    pathway = ROOT / cfg["data"]["pathway_cache"]
    metadata = np.load(linear / "metadata.npz", allow_pickle=False)
    features = np.load(pathway / "features.npz", allow_pickle=False)
    donor_names = metadata["donor_names"].astype(str)
    type_names = metadata["cell_type_names"].astype(str)
    split = metadata["split"].astype(str)
    braak = features["braak"].astype(float)
    if not np.array_equal(donor_names, features["donor_names"].astype(str)):
        raise ValueError("Donor order differs between the internal caches")
    ast_match = np.flatnonzero(type_names == str(cfg["attribution"]["cell_type"]))
    if len(ast_match) != 1:
        raise ValueError(f"Expected exactly one Ast type, found {ast_match.tolist()}")
    ast_index = int(ast_match[0])
    counts_all = sparse.load_npz(linear / "donor_type_counts.npz")
    ast_rows = np.arange(len(donor_names)) * len(type_names) + ast_index
    ast_counts = counts_all[ast_rows].tocsr()
    expression = sparse_logcpm(ast_counts).toarray().astype(np.float32)

    # Resolve stable gene identifiers and display symbols directly from the raw H5AD.
    import anndata as ad
    backed = ad.read_h5ad(cfg["data"]["h5ad_path"], backed="r")
    var_name = backed.var_names.astype(str).to_numpy()
    symbol_key = str(cfg["attribution"].get("gene_symbol_key", "gene_names"))
    symbols = (backed.var[symbol_key].astype(str).to_numpy()
               if symbol_key in backed.var else var_name.copy())
    backed.file.close()
    if expression.shape[1] != len(var_name):
        raise ValueError("Raw H5AD gene order does not match the project pseudobulk cache")

    covariate_names = features["clinical_names"].astype(str)
    covariates = pd.DataFrame(features["clinical_values"].astype(str), columns=covariate_names)
    numeric = [x for x in discovery["adjustment_numeric"] if x in covariates]
    categorical = [x for x in discovery["adjustment_categorical"] if x in covariates]
    dev = np.flatnonzero((split == "train") & np.isfinite(braak))
    expected = int(cfg["data"]["expected_development_donors"])
    if len(dev) != expected:
        raise ValueError(f"Expected {expected} development donors, found {len(dev)}")
    beta, tvalue, pvalue, fdr, prevalence, degrees = association(
        expression, braak, dev, covariates, numeric, categorical
    )
    association_table = pd.DataFrame({
        "var_name": var_name,
        "gene": symbols,
        "adjusted_braak_beta_per_sd": beta,
        "adjusted_braak_t": tvalue,
        "adjusted_braak_p": pvalue,
        "adjusted_braak_fdr": fdr,
        "pseudobulk_expression_prevalence": prevalence,
        "mean_ast_logcpm": expression[dev].mean(axis=0),
    })
    association_table.to_csv(output / "all_gene_adjusted_braak_association.csv.gz", index=False, compression="gzip")

    weights_path = ROOT / cfg["attribution"]["aggregate_dir"] / "student_gene_weights_all.csv"
    weights = pd.read_csv(weights_path)
    # var_name is the unique H5AD feature identifier; gene is retained for display.
    table = weights.merge(association_table, on="var_name", how="left", suffixes=("", "_association"))
    if "gene_association" in table:
        table["gene"] = table["gene"].fillna(table["gene_association"])
    table["student_cell_only_direction_match"] = (
        np.sign(table.student_gene_weight) == np.sign(table.cell_only_gene_weight)
    )
    table["student_association_direction_match"] = (
        np.sign(table.student_gene_weight) == np.sign(table.adjusted_braak_beta_per_sd)
    )
    stable = (
        (table.model_direction_consistency >= float(discovery["minimum_student_model_consistency"])) &
        (table.cell_only_model_direction_consistency >= float(discovery["minimum_cell_only_consistency"])) &
        (table.domain_direction_consistency.fillna(0) >= float(discovery["minimum_domain_consistency"])) &
        (table.pseudobulk_expression_prevalence >= float(discovery["minimum_expression_prevalence"])) &
        table.student_cell_only_direction_match &
        table.student_association_direction_match
    )
    table["passes_stability_and_direction"] = stable
    table["fdr_tier"] = "FDR>0.20"
    table.loc[stable & (table.adjusted_braak_fdr <= 0.20), "fdr_tier"] = "FDR<=0.20"
    table.loc[stable & (table.adjusted_braak_fdr <= 0.10), "fdr_tier"] = "FDR<=0.10"
    table.loc[stable & (table.adjusted_braak_fdr <= 0.05), "fdr_tier"] = "FDR<=0.05"
    table["consensus_score"] = (
        0.30 * percentile(table.student_gene_weight.abs()) +
        0.15 * table.model_direction_consistency.fillna(0) +
        0.10 * table.cell_only_model_direction_consistency.fillna(0) +
        0.10 * table.domain_direction_consistency.fillna(0) +
        0.25 * percentile(table.adjusted_braak_t.abs()) +
        0.10 * table.student_association_direction_match.astype(float)
    )
    strict = stable & (table.adjusted_braak_fdr <= float(discovery["exploratory_fdr_threshold"]))
    table["eligible_tier1"] = strict
    table = table.sort_values(["eligible_tier1", "consensus_score", "gene"], ascending=[False, False, True])
    table.to_csv(output / "student_gene_consensus_all.csv", index=False)
    tier1 = table[strict].sort_values(["consensus_score", "gene"], ascending=[False, True]).copy()
    tier1.to_csv(output / "tier1_fdr20_candidates.csv", index=False)
    tier1[tier1.adjusted_braak_fdr <= 0.10].to_csv(output / "tier1_fdr10_candidates.csv", index=False)
    tier1[tier1.adjusted_braak_fdr <= 0.05].to_csv(output / "tier1_fdr05_candidates.csv", index=False)

    target = int(discovery["panel_size"])
    positive = tier1[tier1.student_gene_weight > 0].head(target // 2)
    negative = tier1[tier1.student_gene_weight < 0].head(target // 2)
    panel = pd.concat([positive, negative])
    if len(panel) < target:
        remainder = tier1.loc[~tier1.index.isin(panel.index)].head(target - len(panel))
        panel = pd.concat([panel, remainder])
    panel = panel.sort_values("consensus_score", ascending=False).copy()
    panel["signature_sign"] = np.sign(panel.student_gene_weight).astype(int)
    panel["panel_rank"] = np.arange(1, len(panel) + 1)
    panel.to_csv(complete, index=False)

    counts = {
        "development_donors": int(len(dev)),
        "genes_in_bh_family": int(np.isfinite(pvalue).sum()),
        "student_hvgs_tested": int(len(weights)),
        "stable_direction_matched": int(stable.sum()),
        "tier1_fdr_le_0_20": int((strict).sum()),
        "tier1_fdr_le_0_10": int((strict & (table.adjusted_braak_fdr <= 0.10)).sum()),
        "tier1_fdr_le_0_05": int((strict & (table.adjusted_braak_fdr <= 0.05)).sum()),
        "locked_panel_size": int(len(panel)),
        "positive_panel_genes": int((panel.student_gene_weight > 0).sum()),
        "negative_panel_genes": int((panel.student_gene_weight < 0).sum()),
        "association_degrees_of_freedom": int(degrees),
    }
    (output / "gene_discovery_manifest.json").write_text(json.dumps({
        **counts,
        "standalone": True,
        "external_project_artifacts_used": False,
        "association_model": "Ast donor pseudobulk logCPM ~ standardized continuous Braak + configured covariates",
        "multiple_testing": "Benjamini-Hochberg across every raw-H5AD gene tested",
        "student_effect": "baseline expected Braak minus gene-knockout expected Braak",
        "panel_rule": "top consensus-ranked Tier1 genes, direction-balanced when available",
    }, indent=2))

    columns = ["gene", "student_gene_weight", "cell_only_gene_weight",
               "adjusted_braak_beta_per_sd", "adjusted_braak_fdr", "consensus_score"]
    report = [
        "# Standalone Ast Student gene-discovery report",
        "",
        f"- Development donors: {len(dev)}",
        f"- BH family: {counts['genes_in_bh_family']} genes",
        f"- Stable direction-matched genes: {counts['stable_direction_matched']}",
        f"- Tier1 FDR <= 0.20: {counts['tier1_fdr_le_0_20']}",
        f"- Tier1 FDR <= 0.10: {counts['tier1_fdr_le_0_10']}",
        f"- Tier1 FDR <= 0.05: {counts['tier1_fdr_le_0_05']}",
        f"- Locked panel: {len(panel)} genes",
        "",
        "## Locked panel",
        "",
        write_markdown_table(panel, columns, maximum=target),
        "",
        "The FDR<=0.20 set is exploratory. FDR<=0.10 and FDR<=0.05 are exported separately.",
        "No PCA-Ridge result or artifact from another project is used in selection.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(report))
    success_tmp = output / "_SUCCESS.json.tmp"
    success_tmp.write_text(json.dumps({"complete": True, **counts}, indent=2))
    success_tmp.replace(success)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(arguments.config, arguments.force)

