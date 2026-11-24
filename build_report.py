#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def markdown_table(frame, columns, n=15):
    view = frame.loc[:, columns].head(n).copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: f"{value:.4g}")
    return view.to_markdown(index=False)


def main(config, force=False):
    cfg = yaml.safe_load(Path(config).read_text())
    os.chdir(ROOT)
    source = Path(cfg["attribution"]["aggregate_dir"])
    output = Path(cfg["attribution"]["report_dir"])
    if output.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    weights = pd.read_csv(source / "student_gene_weights_all.csv")
    metrics = pd.read_csv(source / "student_oof_ensemble_metrics.csv")
    positive = pd.read_csv(source / "top_positive_student_genes.csv")
    negative = pd.read_csv(source / "top_negative_student_genes.csv")
    columns = ["gene", "student_gene_weight", "cell_only_gene_weight", "weight_fdr", "model_direction_consistency",
               "domain_direction_consistency", "expression_prevalence"]
    lines = [
        "# Ast-only Student gene-attribution report", "",
        "## Model performance", "", metrics.to_markdown(index=False), "",
        "## Interpretation", "",
        "The reported Student gene weight is the donor-level mean of `baseline expected Braak - gene-knockout expected Braak`.",
        "A positive value means that the gene's presence raises the Ast-only Student severity score; a negative value means that its presence lowers the score.",
        "Every donor is attributed only with the outer-fold checkpoint that did not train on that donor. The sealed-test labels are never used.", "",
        "These are nonlinear in-silico knockout effects, not linear regression coefficients and not causal effects.", "",
        "## Top stable positive genes", "", markdown_table(positive, columns), "",
        "## Top stable negative genes", "", markdown_table(negative, columns), "",
        "## Output guide", "",
        "- `student_gene_weights_all.csv`: all scVI HVGs with direction, magnitude, FDR and stability.",
        "- `top_positive_student_genes.csv` and `top_negative_student_genes.csv`: experiment-prioritization lists.",
        "- `donor_ensemble_gene_effects.csv.gz`: donor-level effects averaged across seeds.",
        "- `all_model_gene_effects.csv.gz`: seed/fold-level stability table.", "",
        "## Required wording", "",
        "Use 'Student model gene attribution' or 'in-silico knockout effect'. Do not call the values ordinary neural-network coefficients or causal gene effects.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines))
    for name in ("student_gene_weights_all.csv", "top_stable_student_genes.csv",
                 "top_positive_student_genes.csv", "top_negative_student_genes.csv",
                 "prior_candidate_gene_audit.csv", "student_oof_ensemble_metrics.csv", "manifest.json"):
        shutil.copy2(source / name, output / name)
    (output / "manifest.json").write_text(json.dumps({
        **json.loads((source / "manifest.json").read_text()),
        "report": "SUMMARY.md", "primary_table": "student_gene_weights_all.csv",
    }, indent=2))
    print((output / "SUMMARY.md").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.force)
