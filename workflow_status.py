#!/usr/bin/env python3
"""Machine-readable completeness checks for the resumable SLURM workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs" / "student_gene_attribution.yaml"


def load(path):
    return yaml.safe_load(Path(path).read_text())


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def valid_json(path, expected=None):
    if not nonempty(path):
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return expected is None or all(value.get(key) == item for key, item in expected.items())


def student_complete(cfg, seed):
    directory = ROOT / cfg["attribution"]["model_dir"] / f"seed_{seed}"
    folds = int(cfg["attribution"]["outer_folds"])
    required = [directory / f"fold_{fold}.pt" for fold in range(1, folds + 1)]
    required += [directory / "oof_metrics.csv", directory / "oof_predictions.csv",
                 directory / "oof_representations.npz", directory / "fold_manifest.csv"]
    return (all(nonempty(path) for path in required) and
            valid_json(directory / "_SUCCESS.json", {"seed": int(seed), "folds": folds}))


def attribution_location(cfg, task_id):
    seeds = list(map(int, cfg["attribution"]["student_seeds"]))
    folds = int(cfg["attribution"]["outer_folds"])
    n_genes = int(cfg["scvi"]["n_hvg"])
    chunk_size = int(cfg["attribution"]["genes_per_task"])
    chunks = (n_genes + chunk_size - 1) // chunk_size
    model_number, chunk = divmod(task_id, chunks)
    seed_index, fold_zero = divmod(model_number, folds)
    seed, fold = seeds[seed_index], fold_zero + 1
    directory = ROOT / cfg["attribution"]["attribution_dir"] / f"seed_{seed}" / f"fold_{fold}"
    return directory, chunk, seed, fold, len(seeds) * folds * chunks


def attribution_complete(cfg, task_id):
    directory, chunk, seed, fold, total = attribution_location(cfg, task_id)
    if not 0 <= task_id < total:
        return False
    summary = directory / f"gene_summary_chunk_{chunk:03d}.csv.gz"
    donors = directory / f"donor_effects_chunk_{chunk:03d}.csv.gz"
    manifest = directory / f"manifest_chunk_{chunk:03d}.json"
    return (nonempty(summary) and nonempty(donors) and
            valid_json(manifest, {"task_id": task_id, "seed": seed, "fold": fold, "chunk": chunk}))


def compress_ranges(values):
    values = sorted(set(map(int, values)))
    if not values:
        return ""
    groups, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["missing-students", "missing-attribution", "summary"])
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load(args.config)
    seeds = list(map(int, cfg["attribution"]["student_seeds"]))
    missing_students = [index for index, seed in enumerate(seeds) if not student_complete(cfg, seed)]
    _, _, _, _, total = attribution_location(cfg, 0)
    missing_attribution = [task for task in range(total) if not attribution_complete(cfg, task)]
    if args.command == "missing-students":
        print(compress_ranges(missing_students))
    elif args.command == "missing-attribution":
        selected = missing_attribution[:args.limit] if args.limit > 0 else missing_attribution
        print(compress_ranges(selected))
    else:
        print(json.dumps({
            "student_seeds_complete": len(seeds) - len(missing_students),
            "student_seeds_expected": len(seeds),
            "student_seed_indices_missing": missing_students,
            "attribution_tasks_complete": total - len(missing_attribution),
            "attribution_tasks_expected": total,
            "attribution_task_indices_missing_preview": missing_attribution[:30],
            "aggregate_complete": (
                nonempty(ROOT / cfg["attribution"]["aggregate_dir"] / "_SUCCESS.json") and
                nonempty(ROOT / cfg["attribution"]["aggregate_dir"] / "student_gene_weights_all.csv")
            ),
            "discovery_complete": (
                nonempty(ROOT / cfg["gene_discovery"]["output_dir"] / "_SUCCESS.json") and
                nonempty(ROOT / cfg["gene_discovery"]["output_dir"] / "locked_20_gene_panel.csv")
            ),
        }, indent=2))


if __name__ == "__main__":
    main()

