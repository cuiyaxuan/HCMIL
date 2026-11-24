# HCMIL astrocyte Braak analysis

Research code accompanying the study “Interpretable single-nucleus learning reveals an astrocyte program tracking local neocortical tau rather than Braak stage” by Yaxuan Cui and Jingwen Yan. The repository contains the standalone ROSMAP DLPFC workflow for training a pathway- and covariate-aware teacher, learning an astrocyte-only student, computing out-of-fold gene knockout attributions, and deriving a discovery panel of up to 20 genes that pass the configured criteria.

The core workflow was run on IU Quartz H100 using this code lineage. The current verification covers input preparation, teacher OOF evaluation, scVI feature generation, and all five Astrocyte Student OOF seeds; the H100 run's 750 gene-knockout tasks and final aggregation should be allowed to finish before claiming that the full discovery output has been reproduced. See [the reproducibility record](docs/REPRODUCIBILITY.md) for the exact scope and comparison.

This is research software, not a packaged clinical tool. The paper's discovery and validation results depend on access-controlled cohort data and specified data processing; those data, derived donor-level tables, checkpoints, and sealed test identifiers are not included. A synthetic forward-pass demo is provided for checking the environment only. It does not reproduce the published results.

## Quick start with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available. Python 3.11 or 3.12 is supported. In the repository root:

```bash
uv sync
uv run python examples/model_smoke_demo.py
```

For the H5AD analysis stack (AnnData, Scanpy, and scvi-tools):

```bash
uv sync --extra analysis
```

Check the official PyTorch installation guidance if you need a particular CUDA build on Linux. On Apple Silicon, the standard PyTorch wheel can use MPS for supported operations; the full scvi-tools pipeline has not been characterized on every operating system or accelerator.

## Run the cohort workflow

1. Obtain ROSMAP data through the approved access process and place the authorized H5AD at `data/ROSMAP_enriched_metadata.h5ad` (or change `data.h5ad_path` in the selected YAML configuration).
2. Obtain the matching Hallmark and Reactome GMT files under their terms and place them in `resources/` using the versioned filenames in the configuration; see [resources/README.md](resources/README.md).
3. Review all keys, filtering rules, covariates, paths, and cohort split settings in `configs/student_gene_attribution.yaml`.
4. Inspect the input schema before running any stage:

   ```bash
   uv run python inspect_input.py --config configs/student_gene_attribution.yaml
   ```

5. Run the stages on a machine with adequate memory and a supported GPU for practical runtimes. The core sequence is:

   ```bash
   uv run python prepare_standalone_data.py --config configs/student_gene_attribution.yaml
   uv run python train_teacher.py --config configs/student_gene_attribution.yaml
   uv run python train_scvi_and_build_cellscale_features.py --config configs/student_gene_attribution.yaml
   ```

   Teacher training and scVI feature generation both depend on the prepared caches and can be run independently after preparation. Then train the five out-of-fold Astrocyte-only Student seeds, run each of the 750 gene-knockout chunks, and aggregate/discover the panel:

   ```bash
   for seed_index in 0 1 2 3 4; do
     uv run python train_ast_student_models.py --config configs/student_gene_attribution.yaml --seed-index "$seed_index"
   done
   for task_id in $(seq 0 749); do
     uv run python attribute_student_genes.py --config configs/student_gene_attribution.yaml --task-id "$task_id"
   done
   uv run python aggregate_gene_attribution.py --config configs/student_gene_attribution.yaml
   uv run python run_gene_discovery.py --config configs/student_gene_attribution.yaml
   ```

   For a fair method benchmark, the separate `train_students.py` step reports sampled/all-cell variants and is described by its CLI. Outputs are written beneath `artifacts/`, which is git-ignored.

   On a SLURM cluster, run each stage as a batch job and use job dependencies. The five Student seeds are a job array (`--array=0-4`); the 750 attribution chunks are another array (for example, `--array=0-749%5` to cap concurrency at five). A cluster-oriented outline is in [docs/SLURM.md](docs/SLURM.md).

The original execution wrappers targeted IU's SLURM cluster and were excluded. The training, knockout, and discovery calculations are retained, but the full-scale workflow needs suitable compute resources. The 750 knockout tasks are intentionally shown as sequential commands for clarity; parallelize them only with an appropriate job scheduler and output-safe worker setup. The full-scale workflow is not a one-command laptop analysis.

## Reproducibility and interpretation

The manuscript reports a frozen scVI representation, donor-level cell-type summaries, a teacher that combines pseudobulk expression, pathway scores, and biological covariates, and a student trained with reconstruction, distillation, and classification losses. Gene effects are computed by setting a gene's raw astrocyte counts to zero for held-out donors and measuring the change in expected Braak class. Gene selection uses development donors only. The output is capped at 20 candidates and may contain fewer if fewer genes pass the configured criteria. Sealed test labels must remain unavailable until predictions and analysis choices have been frozen.

Do not use sealed outcomes to tune preprocessing, choose genes, or select models. Report the actual cohort, split, software versions, random seeds, missing-data handling, and deviations from the manuscript. The reported 20-gene panel did not pass the pre-registered sealed DLPFC test; transfer to SEA-AD MTG should not be described as broad validation across regions.

## Repository contents

- `student_model.py`, `hcmil_stack/`: neural model and shared metrics used by the analysis.
- `train_*.py`, `attribute_student_genes.py`, `run_gene_discovery.py`: standalone analysis stages.
- `configs/`: study configuration with a portable, local input path.
- `examples/model_smoke_demo.py`: quick environment and model-interface check using synthetic arrays.
- `data/`, `resources/`: instructions for supplying data and pathway collections; no cohort data are included.

## Citation

Please cite the associated manuscript and this code repository. Add the final journal citation and DOI here when assigned.

## License

Code is released under the MIT License. Cohort data and pathway databases remain subject to their own access and license terms.
