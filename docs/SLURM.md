# Running the workflow with SLURM

The commands below show a dependency-safe layout for a cluster. Adapt account, partition, wall time, memory, and GPU directives to local policy. Run `uv sync --extra analysis` once in the project environment before submitting the analysis jobs; on systems where dependency installation must run on a compute node, submit a short setup job first.

All jobs should run from the repository root and use the same YAML file. The examples use `configs/student_gene_attribution.yaml`.

## Stage order

1. Prepare data caches.
2. Run the teacher and scVI feature jobs after preparation. These two jobs can run independently.
3. Run the five Student seeds after both teacher and scVI succeed.
4. Run the 750 gene-attribution chunks after all five Student seeds succeed.
5. Aggregate the attribution chunks, then run gene discovery.

## Array commands

Use one array task per Student seed:

```bash
uv run --no-sync --extra analysis python train_ast_student_models.py \
  --config configs/student_gene_attribution.yaml \
  --seed-index "$SLURM_ARRAY_TASK_ID"
```

Submit it with an array such as `--array=0-4%5`.

Use one array task per gene-attribution chunk:

```bash
uv run --no-sync --extra analysis python attribute_student_genes.py \
  --config configs/student_gene_attribution.yaml \
  --task-id "$SLURM_ARRAY_TASK_ID"
```

For the supplied settings, 5 Student seeds × 5 outer folds × 30 chunks per fold = 750 tasks. An array such as `--array=0-749%5` limits the number running at once to five; choose a concurrency limit appropriate for the cluster and filesystem.

## Dependencies and finalization

Submit arrays with `sbatch --dependency=afterok:<jobid>` so downstream work starts only after upstream success. The final job should run:

```bash
uv run --no-sync --extra analysis python aggregate_gene_attribution.py \
  --config configs/student_gene_attribution.yaml
uv run --no-sync --extra analysis python run_gene_discovery.py \
  --config configs/student_gene_attribution.yaml
uv run --no-sync --extra analysis python workflow_status.py summary \
  --config configs/student_gene_attribution.yaml
```

The status command should report all five Student seeds complete, all 750 attribution tasks complete, and both aggregation and discovery complete. Do not treat an empty queue as proof of success; inspect SLURM exit codes and the workflow status output.
