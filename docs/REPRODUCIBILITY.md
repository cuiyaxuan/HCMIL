# Reproducibility record

Recorded 2026-11-24.

## Scope

This record separates completed checks from the remaining end-to-end work. It does not claim that the public checkout has already regenerated the final discovery panel.

The shared core Python analysis files in this repository were compared with the original `hcmil_ast_student_gene_discovery_fromscratch_standalone_iu_slurm` source directory. No differences were found in those shared Python files. The public checkout uses portable input paths and documents the external GMT resources; neither controlled-access ROSMAP data nor donor-level outputs are included here.

## Completed H100 checks

The public-code workflow was run on IU Quartz with an NVIDIA H100 80 GB GPU. The reported environment was Python 3.12, PyTorch 2.15.0+cu132 (CUDA 13.2), scvi-tools 1.5.2, AnnData 0.12.19, and Scanpy 1.12.6.

Input validation reported an H5AD with 2,663,736 cells and 33,538 genes. The standalone filters retained 421 eligible donors: 336 development donors and 85 held-out donors. Raw Braak class counts (0–2/3–4/5–6) were 111/200/112; eligible counts were 111/198/112. Two labeled donors had no cells in the retained cell types and were excluded. The sealed test labels were not used for training or metric selection.

The completed scVI output manifest matched the reference run on the main feature dimensions: 2,333,874 encoded cells, 367,448 scVI training cells, 3,000 HVGs, 32 latent dimensions, and 336 development donors.

Teacher and five-seed Astrocyte Student OOF metrics from the public-code H100 run were close to the original standalone run:

| Metric | Public-code H100 run | Original standalone run |
| --- | ---: | ---: |
| Teacher accuracy | 0.4911 | 0.5030 |
| Teacher balanced accuracy | 0.5149 | 0.5250 |
| Teacher macro-F1 | 0.4940 | 0.5048 |
| Teacher macro-AUROC | 0.6905 | 0.6898 |
| Student accuracy, mean across five seeds | 0.5042 | 0.5137 |
| Student balanced accuracy, mean across five seeds | 0.5209 | 0.5254 |
| Student macro-F1, mean across five seeds | 0.5047 | 0.5132 |
| Student macro-AUROC, mean across five seeds | 0.6917 | 0.6994 |
| Student raw-Braak Spearman, mean across five seeds | 0.4663 | 0.4592 |

Each Student OOF metric uses 336 development donors. The averages are descriptive comparisons, not a new significance test. Individual seed values differ, so this is broad numerical agreement rather than bit-for-bit identity. The public H100 environment uses Python 3.12/uv, while the older source run logs report Python 3.13/conda; accelerator kernels and dependency versions can also affect numerical results.

## Remaining full-run verification

The current public-code H100 run had completed input preparation, teacher training, scVI feature generation, and all five Student OOF seeds when these results were recorded. Its 750 gene-knockout attribution chunks and final aggregation still need to complete before the public checkout can be described as having reproduced the complete discovery workflow. Use `workflow_status.py summary` to check completion. Only after that should the final discovery files be compared with a reference run.

An earlier run of the source project completed all 750 attribution chunks and attribution aggregation, but its final report step failed because that older conda environment lacked the `tabulate` dependency used by pandas Markdown output. The public `pyproject.toml` declares `tabulate`; use the documented `uv sync --extra analysis` environment. The old output directory contains partial discovery files but no successful completion marker, so those files are not treated as a validated final result. The discovery code can return fewer than 20 genes when fewer candidates pass its filters; the output filename alone does not guarantee 20 rows.

The 85 held-out donors remain sealed. Do not use their labels to tune the model or gene panel. These checks support reproducibility of the development workflow; they do not establish clinical validity or a successful sealed-test result.

## Files used for the reference comparison

- `artifacts/teacher_nestedcv/braak3_oof_metrics.csv`
- `artifacts/standalone_scvi32_cache/manifest.json`
- `artifacts/ast_student_oof_models/seed_*/oof_metrics.csv`

The reference result files are not copied into this public repository. The full cohort H5AD and pathway GMT resources are also external inputs and must be obtained under their respective access terms.
