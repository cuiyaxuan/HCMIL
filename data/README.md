# Input data

Place a cohort-authorized `.h5ad` file here, or update `data.h5ad_path` in the selected YAML configuration. Data are intentionally excluded from this repository. Follow the ROSMAP and SEA-AD access, consent, and redistribution terms. Never commit donor-level data or linkage files.

The pipeline expects donor, cell-type, Braak, and covariate columns named in the YAML configuration. Run `uv run python inspect_input.py --config configs/student_gene_attribution.yaml` to inspect the input schema before starting.
