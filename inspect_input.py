#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from hcmil_stack.classification import braak_to_class

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(root / "configs" / "student_gene_attribution.yaml"))
args = parser.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
import anndata as ad
adata = ad.read_h5ad(cfg["data"]["h5ad_path"], backed="r")
print("shape:", adata.shape)
print("layers:", list(adata.layers.keys()))
print("required obs:")
for key in (cfg["data"]["donor_key"], cfg["data"]["cell_type_key"], cfg["data"]["braak_key"]):
    print(" ", key, "present=" + str(key in adata.obs), "unique=" + str(adata.obs[key].nunique(dropna=True) if key in adata.obs else "NA"))
dk, tk, bk = cfg["data"]["donor_key"], cfg["data"]["cell_type_key"], cfg["data"]["braak_key"]
label = pd.DataFrame({dk: adata.obs[dk].astype(str), bk: adata.obs[bk].astype(str)}).groupby(dk, sort=False).first()
valid_label = pd.to_numeric(label[bk], errors="coerce").notna(); labeled = set(label.index[valid_label].astype(str))
donor = adata.obs[dk].astype("string"); cell_type = adata.obs[tk].astype("string")
valid_cell = donor.notna() & cell_type.notna() & donor.astype(str).isin(labeled)
type_counts = cell_type[valid_cell].astype(str).value_counts()
retained_types = set(type_counts[type_counts >= int(cfg["data"]["minimum_cells_per_type"])].index.astype(str))
eligible_cell = valid_cell & cell_type.astype(str).isin(retained_types)
eligible_donors = set(donor[eligible_cell].astype(str)); excluded = sorted(labeled - eligible_donors)
print("Braak-labeled donors:", len(labeled))
print("retained cell types:", len(retained_types))
print("eligible donors before split:", len(eligible_donors))
print("excluded zero-eligible-cell donors:", excluded)
raw_braak = pd.to_numeric(label.loc[valid_label, bk], errors="coerce").to_numpy(float)
eligible_braak = pd.to_numeric(label.reindex(sorted(eligible_donors))[bk], errors="coerce").to_numpy(float)
print("raw Braak 3-class counts (0-2/3-4/5-6):", np.bincount(braak_to_class(raw_braak), minlength=3).tolist())
print("eligible Braak 3-class counts:", np.bincount(braak_to_class(eligible_braak), minlength=3).tolist())
layer = cfg["data"].get("counts_layer"); layer = None if layer in (None, "", "null") else layer
source = adata.X if layer is None else adata.layers[layer]
value = source[:min(2048, adata.n_obs)]
if hasattr(value, "toarray"): value = value.toarray()
value = np.asarray(value); finite = value[np.isfinite(value)]
print("configured count source:", "X" if layer is None else f"layers[{layer!r}]")
print("sample min/max:", float(finite.min()), float(finite.max()))
print("sample integer-like:", bool(np.allclose(finite, np.rint(finite))))
adata.file.close()
