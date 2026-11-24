"""Minimal CPU forward-pass demo; this is an interface check, not a paper reproduction."""
from __future__ import annotations

import torch
import sys
from pathlib import Path

# Make the repository-root modules importable when this file is run directly
# as `python examples/model_smoke_demo.py` from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student_model import CellScaleTeacherStudent


def main() -> None:
    torch.manual_seed(7)
    cfg = {
        "token_hidden": 16, "token_dim": 8, "donor_dim": 6,
        "type_embedding_dim": 4, "dropout": 0.0, "n_classes": 3,
    }
    model = CellScaleTeacherStudent(input_dim=160, n_types=3, biological_dim=2, cfg=cfg)
    state = torch.randn(5, 3, 160)
    counts = torch.tensor([[80, 25, 0]] * 5)
    biological = torch.randn(5, 2)
    student, cell_only, donor, prototype = model(state, counts, biological)
    assert student.shape == (5, 3) and cell_only.shape == (5, 3)
    assert donor.shape == (5, 6) and prototype.shape == (5, 3, 32)
    print("HCMIL model forward pass succeeded")
    print(f"student logits: {tuple(student.shape)}; device: {student.device}")
    print("This demo uses synthetic tensors and does not train or evaluate a model.")


if __name__ == "__main__":
    main()
