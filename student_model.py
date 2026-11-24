from __future__ import annotations

import torch
from torch import nn


class CellScaleTeacherStudent(nn.Module):
    def __init__(self, input_dim, n_types, biological_dim, cfg):
        super().__init__(); hidden, token_dim, donor_dim = int(cfg["token_hidden"]), int(cfg["token_dim"]), int(cfg["donor_dim"])
        type_dim, drop, n_classes = int(cfg["type_embedding_dim"]), float(cfg["dropout"]), int(cfg["n_classes"])
        self.type_embedding = nn.Parameter(torch.randn(n_types, type_dim) * .02)
        self.token_encoder = nn.Sequential(nn.Linear(input_dim + type_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
                                           nn.Linear(hidden, token_dim), nn.LayerNorm(token_dim), nn.GELU())
        self.donor_encoder = nn.Sequential(nn.Linear(token_dim, donor_dim), nn.LayerNorm(donor_dim), nn.GELU(), nn.Dropout(drop))
        self.cell_only_head = nn.Linear(donor_dim, n_classes)
        self.student_head = nn.Sequential(nn.Linear(donor_dim + biological_dim, 24), nn.LayerNorm(24), nn.GELU(), nn.Dropout(drop), nn.Linear(24, n_classes))
        self.prototype_decoder = nn.Sequential(nn.Linear(donor_dim + type_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 32))

    def forward(self, state, exact_counts, biological):
        types = self.type_embedding[None].expand(len(state), -1, -1)
        token = self.token_encoder(torch.cat((state, types), -1)); present = exact_counts > 0
        weight = exact_counts.float().sqrt() * present; weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        donor = self.donor_encoder((token * weight[..., None]).sum(1))
        cell_only = self.cell_only_head(donor)
        student = self.student_head(torch.cat((donor, biological), -1))
        donor_by_type = donor[:, None].expand(-1, state.shape[1], -1)
        prototype = self.prototype_decoder(torch.cat((donor_by_type, types), -1))
        return student, cell_only, donor, prototype
