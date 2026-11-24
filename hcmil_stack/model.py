from __future__ import annotations

import torch
from torch import nn


class PathwayOnlyNet(nn.Module):
    def __init__(self, n_pathways, n_types, cfg):
        super().__init__(); ph, td = int(cfg["pathway_hidden"]), int(cfg["token_dim"])
        self.reliability_cells = float(cfg["reliability_cells"])
        self.pathway_encoder = nn.Sequential(
            nn.Linear(n_pathways, ph), nn.LayerNorm(ph), nn.GELU(), nn.Dropout(float(cfg["dropout"])),
            nn.Linear(ph, td), nn.LayerNorm(td), nn.GELU(),
        )
        self.type_embedding = nn.Parameter(torch.randn(n_types, int(cfg["type_embedding_dim"])) * .02)
        self.token = nn.Sequential(nn.Linear(td + int(cfg["type_embedding_dim"]), td), nn.LayerNorm(td), nn.GELU(),
                                   nn.Dropout(float(cfg["dropout"])))
        self.regression_head = nn.Linear(td, 1)

    def forward(self, pathway, counts):
        encoded = self.pathway_encoder(pathway)
        types = self.type_embedding[None].expand(len(pathway), -1, -1)
        token = self.token(torch.cat((encoded, types), dim=-1))
        present = counts > 0
        reliability = (counts.float() / self.reliability_cells).clamp(0, 1) * present
        weight = reliability / reliability.sum(1, keepdim=True).clamp_min(1e-8)
        return self.regression_head((token * weight[:, :, None]).sum(1)).squeeze(-1)
