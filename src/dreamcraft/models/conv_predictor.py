"""Mini predicteur de frames action-conditionne (Phase 0).

Architecture : encoder-decoder convolutionnel avec skips (style U-Net leger).
L'action module le bottleneck via FiLM (scale/shift appris) : c'est par la
que "avancer" vs "tourner" vs "taper" peut changer la prediction.

Entree  : frame_t (B, 3, 64, 64) dans [0,1] + action (B, 10)
Sortie  : frame_{t+1} predite (B, 3, 64, 64) dans [0,1]

Note honnetete : les skips donnent au modele la copie "gratuite" des zones
statiques -- c'est voulu (la copie n'est pas le crime ; le crime est de ne
faire QUE copier). Le verdict se joue sur la metrique Delta-region.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


def _deconv(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class ConvPredictor(nn.Module):
    """in_frames=1 : le modele Phase 0 (une frame d'entree).
    in_frames=C : contexte de C frames empilees sur les canaux (3*C canaux)
    -> le modele peut percevoir le mouvement en cours. La sortie reste
    la seule frame suivante."""

    def __init__(self, action_dim: int = 10, base: int = 32, in_frames: int = 1):
        super().__init__()
        self.in_frames = in_frames
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8  # 32,64,128,256

        self.e1 = _conv(3 * in_frames, c1)    # 64 -> 32
        self.e2 = _conv(c1, c2)   # 32 -> 16
        self.e3 = _conv(c2, c3)   # 16 -> 8
        self.e4 = _conv(c3, c4)   # 8  -> 4

        # FiLM : action -> (scale, shift) sur les canaux du bottleneck
        self.film = nn.Sequential(
            nn.Linear(action_dim, 128), nn.SiLU(),
            nn.Linear(128, c4 * 2),
        )

        self.d4 = _deconv(c4, c3)          # 4  -> 8
        self.d3 = _deconv(c3 + c3, c2)     # 8  -> 16  (skip e3)
        self.d2 = _deconv(c2 + c2, c1)     # 16 -> 32  (skip e2)
        self.d1 = _deconv(c1 + c1, c1)     # 32 -> 64  (skip e1)
        self.out = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """frame : (B, 3*in_frames, 64, 64) -- contexte empile, le plus recent en dernier."""
        h1 = self.e1(frame)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)

        scale, shift = self.film(action).chunk(2, dim=-1)
        h4 = h4 * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        y = self.d4(h4)
        y = self.d3(torch.cat([y, h3], dim=1))
        y = self.d2(torch.cat([y, h2], dim=1))
        y = self.d1(torch.cat([y, h1], dim=1))
        return torch.sigmoid(self.out(y))
