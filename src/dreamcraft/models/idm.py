"""IDM -- Inverse Dynamics Model (le tour de magie de VPT, version mini).

Tache INVERSE du world model : voir les frames AVANT et APRES -> deviner
l'action pressee entre les deux. Beaucoup plus facile que predire le futur
(la camera qui tourne, les fissures, le bras : c'est visible a l'ecran).

Usage : pseudo-etiqueter des videos YouTube (qui n'ont pas les touches)
pour les rendre digestes par notre pipeline (frame, action).

Entree  : 5 frames consecutives empilees (B, 15, 64, 64), centre = t
Sorties : logits boutons (B, 8)  [avancer..utiliser]
          camera (B, 2)          [dx, dy normalises, tanh]
          logit gui (B, 1)       [menu ouvert ?]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IDM(nn.Module):
    def __init__(self, n_frames: int = 5, base: int = 48):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.conv = nn.Sequential(
            nn.Conv2d(3 * n_frames, c1, 4, 2, 1), nn.GroupNorm(8, c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, 2, 1), nn.GroupNorm(8, c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.SiLU(),
            nn.Conv2d(c3, c4, 4, 2, 1), nn.GroupNorm(8, c4), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c4 * 4 * 4, 512), nn.SiLU(),
        )
        self.buttons = nn.Linear(512, 8)
        self.camera = nn.Linear(512, 2)
        self.gui = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        h = self.head(self.conv(x))
        return self.buttons(h), torch.tanh(self.camera(h)), self.gui(h)
