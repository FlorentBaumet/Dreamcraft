"""Les deux baselines a battre (protocole gele).

B0 -- copie : pred = frame_t. La 'reine de la triche' : parfaite partout ou
      rien ne bouge, nulle partout ou vit la dynamique.
B1 -- flux optique : estime le mouvement entre frame_{t-1} et frame_t
      (Farneback), suppose qu'il continue tel quel, et deforme frame_t en
      consequence. Imite le mouvement de camera SANS rien comprendre au jeu.
"""
from __future__ import annotations

import cv2
import numpy as np


def b0_copy(frame_t: np.ndarray) -> np.ndarray:
    return frame_t


def b1_optical_flow(frame_prev: np.ndarray, frame_t: np.ndarray) -> np.ndarray:
    """Extrapole frame_{t+1} en supposant le flux (t-1 -> t) constant.

    Warp arriere : pred(x) = frame_t(x - flow(x)) -- approximation standard
    pour un baseline (le flux est evalue a la position d'arrivee).
    Entrees/sortie : float32 [0,1], (H, W, 3).
    """
    g_prev = cv2.cvtColor((frame_prev * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g_t = cv2.cvtColor((frame_t * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g_prev, g_t, None,
        pyr_scale=0.5, levels=2, winsize=9, iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )
    h, w = g_t.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    pred = cv2.remap(frame_t, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    return pred
