"""Metriques du protocole gele.

Deux familles :
  - pixel GLOBAL (MSE/PSNR) : reportee mais TRICHABLE (sur scene statique,
    'copier la derniere frame' est quasi parfait) ;
  - Delta-region (LA metrique honnete) : la meme erreur, mais calculee
    UNIQUEMENT sur les pixels qui changent vraiment entre frame_t et
    frame_{t+1}. C'est la que vit la dynamique, et la que la triche s'effondre.

Toutes les images : float32 dans [0, 1], shape (H, W, 3).
"""
from __future__ import annotations

import numpy as np

# Un pixel est "changeant" si un de ses canaux bouge de plus de ~15/255.
CHANGE_THRESHOLD = 0.06
# En dessous de ce nombre de pixels changeants, la paire est consideree
# statique : pas de dynamique a mesurer -> exclue de la moyenne Delta-region.
MIN_CHANGED_PIXELS = 10


def change_mask(prev: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Masque bool (H, W) des pixels qui changent vraiment entre t et t+1."""
    return np.abs(target - prev).max(axis=-1) > CHANGE_THRESHOLD


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def psnr(mse_value: float) -> float:
    if mse_value <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse_value))


def delta_region_mse(pred: np.ndarray, target: np.ndarray, prev: np.ndarray) -> tuple[float, int]:
    """MSE restreinte aux pixels changeants. Retourne (mse, nb_pixels_changeants).

    mse = nan si la paire est statique (moins de MIN_CHANGED_PIXELS changeants).
    """
    mask = change_mask(prev, target)
    n = int(mask.sum())
    if n < MIN_CHANGED_PIXELS:
        return float("nan"), n
    err = (pred[mask] - target[mask]) ** 2
    return float(err.mean()), n
