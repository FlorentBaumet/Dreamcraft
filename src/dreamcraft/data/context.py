"""Fenetres a contexte multi-frames, partagees par train / rollout / eval.

Conventions :
  - contexte C : le modele voit frames[s .. s+C-1] (empiles sur les canaux,
    le plus recent en DERNIER), le "present" est t = s+C-1 ;
  - pas h (1..K) : predire frame[t+h] avec action[t+h-1] ;
  - un depart s est valide si les transitions s .. t+K-1 sont toutes valides
    (pas de GUI) -> continuite du contexte ET du deroule.

Les frames restent en uint8 en RAM (corpus 100k+ frames) ; la conversion
float se fait par batch, sur GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import vpt

# Toutes les evaluations comparatives utilisent ce pad d'eligibilite constant,
# pour que differents modeles (C=1..4) soient notes sur LES MEMES fenetres.
EVAL_CONTEXT_PAD = 4


def load_episode_u8(npz_path: Path):
    """(frames uint8 (N,3,64,64), actions float32 (N-1,10), ok bool (N-1,))."""
    frames_u8, actions, gui = vpt.load_episode(npz_path)
    frames = np.ascontiguousarray(np.transpose(frames_u8, (0, 3, 1, 2)))
    n = frames.shape[0]
    ok = np.ones(n - 1, dtype=bool)
    ok[gui == 1] = False
    ok[0] = False
    return frames, actions.astype(np.float32), ok


def valid_starts(ok: np.ndarray, context: int, horizon: int) -> np.ndarray:
    """Departs s tels que les (context-1 + horizon) transitions consecutives
    a partir de s sont toutes valides."""
    k = context - 1 + horizon
    conv = np.convolve(ok.astype(int), np.ones(k, dtype=int), mode="valid")
    return np.flatnonzero(conv == k)


def stack_context(frames_u8: np.ndarray, starts: np.ndarray, context: int) -> np.ndarray:
    """(S,) departs -> (S, 3*context, 64, 64) uint8, plus recent en dernier."""
    cols = [frames_u8[starts + i] for i in range(context)]  # chacun (S,3,64,64)
    return np.concatenate(cols, axis=1)


def roll_context(ctx: "np.ndarray | object", new_frame):
    """Fait glisser le contexte : retire la frame la plus ancienne (3 premiers
    canaux), ajoute la nouvelle en fin. Marche pour numpy ET torch."""
    if hasattr(ctx, "narrow"):  # tensor torch
        import torch
        return torch.cat([ctx[:, 3:], new_frame], dim=1)
    return np.concatenate([ctx[:, 3:], new_frame], axis=1)
