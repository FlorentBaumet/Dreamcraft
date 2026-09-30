"""
Demos-reves : le modele reve librement sous une action CONSTANTE.

  - dream walk      : action "avancer" maintenue -> le modele imagine le voyage
  - dream lumberjack: action "taper" maintenue   -> le modele imagine le bucheronnage

Ce sont des reves contrefactuels (pas de verite terrain a comparer) : on
regarde la coherence, pas l'exactitude. Le contexte initial vient de vraies
frames ; ensuite le modele est seul avec son imagination.

Sorties : outputs/demos/<tag>/  (dream_<action>_t*.gif + strips)
Lance   : .venv\\Scripts\\python.exe scripts\\dream_demo.py <ckpt> <episode_stem> [tag] [n_steps]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

UP = 4
FPS = 8
N_STARTS = 3

ACTIONS = {
    "avancer": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "taper": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "avancer_taper": np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
}


def load_model(ckpt: Path, device: str):
    cfg = {"base": 32, "in_frames": 1}
    cfg_file = ckpt.parent / "config.json"
    if cfg_file.exists():
        cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
    model = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model, cfg["in_frames"]


def to_img(x):
    return (np.transpose(np.clip(x, 0, 1), (1, 2, 0)) * 255).astype(np.uint8)


def main():
    ckpt = Path(sys.argv[1])
    stem = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else ckpt.parent.name
    n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 48

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "outputs" / "demos" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_u8, actions, ok = C.load_episode_u8(ROOT / "data" / "processed" / f"{stem}.npz")
    model, ctx = load_model(ckpt, device)

    cand = C.valid_starts(ok, C.EVAL_CONTEXT_PAD, 1)
    starts = cand[np.linspace(0, len(cand) - 1, N_STARTS + 2).astype(int)][1:-1]
    t = starts + C.EVAL_CONTEXT_PAD - 1
    ctx_starts = t - ctx + 1

    for name, avec in ACTIONS.items():
        act = torch.from_numpy(np.tile(avec, (len(starts), 1))).to(device)
        x = torch.from_numpy(C.stack_context(frames_u8, ctx_starts, ctx)).to(device).float() / 255.0
        dreams = []
        with torch.no_grad():
            for _ in range(n_steps):
                pred = model(x, act)
                dreams.append(pred.cpu().numpy())
                x = C.roll_context(x, pred)
        dreams = np.stack(dreams, axis=1)  # (S, n_steps, 3, 64, 64)

        for si in range(len(starts)):
            tt = int(t[si])
            gif = []
            start_img = to_img(frames_u8[tt].astype(np.float32) / 255.0)
            for h in range(n_steps):
                dream = to_img(dreams[si, h])
                canvas = np.concatenate(
                    [start_img, np.full((64, 2, 3), 255, np.uint8), dream], axis=1)
                big = cv2.resize(canvas, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
                cv2.putText(big, "DEPART (reel)", (6, big.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(big, f"REVE '{name}' h={h+1}", (64 * UP + 2 * UP + 6, big.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                gif.append(big)
            imageio.mimsave(out_dir / f"dream_{name}_t{tt}.gif", gif, fps=FPS, loop=0)

            steps = [1, 2, 4, 8, 16, 32, min(48, n_steps)]
            row = np.concatenate([to_img(dreams[si, s - 1]) for s in steps], axis=1)
            big = cv2.resize(row, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
            for i, s in enumerate(steps):
                cv2.putText(big, f"h={s}", (i * 64 * UP + 6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"strip_{name}_t{tt}.png"),
                        cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        print(f"[{name}] {len(starts)} reves de {n_steps} pas -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
