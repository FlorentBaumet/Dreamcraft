"""
Eval 1-pas du protocole gele (B0 / B1 / modele, global + Delta-region)
sur UN episode et UN checkpoint. Compatible contexte multi-frames (config.json).
Les paires evaluees utilisent un pad d'eligibilite CONSTANT -> tous les
modeles sont notes sur exactement les memes paires.

Lance : .venv\\Scripts\\python.exe scripts\\eval_1step.py <ckpt> <episode_stem>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.eval import baselines, metrics  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402


def load_model(ckpt: Path, device: str):
    cfg = {"base": 32, "in_frames": 1}
    cfg_file = ckpt.parent / "config.json"
    if cfg_file.exists():
        cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
    model = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model, cfg["in_frames"]


def main():
    ckpt, stem = Path(sys.argv[1]), sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames_u8, actions, ok = C.load_episode_u8(ROOT / "data" / "processed" / f"{stem}.npz")
    frames = frames_u8.astype(np.float32) / 255.0
    model, ctx = load_model(ckpt, device)

    pad = C.EVAL_CONTEXT_PAD
    starts = C.valid_starts(ok, pad, 1)
    t = starts + pad - 1
    ctx_starts = t - ctx + 1

    sums = {"g": {"B0": 0.0, "B1": 0.0, "M": 0.0}, "d": {"B0": 0.0, "B1": 0.0, "M": 0.0}}
    n_all, n_dyn = 0, 0
    with torch.no_grad():
        for i in range(0, len(t), 256):
            tb = t[i: i + 256]
            cb = ctx_starts[i: i + 256]
            x = torch.from_numpy(C.stack_context(frames_u8, cb, ctx)).to(device).float() / 255.0
            act = torch.from_numpy(actions[tb]).to(device)
            preds = model(x, act).cpu().numpy()
            for j, tt in enumerate(tb):
                cur = np.transpose(frames[tt], (1, 2, 0))
                prev = np.transpose(frames[tt - 1], (1, 2, 0))
                tgt = np.transpose(frames[tt + 1], (1, 2, 0))
                pm = np.transpose(preds[j], (1, 2, 0))
                p1 = baselines.b1_optical_flow(prev, cur)
                n_all += 1
                sums["g"]["B0"] += metrics.mse(cur, tgt)
                sums["g"]["B1"] += metrics.mse(p1, tgt)
                sums["g"]["M"] += metrics.mse(pm, tgt)
                d0, _ = metrics.delta_region_mse(cur, tgt, cur)
                d1, _ = metrics.delta_region_mse(p1, tgt, cur)
                dm, _ = metrics.delta_region_mse(pm, tgt, cur)
                if not np.isnan(d0):
                    n_dyn += 1
                    sums["d"]["B0"] += d0
                    sums["d"]["B1"] += d1
                    sums["d"]["M"] += dm

    out = {
        "ckpt": str(ckpt), "episode": stem, "contexte": ctx,
        "paires": n_all, "dynamiques": n_dyn,
        "globale": {k: v / n_all for k, v in sums["g"].items()},
        "delta": {k: v / max(n_dyn, 1) for k, v in sums["d"].items()},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
