"""
GIF du RÉSULTAT : même instant de minage rêvé par le modèle TIMIDE (baseline)
vs le COURAGEUX (v13b, gavé de données in-domain). 3 colonnes : réel | timide | courageux.

Montre visuellement que le courageux fait DISPARAITRE le bloc visé (minage),
là où le timide fige la scène.

Lance : .venv\\Scripts\\python.exe scripts\\gif_courage.py <ep> <ckpt_timide> <ckpt_courageux>
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

OUT = ROOT / "outputs" / "gif_courage"
H = 20
UP = 5
FPS = 6
PATCH = (26, 38)
THETA = 0.10


def load_ep(stem):
    for base in ["processed", "processed_youtube"]:
        p = ROOT / "data" / base / f"{stem}.npz"
        if p.exists():
            return C.load_episode_u8(p)
    raise FileNotFoundError(stem)


def load_model(ckpt, device):
    cfg = {"base": 32, "in_frames": 1}
    cf = Path(ckpt).parent / "config.json"
    if cf.exists():
        cfg.update(json.loads(cf.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


@torch.no_grad()
def dream(model, ctx_u8, acts, device):
    x = torch.from_numpy(ctx_u8).to(device).float() / 255.0
    out = np.zeros((acts.shape[0], 3, 64, 64), dtype=np.float32)
    outs = []
    for h in range(acts.shape[1]):
        pred = model(x, torch.from_numpy(acts[:, h]).to(device))
        outs.append(pred.cpu().numpy())
        x = C.roll_context(x, pred)
    return np.stack(outs, axis=1)   # (B,H,3,64,64)


def to_img(x):
    return (np.transpose(np.clip(x, 0, 1), (1, 2, 0)) * 255).astype(np.uint8)


def main():
    stem, ck_timide, ck_brave = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    fr_u8, actions, ok = load_ep(stem)
    frames = fr_u8.astype(np.float32) / 255.0
    n = len(frames)
    tap = actions[:, 6] > 0.5
    # evenements de minage avec cassage reel confirme (le bloc central disparait)
    y0, y1 = PATCH
    ev = []
    for t in range(6, n - H - 2):
        if tap[t] and not tap[t - 1] and tap[t:t + 6].all() and ok[t - 5:t + H].all():
            ref = frames[t][:, y0:y1, y0:y1]
            d = np.abs(frames[t + 1:t + 1 + H][:, :, y0:y1, y0:y1] - ref).mean(axis=(1, 2, 3))
            if (d > THETA).any():
                ev.append(t)
    print(f"{len(ev)} evenements de minage avec cassage reel")
    if not ev:
        return
    picks = [ev[i] for i in np.linspace(0, len(ev) - 1, min(4, len(ev))).astype(int)]

    mt, ctxt = load_model(ck_timide, device)
    mb, ctxb = load_model(ck_brave, device)

    for t0 in picks:
        acts = actions[t0:t0 + H][None].astype(np.float32)
        dt = dream(mt, C.stack_context(fr_u8, np.array([t0 - ctxt + 1]), ctxt), acts, device)[0]
        db = dream(mb, C.stack_context(fr_u8, np.array([t0 - ctxb + 1]), ctxb), acts, device)[0]
        gif = []
        for h in range(H):
            real = to_img(frames[t0 + h + 1]); tim = to_img(dt[h]); bra = to_img(db[h])
            sep = np.full((64, 2, 3), 255, np.uint8)
            row = np.concatenate([real, sep, tim, sep, bra], axis=1)
            big = cv2.resize(row, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
            for x, lab in [(6, "REEL"), (64 * UP + 10, "TIMIDE"), (2 * (64 * UP + 10), "COURAGEUX")]:
                cv2.putText(big, lab, (x, big.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(big, f"h={h+1}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            gif.append(big)
        p = OUT / f"courage_{stem}_t{t0}.gif"
        imageio.mimsave(p, gif, fps=FPS, loop=0)
        # strip figee h=1,4,8,12,16
        steps = [1, 4, 8, 12, 16]
        rows = []
        for label, src in [("REEL", frames[t0 + 1:]), ("TIMIDE", dt), ("COURAGEUX", db)]:
            if label == "REEL":
                row = np.concatenate([to_img(frames[t0 + s]) for s in steps], axis=1)
            else:
                row = np.concatenate([to_img(src[s - 1]) for s in steps], axis=1)
            rows.append(row)
        strip = cv2.resize(np.concatenate(rows, axis=0), None, fx=UP, fy=UP,
                           interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"strip_courage_{stem}_t{t0}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(f"-> {p.name}")


if __name__ == "__main__":
    main()
