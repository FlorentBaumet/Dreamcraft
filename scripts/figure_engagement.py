"""
La courbe d'engagement.

Pour chaque pas h du reve, on mesure de combien le BLOC VISE (patch croix) a
change par rapport a l'instant du coup de pioche. Moyenne sur tous les
evenements de minage d'episodes JAMAIS VUS.

  - REEL      : monte franchement (le bloc casse)
  - TIMIDE    : reste plat (le reve fige la scene)
  - COURAGEUX : monte (le reve s'engage)

Deux domaines : DENSE (strip) et RARE (nvcave). Ecrit les courbes moyennes dans
outputs/diag/engagement.json ; la figure est dessinee par scripts/figures_readme.py.

Lance : .venv\\Scripts\\python.exe scripts\\figure_engagement.py
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
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

H = 20
PATCH = (26, 38)
MAX_EV = 120
OUT = ROOT / "outputs" / "diag" / "engagement.json"
MODELS = {"timide": ROOT / "outputs" / "v6-nvcave" / "model_best.pt",
          "nourri": ROOT / "outputs" / "v13b-indomain" / "model_best.pt"}
PANELS = [("dense", ["strip_051", "strip_052"]),       # strip, ~80 % de minage
          ("rare", ["nvcave_008", "nvcave_026"])]      # grottes, ~35 % de minage


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
def curves(model, ctx, fr_u8, actions, evs, device):
    """Retourne (n_ev, H) : changement du patch vise a chaque pas du reve."""
    y0, y1 = PATCH
    frames = fr_u8.astype(np.float32) / 255.0
    out = np.zeros((len(evs), H))
    for i in range(0, len(evs), 64):
        b = np.array(evs[i:i + 64])
        x = torch.from_numpy(C.stack_context(fr_u8, b - ctx + 1, ctx)).to(device).float() / 255.0
        acts = np.stack([actions[t:t + H] for t in b]).astype(np.float32)
        for h in range(H):
            pred = model(x, torch.from_numpy(acts[:, h]).to(device))
            p = pred.cpu().numpy()
            for j, t in enumerate(b):
                ref = frames[t][:, y0:y1, y0:y1]
                out[i + j, h] = np.abs(p[j][:, y0:y1, y0:y1] - ref).mean()
            x = C.roll_context(x, pred)
    return out


def real_curve(fr_u8, evs):
    y0, y1 = PATCH
    frames = fr_u8.astype(np.float32) / 255.0
    out = np.zeros((len(evs), H))
    for j, t in enumerate(evs):
        ref = frames[t][:, y0:y1, y0:y1]
        out[j] = np.abs(frames[t + 1:t + 1 + H][:, :, y0:y1, y0:y1] - ref).mean(axis=(1, 2, 3))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {n: load_model(p, device) for n, p in MODELS.items()}
    res = {}
    for title, stems in PANELS:
        acc = {"reel": [], **{n: [] for n in MODELS}}
        for stem in stems:
            fr, actions, ok = load_ep(stem)
            tap = actions[:, 6] > 0.5
            evs = [t for t in range(6, len(actions) - H - 2)
                   if tap[t] and not tap[t - 1] and tap[t:t + 6].all() and ok[t - 5:t + H].all()]
            if len(evs) > MAX_EV:
                evs = [evs[i] for i in np.linspace(0, len(evs) - 1, MAX_EV).astype(int)]
            if not evs:
                continue
            acc["reel"].append(real_curve(fr, evs))
            for n, (m, ctx) in loaded.items():
                acc[n].append(curves(m, ctx, fr, actions, evs, device))
        res[title] = {n: np.concatenate(a).mean(axis=0).tolist() for n, a in acc.items() if a}
        res[title]["evenements"] = int(sum(len(a) for a in acc["reel"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {OUT.relative_to(ROOT)}  (figure : scripts/figures_readme.py)")


if __name__ == "__main__":
    main()
