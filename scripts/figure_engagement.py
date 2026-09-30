"""
La courbe d'engagement.

Pour chaque pas h du reve, on mesure de combien le BLOC VISE (patch croix) a
change par rapport a l'instant du coup de pioche. Moyenne sur tous les
evenements de minage d'episodes JAMAIS VUS.

  - REEL      : monte franchement (le bloc casse)
  - TIMIDE    : reste plat (le reve fige la scene)
  - COURAGEUX : monte (le reve s'engage)

Deux panneaux : domaine DENSE (strip) vs domaine RARE (nvcave) -> montre que
le meme modele s'engage la ou il a vu l'evenement souvent, et se fige ailleurs.

Lance : .venv\\Scripts\\python.exe scripts\\figure_engagement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

H = 20
PATCH = (26, 38)
MAX_EV = 120
MODELS = {"timide (baseline)": ROOT / "outputs" / "v6-nvcave" / "model_best.pt",
          "courageux (in-domain)": ROOT / "outputs" / "v13b-indomain" / "model_best.pt"}
PANELS = [("domaine DENSE - strip (~80 % minage)", ["strip_051", "strip_052"]),
          ("domaine RARE - nvcave (~35 % minage)", ["nvcave_008", "nvcave_026"])]


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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (title, stems) in zip(axes, PANELS):
        acc = {"REEL": [], **{n: [] for n in MODELS}}
        for stem in stems:
            fr, actions, ok = load_ep(stem)
            tap = actions[:, 6] > 0.5
            evs = [t for t in range(6, len(actions) - H - 2)
                   if tap[t] and not tap[t - 1] and tap[t:t + 6].all() and ok[t - 5:t + H].all()]
            if len(evs) > MAX_EV:
                evs = [evs[i] for i in np.linspace(0, len(evs) - 1, MAX_EV).astype(int)]
            if not evs:
                continue
            acc["REEL"].append(real_curve(fr, evs))
            for n, (m, ctx) in loaded.items():
                acc[n].append(curves(m, ctx, fr, actions, evs, device))
        hs = np.arange(1, H + 1)
        styles = {"REEL": ("k", "-", 2.8), "timide (baseline)": ("tab:gray", "--", 2.2),
                  "courageux (in-domain)": ("tab:green", "-", 2.6)}
        for n, arrs in acc.items():
            if not arrs:
                continue
            m = np.concatenate(arrs).mean(axis=0)
            c, ls, lw = styles[n]
            ax.plot(hs, m, color=c, ls=ls, lw=lw, label=n)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("pas du reve h")
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("changement du bloc vise (0 = scene figee)")
    fig.suptitle("Le reve s'engage-t-il sur l'evenement ?  -  moyenne sur episodes JAMAIS VUS",
                 fontsize=13)
    fig.tight_layout()
    out = ROOT / "outputs" / "figure_engagement.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
