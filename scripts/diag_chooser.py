"""
Diagnostic du chooseur : quand le joueur reel se met a miner, l'agent choisit-il
"taper" ?

Protocole (episodes JAMAIS VUS) :
  - on repere les instants ou le joueur reel DECLENCHE un coup de pioche,
  - a cet instant precis on donne au chooseur le meme contexte,
  - bonne reponse = il choisit un plan qui tape.
  - controle : memes mesures sur des instants tires au hasard.

On balaye deux axes :
  - la PORTEE H du reve (4..20) : l'evenement doit tenir DANS le reve,
  - la FORME du score : brut vs contrastif (on retranche le reve "ne rien faire",
    ce qui annule le changement ambiant et le mouvement de camera).

Lance : .venv\Scripts\python.exe scripts\diag_chooser.py
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

import importlib.util as _ilu
_s = _ilu.spec_from_file_location("al", Path(__file__).parent / "agent_loop.py")
_al = _ilu.module_from_spec(_s); _s.loader.exec_module(_al)

HS = [4, 8, 12, 16, 20]
EPS = ["strip_051", "strip_052", "nvcave_008"]
MAXN = 60
TAPE = {"taper", "taper_avancer"}
CKPT = ROOT / "outputs" / "v13b-indomain" / "model_best.pt"
GOAL = ROOT / "outputs" / "goal_detector" / "model_best.pt"


def load_ep(stem):
    for base in ["processed", "processed_youtube"]:
        p = ROOT / "data" / base / f"{stem}.npz"
        if p.exists():
            return C.load_episode_u8(p)
    raise FileNotFoundError(stem)


@torch.no_grad()
def scores(pol, ctx_u8_batch, h, device):
    """(B, n_plans) pour chaque variante de score. ctx_u8_batch : (B,CTX,64,64,3)."""
    n = len(pol.names)
    B = len(ctx_u8_batch)
    recent = ctx_u8_batch[:, -pol.nctx:]
    stack = recent.transpose(0, 1, 4, 2, 3).reshape(B, -1, 64, 64)
    x = torch.from_numpy(np.repeat(stack, n, 0)).to(device).float() / 255.0
    start = x[:, -3:].clone()
    plans = torch.from_numpy(np.tile(pol.plans, (B, 1, 1))).to(device)
    dreams = []
    for step in range(h):
        pred = pol.model(x, plans[:, step])
        dreams.append(pred)
        x = C.roll_context(x, pred)
    d = (dreams[-1] - start).abs().mean(1)
    c = d[:, pol.cm].mean(1).view(B, n)
    p = d[:, ~pol.cm].mean(1).view(B, n)
    ratio = (c / (c + p + 1e-6))
    i_rien = pol.names.index("rien")
    contrast = c - c[:, i_rien:i_rien + 1]        # "en plus de ne rien faire"
    seq = torch.stack([start] + dreams, dim=1)
    gl = torch.zeros_like(ratio)
    if pol.goal is not None and seq.shape[1] >= 5:
        wins = torch.stack([seq[:, i:i + 5].flatten(1, 2) for i in range(seq.shape[1] - 4)], 1)
        bb, w = wins.shape[:2]
        gl = torch.sigmoid(pol.goal(wins.flatten(0, 1))).view(bb, w)
        gl = gl.topk(min(3, w), dim=1).values.mean(1).view(B, n)
    return {"ratio (actuel)": ratio, "ratio+goal (actuel)": ratio + gl,
            "contrastif": contrast, "contrastif+goal": contrast + gl}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pol = _al.ChooserPolicy(CKPT, GOAL, device, max(HS))
    rng = np.random.default_rng(0)
    ev_ctx, rd_ctx = [], []
    for stem in EPS:
        fr, act, ok = load_ep(stem)
        tap = act[:, 6] > 0.5
        ev = [t for t in range(8, len(act) - 25)
              if tap[t] and not tap[t - 1] and tap[t:t + 6].all() and ok[t - 6:t + 22].all()]
        nt = [t for t in range(8, len(act) - 25)
              if not tap[t - 4:t + 6].any() and ok[t - 6:t + 22].all()]
        for pool, dst in [(ev, ev_ctx), (nt, rd_ctx)]:
            if not pool:
                continue
            sel = rng.choice(pool, size=min(MAXN, len(pool)), replace=False)
            dst += [fr[t - 3:t + 1].transpose(0, 2, 3, 1) for t in sel]
    ev_ctx, rd_ctx = np.stack(ev_ctx), np.stack(rd_ctx)
    print(f"{len(ev_ctx)} instants de MINAGE reel, {len(rd_ctx)} instants de controle\n")

    rows = {}
    for h in HS:
        pol.h = h
        for label, arr in [("minage", ev_ctx), ("controle", rd_ctx)]:
            acc = {}
            for i in range(0, len(arr), 16):
                for k, v in scores(pol, arr[i:i + 16], h, device).items():
                    acc.setdefault(k, []).append(v.cpu().numpy())
            for k, v in acc.items():
                pick = np.concatenate(v).argmax(1)
                rate = np.mean([pol.names[j] in TAPE for j in pick])
                rows.setdefault(k, {}).setdefault(h, {})[label] = rate

    print(f"{'score':<22}" + "".join(f"  H={h:<10}" for h in HS))
    print(f"{'':<22}" + "".join(f"  {'mine/ctrl':<10}" for _ in HS))
    for k, per_h in rows.items():
        line = f"{k:<22}"
        for h in HS:
            line += f"  {per_h[h]['minage']:.0%}/{per_h[h]['controle']:.0%}".ljust(12)
        print(line)
    print("\nlecture : 'mine' = % ou l'agent choisit de taper quand le joueur reel tape")
    print("          'ctrl' = idem sur des instants sans minage (doit etre plus bas)")
    print("          l'ecart mine-ctrl = le vrai signal de decision")


if __name__ == "__main__":
    main()
