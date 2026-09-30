"""
Sonde regle #5 : "tomber de >3 blocs -> degats" -- sur les chutes minees.

Pour chaque chute reelle : reve seede EN PLEINE CHUTE (le contexte contient
le debut de la chute), actions reelles. On mesure :
  - le reve ATTERRIT-il (le flux vertical s'arrete, comme en vrai) ?
  - le flash rouge de degats apparait-il apres l'atterrissage reve ?
Controles : taux de flash reel (l'eau annule les degats -> pas 100 %) et
taux de flash spontane dans des reves sans chute (fausse alarme).

Caveat declare : la plupart des chutes sont dans des episodes d'ENTRAINEMENT
(les 4 held-out n'en avaient aucune) -> taux possiblement flattes.

Lance : .venv\\Scripts\\python.exe scripts\\probe_chute.py --models n=ckpt ...
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

OUT_DIR = ROOT / "outputs" / "regles_du_jeu"
H = 16
FLASH_DELTA = 0.03
DY_FALL = -1.0
DY_STOP = -0.4


def load_model(ckpt: Path, device):
    cfg = {"base": 32, "in_frames": 1}
    cf = ckpt.parent / "config.json"
    if cf.exists():
        cfg.update(json.loads(cf.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


def find_ep(stem):
    for base in ["processed", "processed_youtube"]:
        p = ROOT / "data" / base / f"{stem}.npz"
        if p.exists():
            return p
    return None


@torch.no_grad()
def dream(model, ctx_u8, acts, device):
    x = torch.from_numpy(ctx_u8).to(device).float() / 255.0
    out = np.zeros((x.shape[0], acts.shape[1], 3, 64, 64), dtype=np.float32)
    for h in range(acts.shape[1]):
        pred = model(x, torch.from_numpy(acts[:, h]).to(device))
        out[:, h] = pred.cpu().numpy()
        x = C.roll_context(x, pred)
    return out


def redness(fr):
    r, g, b = fr[0, :50], fr[1, :50], fr[2, :50]
    return float((r - (g + b) / 2).mean())


def dy_seq(frames_f32):
    g = [(cv2.cvtColor((f.transpose(1, 2, 0) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2GRAY)) for f in frames_f32]
    return [float(cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None,
                  0.5, 2, 9, 3, 5, 1.1, 0)[8:48, 8:56, 1].mean())
            for i in range(len(g) - 1)]


def main():
    args = sys.argv[1:]
    sep = args.index("--models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for spec in args[sep + 1:]:
        name, ck = spec.split("=", 1)
        models[name] = load_model(Path(ck), device)

    falls = json.loads((ROOT / "outputs" / "mine_events" / "falls.json").read_text(encoding="utf-8"))
    by_ep = defaultdict(list)
    for f in falls:
        by_ep[f["ep"]].append(f)
    print(f"{len(falls)} chutes dans {len(by_ep)} episodes")

    rng = np.random.default_rng(0)
    real = {"n": 0, "flash": 0}
    dstat = {n: {"n": 0, "atterrit": 0, "flash": 0, "null_flash": 0, "null_n": 0}
             for n in models}

    for ep, evs in by_ep.items():
        p = find_ep(ep)
        frames_u8, actions, ok = C.load_episode_u8(p)
        frames = frames_u8.astype(np.float32) / 255.0
        n = frames.shape[0]

        seeds = []
        for f in evs:
            t0, land = f["debut"], f["fin"]
            if t0 < 6 or t0 + 2 + H >= n or land - t0 - 2 >= H - 2:
                continue
            base = np.median([redness(frames[j]) for j in range(max(t0 - 8, 0), t0)])
            post = max(redness(frames[j]) for j in range(land, min(land + 4, n)))
            real["n"] += 1
            if post - base > FLASH_DELTA:
                real["flash"] += 1
            seeds.append((t0 + 2, land, base))
        if not seeds:
            continue

        starts = np.array([s for s, _, _ in seeds])
        acts = np.stack([actions[s: s + H] for s in starts]).astype(np.float32)
        nulls = []
        while len(nulls) < len(seeds):
            t0 = int(rng.integers(6, n - H - 2))
            if ok[t0 - 4: t0 + H].all():
                nulls.append(t0)
        nulls = np.array(nulls)
        acts_null = np.stack([actions[t: t + H] for t in nulls]).astype(np.float32)

        for name, (model, ctx) in models.items():
            dr = dream(model, C.stack_context(frames_u8, starts - ctx + 1, ctx), acts, device)
            dn = dream(model, C.stack_context(frames_u8, nulls - ctx + 1, ctx), acts_null, device)
            for j, (s, land, base) in enumerate(seeds):
                dys = dy_seq(dr[j])
                landed = None
                for h in range(1, len(dys)):
                    if dys[h - 1] < DY_FALL and dys[h] > DY_STOP:
                        landed = h
                        break
                dstat[name]["n"] += 1
                if landed is not None:
                    dstat[name]["atterrit"] += 1
                    post = max(redness(dr[j][h]) for h in range(landed, min(landed + 4, H)))
                    if post - base > FLASH_DELTA:
                        dstat[name]["flash"] += 1
            for j in range(len(nulls)):
                base_n = redness(frames[nulls[j] - 1])
                post_n = max(redness(dn[j][h]) for h in range(H))
                dstat[name]["null_n"] += 1
                if post_n - base_n > FLASH_DELTA:
                    dstat[name]["null_flash"] += 1

    print("\n=== REGLE #5 : chute -> degats (flash rouge) ===")
    pr = real["flash"] / max(real["n"], 1)
    print(f"realite : {real['n']} chutes exploitables | flash apres atterrissage {pr:.0%} "
          f"(<100% attendu : eau)")
    results = {"realite": {"n": real["n"], "taux_flash": pr}, "reves": {}}
    for name, s in dstat.items():
        at = s["atterrit"] / max(s["n"], 1)
        fl = s["flash"] / max(s["atterrit"], 1)
        nf = s["null_flash"] / max(s["null_n"], 1)
        print(f"{name:8s}: atterrit {at:.0%} | flash si atterri {fl:.0%} | "
              f"flash spontane (null) {nf:.0%} (n={s['n']})")
        results["reves"][name] = {"n": s["n"], "atterrit": at, "flash_si_atterri": fl,
                                  "flash_spontane": nf}

    (OUT_DIR / "results_chute.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {(OUT_DIR / 'results_chute.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
