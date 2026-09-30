"""
Sonde regle #10 : "clic droit maintenu avec nourriture -> la barre
de faim remonte" -- testee sur les repas mines dans tout le corpus.

Pour chaque repas reel : le reve (seede juste avant, action 'utiliser'
maintenue) reproduit-il un changement de la zone faim comparable au reel ?
Controle : reves "null" seedes a des moments sans repas (la zone faim ne
doit PAS y changer -- sinon le detecteur mesure du bruit de reve).

Lance : .venv\\Scripts\\python.exe scripts\\probe_faim.py --models n=ckpt ...
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

OUT_DIR = ROOT / "outputs" / "regles_du_jeu"
HUNGER = (slice(52, 56), slice(33, 50))
H = 32
SEUIL_EFFET = 0.5   # le reve "nourrit" si delta >= 50% du delta reel du repas


def load_model(ckpt: Path, device):
    cfg = {"base": 32, "in_frames": 1}
    cf = ckpt.parent / "config.json"
    if cf.exists():
        cfg.update(json.loads(cf.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


@torch.no_grad()
def dream(model, ctx_u8, acts, device):
    x = torch.from_numpy(ctx_u8).to(device).float() / 255.0
    out = np.zeros((x.shape[0], acts.shape[1], 3, 64, 64), dtype=np.float32)
    for h in range(acts.shape[1]):
        pred = model(x, torch.from_numpy(acts[:, h]).to(device))
        out[:, h] = pred.cpu().numpy()
        x = C.roll_context(x, pred)
    return out


def hunger_delta(frame_a, frame_b):
    return float(np.abs(frame_b[:, HUNGER[0], HUNGER[1]]
                        - frame_a[:, HUNGER[0], HUNGER[1]]).mean())


def main():
    args = sys.argv[1:]
    sep = args.index("--models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for spec in args[sep + 1:]:
        name, ck = spec.split("=", 1)
        models[name] = load_model(Path(ck), device)

    meals = json.loads((ROOT / "outputs" / "mine_events" / "meals.json").read_text(encoding="utf-8"))
    by_ep = defaultdict(list)
    for m in meals:
        by_ep[m["ep"]].append(m)
    print(f"{len(meals)} repas dans {len(by_ep)} episodes")

    rng = np.random.default_rng(0)
    stats = {n: {"reel": [], "reve": [], "null": [], "paire": []} for n in models}

    for ep, evs in by_ep.items():
        frames_u8, actions, ok = C.load_episode_u8(ROOT / "data" / "processed" / f"{ep}.npz")
        frames = frames_u8.astype(np.float32) / 255.0
        n = frames.shape[0]
        seeds, deltas_reels = [], []
        for m in evs:
            s = m["debut"]
            if s < 6 or s + H + 2 >= n or not ok[s - 4: s + H].all():
                continue
            seeds.append(s)
            deltas_reels.append(hunger_delta(frames[s - 2], frames[min(m["fin"] + 2, n - 1)]))
        if not seeds:
            continue
        seeds = np.array(seeds)
        acts = np.stack([actions[s: s + H] for s in seeds]).astype(np.float32)
        # moments null (pas de use), memes episodes
        nulls = []
        while len(nulls) < len(seeds):
            t0 = int(rng.integers(6, n - H - 2))
            if actions[t0: t0 + H, 7].max() < 0.5 and ok[t0 - 4: t0 + H].all():
                nulls.append(t0)
        nulls = np.array(nulls)
        acts_null = np.stack([actions[t: t + H] for t in nulls]).astype(np.float32)

        # contrefactuel apparie : memes graines, 'utiliser' coupe
        acts_sans = acts.copy()
        acts_sans[:, :, 7] = 0.0
        for name, (model, ctx) in models.items():
            ctxs = C.stack_context(frames_u8, seeds - ctx + 1, ctx)
            dr = dream(model, ctxs, acts, device)
            ds = dream(model, ctxs, acts_sans, device)
            dn = dream(model, C.stack_context(frames_u8, nulls - ctx + 1, ctx), acts_null, device)
            for j, s in enumerate(seeds):
                stats[name]["reel"].append(deltas_reels[j])
                stats[name]["reve"].append(hunger_delta(frames[s - 1], dr[j][-1]))
                # la fonte s'annule dans la paire : seul l'effet de MANGER reste
                stats[name]["paire"].append(hunger_delta(ds[j][-1], dr[j][-1]))
            for j, t in enumerate(nulls):
                stats[name]["null"].append(hunger_delta(frames[t - 1], dn[j][-1]))

    print("\n=== REGLE #10 : manger -> la barre de faim change ===")
    results = {}
    for name, st in stats.items():
        reel = np.array(st["reel"])
        reve = np.array(st["reve"])
        nul = np.array(st["null"])
        paire = np.array(st["paire"])
        effet = float((reve >= SEUIL_EFFET * reel).mean())
        specificite = float(np.median(nul) / max(np.median(reve), 1e-6))
        # effet apparie : manger vs ne pas manger, meme graine -> la fonte s'annule
        effet_paire = float((paire > np.median(nul) * 0.3).mean())
        print(f"{name:8s}: n={len(reel)} | reel {np.median(reel):.4f} | reve {np.median(reve):.4f} | "
              f"null {np.median(nul):.4f} | PAIRE {np.median(paire):.4f}")
        print(f"          -> effet apparie (manger vs pas, meme graine) : "
              f"median {np.median(paire):.4f}, present dans {effet_paire:.0%} des paires")
        results[name] = {"n": int(len(reel)), "delta_reel_med": float(np.median(reel)),
                         "delta_reve_med": float(np.median(reve)),
                         "delta_null_med": float(np.median(nul)),
                         "delta_paire_med": float(np.median(paire)),
                         "taux_effet_naif": effet, "taux_effet_paire": effet_paire,
                         "bruit_ratio": specificite}

    out = OUT_DIR / "results_faim.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
