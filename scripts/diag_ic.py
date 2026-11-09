"""
Intervalles de confiance de diag_utilite.py, et test de son label.

Deux questions :
  1. L'AUC du score jumeau est-elle vraiment sous 50 % ? (bootstrap sur les coups)
  2. Le label "le bloc vise a bouge en 12 frames" est-il pollue par la camera du
     joueur ? On refait la mesure sur le quart des coups ou la camera bouge le
     moins, et on regarde ce que l'amplitude de la camera predit a elle seule.

Bootstrap sur les coups : il ignore la dependance a l'interieur d'un episode,
donc ces intervalles sont des bornes basses de l'incertitude reelle.

Lance : .venv\\Scripts\\python.exe scripts\\diag_ic.py
"""
from __future__ import annotations

import importlib.util as ilu
from pathlib import Path

import numpy as np
import torch

_s = ilu.spec_from_file_location("du", Path(__file__).parent / "diag_utilite.py")
U = ilu.module_from_spec(_s)
_s.loader.exec_module(U)

N_BOOT = 2000
MODELS = ["v6 timide", "v13b courageux"]


def ic(score, label, rng):
    n = len(score)
    vals = [U.auc(score[i], label[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return U.auc(score, label), lo, hi


def fmt(t):
    return f"{t[0]:.1%} [{t[1]:.1%}, {t[2]:.1%}]"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    y0, y1 = U.PATCH
    for title, stems in U.PANELS:
        print(f"\n===== {title} =====")
        rows = []
        for stem in stems:
            fr, acts, ok = U.load_ep(stem)
            tap = acts[:, 6] > 0.5
            ev = [t for t in range(8, len(acts) - max(U.HREAL, U.HDREAM) - 2)
                  if tap[t] and not tap[t - 1] and ok[t - 8:t + U.HREAL + 1].all()]
            if len(ev) > U.MAXN:
                ev = [ev[i] for i in np.linspace(0, len(ev) - 1, U.MAXN).astype(int)]
            f = fr.astype(np.float32) / 255.
            eff = np.array([np.abs(f[t + U.HREAL][:, y0:y1, y0:y1]
                                   - f[t][:, y0:y1, y0:y1]).mean() for t in ev])
            cam = np.array([np.abs(acts[t:t + U.HREAL, 8:10]).mean() for t in ev])
            rows.append((fr, acts, ev, eff, cam))
        eff = np.concatenate([r[3] for r in rows])
        cam = np.concatenate([r[4] for r in rows])
        tex = np.concatenate([U.b0_texture(r[0], r[2]) for r in rows])
        lab = eff > np.median(eff)
        still = cam < np.quantile(cam, 0.25)     # le quart des coups ou la camera bouge le moins
        print(f"{len(eff)} coups | camera mediane {np.median(cam):.3f} | quart le plus calme : "
              f"{still.sum()} coups, camera < {cam[still].max():.3f}")
        # labels recalcules dans le sous-ensemble calme
        lab_s = eff[still] > np.median(eff[still]) if still.sum() > 40 else None
        scores = {"amplitude camera (sans modele)": cam,
                  "contraste du bloc (sans modele)": tex,
                  "ratio sur le passe (sans modele)":
                      np.concatenate([U.b0_passe(r[0], r[2]) for r in rows])}
        for name in MODELS:
            m, nctx = U.load_model(U.MODELS[name], device)
            scores[f"{name} : jumeau"] = np.concatenate(
                [U.twin_scores(m, nctx, r[0], r[1], r[2], device) for r in rows])
            scores[f"{name} : ratio"] = np.concatenate(
                [U.ratio_scores(m, nctx, r[0], r[1], r[2], device) for r in rows])
        print(f"{'score':<36}{'tous les coups':<28}{'quart camera calme':<28}{'stratifie texture'}")
        for k, s in scores.items():
            a = fmt(ic(s, lab, rng))
            b = fmt(ic(s[still], lab_s, rng)) if lab_s is not None else "n/a"
            c = U.auc_strat(s, lab, tex) if not k.startswith("contraste") else float("nan")
            print(f"{k:<36}{a:<28}{b:<28}{c:.1%}")


if __name__ == "__main__":
    main()
