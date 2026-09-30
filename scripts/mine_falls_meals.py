"""
Mineur corpus-entier pour les regles #5 (chutes) et #10 (repas).

Chutes : detecteur bon marche (correlation de profils de lignes -> decalage
vertical entier par frame), runs de descente soutenue, confirmation flux
optique Farneback sur les candidats seulement. Seuils calibres sur la
distribution empirique (et non fixes a l'aveugle -- lecon du n=0).

Repas : episodes VPT uniquement (actions reelles), 'utiliser' maintenu >=20
pas sans taper, ET changement de la zone faim (filtre les arcs/boucliers).

Sorties : outputs/mine_events/{falls.json, meals.json} + stats console
Lance   : .venv\\Scripts\\python.exe scripts\\mine_falls_meals.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mine_events"

SHIFT_MAX = 4
FALL_RUN = 6
HUNGER = (slice(52, 56), slice(33, 50))


def row_profiles(frames_u8):
    """(N,64,64,3) -> profils de lignes gris (N,40) float32.

    Lignes 8..48 UNIQUEMENT : le HUD (fixe) ancrait la correlation a zero
    et le ciel (uniforme) la diluait -- bug du premier run (0 chute)."""
    return frames_u8[:, 8:48].mean(axis=(2, 3)).astype(np.float32)


def vertical_shifts(prof):
    """Decalage vertical entier estime entre frames consecutives. (N-1,)"""
    n = prof.shape[0]
    shifts = np.zeros(n - 1, dtype=np.int8)
    for t in range(n - 1):
        a, b = prof[t], prof[t + 1]
        best, arg = -1e9, 0
        m = len(a)
        for s in range(-SHIFT_MAX, SHIFT_MAX + 1):
            if s >= 0:
                c = float(np.dot(a[s:], b[: m - s])) / (m - abs(s))
            else:
                c = float(np.dot(a[: m + s], b[-s:])) / (m - abs(s))
            if c > best:
                best, arg = c, s
        shifts[t] = arg
    return shifts


def confirm_fall(frames_u8, t0, t1):
    """Confirmation Farneback sur le segment candidat."""
    g0 = cv2.cvtColor(frames_u8[t0], cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(frames_u8[min(t0 + 3, t1)], cv2.COLOR_RGB2GRAY)
    f = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 2, 9, 3, 5, 1.1, 0)
    return float(f[8:48, 8:56, 1].mean()) < -0.8 * 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vpt_eps = sorted((ROOT / "data" / "processed").glob("*.npz"))
    all_eps = vpt_eps + sorted((ROOT / "data" / "processed_youtube").glob("*.npz"))

    falls, meals = [], []
    shift_stats = []

    for i, p in enumerate(all_eps):
        d = np.load(p)
        frames = d["frames"]
        shifts = vertical_shifts(row_profiles(frames))
        shift_stats.append(np.percentile(shifts, [1, 5]))

        # runs de descente : shift <= -1 soutenu
        neg = shifts <= -1
        t = 0
        while t < len(neg) - FALL_RUN:
            if neg[t: t + FALL_RUN].all():
                end = t + FALL_RUN
                while end < len(neg) and neg[end]:
                    end += 1
                if confirm_fall(frames, t, end):
                    falls.append({"ep": p.stem, "debut": int(t), "fin": int(end),
                                  "duree": int(end - t)})
                t = end + 10
            else:
                t += 1

        # repas : actions reelles VPT seulement
        if p in vpt_eps:
            actions = d["actions"]
            use = (actions[:, 7] > 0.5) & (actions[:, 6] < 0.5)
            dd = np.diff(np.concatenate([[0], use.astype(int), [0]]))
            starts, ends = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)
            for s, e in zip(starts, ends):
                if e - s < 20 or s < 4 or e + 4 >= len(frames):
                    continue
                pre = frames[s - 2][HUNGER[0], HUNGER[1]].astype(np.float32) / 255
                post = frames[e + 2][HUNGER[0], HUNGER[1]].astype(np.float32) / 255
                delta = float(np.abs(post - pre).mean())
                if delta > 0.02:
                    meals.append({"ep": p.stem, "debut": int(s), "fin": int(e),
                                  "delta_faim": round(delta, 4)})

        if (i + 1) % 60 == 0:
            print(f"{i+1}/{len(all_eps)} | chutes {len(falls)} | repas {len(meals)}", flush=True)

    (OUT / "falls.json").write_text(json.dumps(falls, indent=2), encoding="utf-8")
    (OUT / "meals.json").write_text(json.dumps(meals, indent=2), encoding="utf-8")

    ss = np.array(shift_stats)
    print(f"\ncalibration decalages : p1 median {np.median(ss[:,0]):.1f}, "
          f"p5 median {np.median(ss[:,1]):.1f}")
    durees = [f["duree"] for f in falls]
    print(f"\n=== CHUTES : {len(falls)} confirmees "
          f"(duree mediane {np.median(durees) if durees else '-'} pas) ===")
    print(f"=== REPAS : {len(meals)} avec effet visible sur la barre de faim ===")
    verdict = lambda n: "TESTABLE" if n >= 30 else ("LIMITE" if n >= 10 else "INSUFFISANT")
    print(f"regle #5 chute : {verdict(len(falls))} | regle #10 faim : {verdict(len(meals))}")


if __name__ == "__main__":
    main()
