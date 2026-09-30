"""
MINEUR D'EVENEMENTS RARES -- trouve dans le corpus les instances des regles
du jeu qui demandent des evenements peu frequents.

Signatures cherchees (action + visuel, cheap) :
  #2 SEAU  : impulsion 'utiliser' + grosse zone eau/lave au centre qui
             apparait/disparait en <=2 frames.
  #4 LIT   : impulsion 'utiliser' -> ecran quasi noir >=8 frames -> retour
             lumineux (le saut de nuit).
  #6 MOB   : impulsions 'taper' COURTES (1-4 frames, un coup, pas du minage)
             -> candidats seulement, a valider visuellement (lecon 8).
  #9 BATEAU: impulsion 'utiliser' avec beaucoup d'eau en bas d'image
             -> candidats seulement.

Sorties : outputs/mine_events/{counts.json, hits_*.json, sheet_*.png}
Lance   : .venv\\Scripts\\python.exe scripts\\mine_events.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mine_events"

WATER_BGR = dict(b_min=0.35, b_over_r=0.06)   # eau : bleu dominant
LAVA = dict(r_min=0.45, r_over_b=0.12)        # lave : orange dominant


def pulses(track: np.ndarray, min_len=1, max_len=4):
    """Debuts des runs de 1 dans track, de longueur [min_len, max_len]."""
    t = track.astype(int)
    d = np.diff(np.concatenate([[0], t, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [s for s, e in zip(starts, ends) if min_len <= e - s <= max_len]


def water_share(img, y0=0, y1=64):
    f = img[y0:y1].astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float(((b > WATER_BGR["b_min"]) & (b > r + WATER_BGR["b_over_r"])).mean())


def lava_share(img, y0=20, y1=44, x0=20, x1=44):
    f = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float(((r > LAVA["r_min"]) & (r > b + LAVA["r_over_b"])).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps = sorted((ROOT / "data" / "processed").glob("*.npz")) \
        + sorted((ROOT / "data" / "processed_youtube").glob("*.npz"))
    counts = {"seau": 0, "lit": 0, "mob_candidats": 0, "bateau_candidats": 0}
    hits = {k: [] for k in counts}

    for i, p in enumerate(eps):
        d = np.load(p)
        frames, actions = d["frames"], d["actions"]
        n = len(actions)
        luma = frames.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
        use = actions[:, 7] > 0.5
        tap = actions[:, 6] > 0.5

        # --- #2 SEAU : use-pulse + delta eau/lave central brutal ---
        for t in pulses(use, 1, 3):
            if t + 3 >= n:
                continue
            w0 = water_share(frames[t], 20, 44)
            w1 = water_share(frames[t + 2], 20, 44)
            l0, l1 = lava_share(frames[t]), lava_share(frames[t + 2])
            if abs(w1 - w0) > 0.12 or abs(l1 - l0) > 0.12:
                counts["seau"] += 1
                hits["seau"].append({"ep": p.stem, "t": int(t)})

        # --- #4 LIT : use-pulse -> noir soutenu -> retour lumineux ---
        for t in pulses(use, 1, 3):
            if t + 60 >= n:
                continue
            seg = luma[t + 2: t + 40]
            dark = seg < 0.06
            if dark.sum() >= 8:
                after = luma[t + 12: t + 100]
                if len(after) and after.max() > 0.30 and luma[max(t - 5, 0):t].mean() < 0.35:
                    counts["lit"] += 1
                    hits["lit"].append({"ep": p.stem, "t": int(t)})

        # --- #6 MOB : coups courts (pas du minage) -> candidats ---
        for t in pulses(tap, 1, 4):
            counts["mob_candidats"] += 1
            if len(hits["mob_candidats"]) < 4000:
                hits["mob_candidats"].append({"ep": p.stem, "t": int(t)})

        # --- #9 BATEAU : use-pulse avec beaucoup d'eau en bas ---
        for t in pulses(use, 1, 3):
            if water_share(frames[t], 40, 64) > 0.25:
                counts["bateau_candidats"] += 1
                hits["bateau_candidats"].append({"ep": p.stem, "t": int(t)})

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(eps)} eps | {counts}")

    (OUT / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    for k, v in hits.items():
        (OUT / f"hits_{k}.json").write_text(json.dumps(v, indent=2), encoding="utf-8")

    # planches de validation visuelle (lecon 8) : 12 exemples par categorie
    rng = np.random.default_rng(0)
    for k in ["seau", "lit", "mob_candidats", "bateau_candidats"]:
        if not hits[k]:
            continue
        sel = rng.permutation(len(hits[k]))[:12]
        tiles = []
        cache = {}
        for si in sel:
            h = hits[k][si]
            if h["ep"] not in cache:
                for base in ["processed", "processed_youtube"]:
                    q = ROOT / "data" / base / f"{h['ep']}.npz"
                    if q.exists():
                        cache[h["ep"]] = np.load(q)["frames"]
                        break
            fr = cache[h["ep"]]
            trio = np.concatenate([fr[max(h["t"]-1,0)], fr[min(h["t"]+1, len(fr)-1)],
                                   fr[min(h["t"]+3, len(fr)-1)]], axis=1)
            tiles.append(trio)
        rows = [np.concatenate(tiles[j:j+3], axis=1) for j in range(0, min(len(tiles), 12), 3)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]
        sheet = np.concatenate(rows, axis=0)
        big = cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"sheet_{k}.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

    print("\n=== VERDICT MINABILITE ===")
    for k, v in counts.items():
        verdict = "TESTABLE" if v >= 30 else ("LIMITE" if v >= 10 else "INSUFFISANT")
        print(f"  {k:<18} {v:6d}  -> {verdict}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
