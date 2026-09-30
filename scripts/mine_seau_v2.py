"""
Detecteur SEAU v2 (regle #2) -- resserre apres tri visuel du v1 (~50 % de faux).

Corrections v2 :
  a) le changement eau/lave doit etre un BLOB centre pres de la croix
     (centre de masse a <= 14 px du centre, taille >= 25 px),
  b) zone objet-en-main EXCLUE (bas-droite : reflets d'items enchantes),
  c) nettete temporelle : changement brutal (t-1 -> t+2) >= 3x la derive
     ambiante (t-6 -> t-3) -- filtre les passages devant l'eau,
  d) signe dominant : la zone changee est majoritairement apparue OU disparue.

Sortie : outputs/mine_events/{seau_v3.json, sheet_seau_v3.png}
Lance  : .venv\\Scripts\\python.exe scripts\\mine_seau_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mine_events"

CENTER = np.array([32.0, 32.0])
R_MAX = 14.0
BLOB_MIN = 25
SHARP = 3.0
HAND = (slice(38, 58), slice(42, 64))   # zone objet-en-main (exclue)


def liquid_mask(img_u8):
    """v3 : EAU seulement -- le masque 'lave' (teintes chaudes) attrapait
    torches et planches de bois (tri visuel v2, ~30 % de precision)."""
    f = img_u8.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = (b > 0.35) & (b > r + 0.06) & (b > g + 0.02)
    m[HAND] = False
    m[52:] = False           # HUD
    return m


def pulses(track, lo=1, hi=3):
    t = track.astype(int)
    d = np.diff(np.concatenate([[0], t, [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [a for a, b in zip(s, e) if lo <= b - a <= hi]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps = sorted((ROOT / "data" / "processed").glob("*.npz")) \
        + sorted((ROOT / "data" / "processed_youtube").glob("*.npz"))
    hits = []
    for i, p in enumerate(eps):
        d = np.load(p)
        frames, actions = d["frames"], d["actions"]
        n = len(actions)
        for t in pulses(actions[:, 7] > 0.5):
            if t < 7 or t + 3 >= n:
                continue
            m_pre = liquid_mask(frames[t - 1])
            m_post = liquid_mask(frames[t + 2])
            appeared = m_post & ~m_pre
            gone = m_pre & ~m_post
            blob = appeared if appeared.sum() >= gone.sum() else gone
            size = int(blob.sum())
            if size < BLOB_MIN:
                continue
            ys, xs = np.nonzero(blob)
            com = np.array([ys.mean(), xs.mean()])
            if np.linalg.norm(com - CENTER) > R_MAX:
                continue
            # nettete : derive ambiante avant l'impulsion
            drift = np.logical_xor(liquid_mask(frames[t - 6]), liquid_mask(frames[t - 3])).sum()
            if size < SHARP * max(drift, 4):
                continue
            dominant = max(appeared.sum(), gone.sum()) / max(appeared.sum() + gone.sum(), 1)
            if dominant < 0.7:
                continue
            # v3 : l'eau ONDULE -- la zone doit avoir varie temporellement AVANT
            # l'evenement (pour 'ramasse') ou APRES (pour 'pose')
            zone = blob
            if appeared.sum() >= gone.sum():
                anim = np.abs(frames[t + 3][zone].astype(np.float32)
                              - frames[t + 2][zone].astype(np.float32)).mean()
            else:
                anim = np.abs(frames[t - 2][zone].astype(np.float32)
                              - frames[t - 4][zone].astype(np.float32)).mean()
            if anim < 2.0:
                continue
            hits.append({"ep": p.stem, "t": int(t), "taille": size,
                         "sens": "pose" if appeared.sum() >= gone.sum() else "ramasse"})
        if (i + 1) % 80 == 0:
            print(f"{i+1}/{len(eps)} | seaux v2: {len(hits)}", flush=True)

    (OUT / "seau_v3.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")
    print(f"\n{len(hits)} evenements seau v3 (v2 : 184, v1 : 337)")

    # planche de verification
    rng = np.random.default_rng(1)
    sel = rng.permutation(len(hits))[:12]
    cache, tiles = {}, []
    for si in sel:
        h = hits[si]
        if h["ep"] not in cache:
            for base in ["processed", "processed_youtube"]:
                q = ROOT / "data" / base / f"{h['ep']}.npz"
                if q.exists():
                    cache[h["ep"]] = np.load(q)["frames"]
                    break
        fr = cache[h["ep"]]
        tiles.append(np.concatenate([fr[h["t"] - 1], fr[h["t"] + 1], fr[h["t"] + 3]], axis=1))
    if tiles:
        rows = [np.concatenate(tiles[j:j + 3], axis=1) for j in range(0, len(tiles), 3)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        big = cv2.resize(np.concatenate(rows, axis=0), None, fx=2, fy=2,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / "sheet_seau_v3.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        print("-> sheet_seau_v3.png (verification visuelle)")


if __name__ == "__main__":
    main()
