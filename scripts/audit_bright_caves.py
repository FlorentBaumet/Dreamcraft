"""
Audit affine : y a-t-il dans le corpus des joueurs en FULL BRIGHTNESS ?
(= des frames SOUTERRAINES et pourtant CLAIRES)

Heuristiques 64px :
  - frame "en surface" si du ciel est visible dans le tiers haut de l'image
    (pixels bleus clairs ou tres blancs) ;
  - sinon "souterraine" ;
  - signature full-brightness : part souterraine >= 20 % ET luminance mediane
    des frames souterraines >= 0.22 (defaut Minecraft : ~0.05-0.12).

Sortie : outputs/audit_bright_caves.json + classement console.
Lance  : .venv\\Scripts\\python.exe scripts\\audit_bright_caves.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def sky_share(frames: np.ndarray) -> np.ndarray:
    """Part de pixels 'ciel' dans le tiers haut. frames uint8 (N,64,64,3)."""
    top = frames[:, :21].astype(np.float32) / 255.0     # (N,21,64,3)
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    bluesky = (b > 0.45) & (b > r + 0.05) & (b > g + 0.02)
    clouds = (r > 0.72) & (g > 0.72) & (b > 0.72)
    return (bluesky | clouds).mean(axis=(1, 2))          # (N,)


def main():
    rows = []
    files = sorted(PROC_DIR.glob("*.npz"))
    for i, p in enumerate(files):
        frames = np.load(p)["frames"][::10]              # 1 frame sur 10
        sky = sky_share(frames)
        under = sky < 0.02                               # pas de ciel -> sous terre
        luma = frames.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
        part_under = float(under.mean())
        luma_under = float(np.median(luma[under])) if under.sum() >= 10 else None
        rows.append({"ep": p.stem, "part_souterrain": part_under,
                     "luma_souterrain": luma_under})
        if (i + 1) % 40 == 0:
            print(f"{i+1}/{len(files)}")

    (ROOT / "outputs" / "audit_bright_caves.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    cands = [r for r in rows if r["luma_souterrain"] is not None
             and r["part_souterrain"] >= 0.2]
    cands.sort(key=lambda r: -r["luma_souterrain"])
    print(f"\n{len(cands)} episodes avec >=20% de frames souterraines. "
          "Classement par CLARTE du souterrain :")
    for r in cands[:15]:
        tag = "  <-- FULL BRIGHTNESS ?" if r["luma_souterrain"] >= 0.22 else ""
        print(f"  luma_sous={r['luma_souterrain']:.3f}  "
              f"part_sous={r['part_souterrain']:.0%}  {r['ep']}{tag}")
    n_fb = sum(1 for r in cands if r["luma_souterrain"] >= 0.22)
    print(f"\n=> {n_fb} candidats 'full brightness' (souterrain clair)")


if __name__ == "__main__":
    main()
