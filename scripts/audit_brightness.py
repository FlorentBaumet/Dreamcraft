"""
Audit de luminosite du corpus : qui joue en "full brightness" ?

Pour chaque episode npz : luminance mediane, 10e percentile (sombre typique),
et part des frames sombres (luma < 0.15). Un joueur en full brightness a des
grottes CLAIRES -> percentile bas eleve malgre des scenes souterraines.

Sortie : outputs/audit_brightness.json + classement console.
Lance  : .venv\\Scripts\\python.exe scripts\\audit_brightness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def main():
    rows = []
    files = sorted(PROC_DIR.glob("*.npz"))
    for i, p in enumerate(files):
        frames = np.load(p)["frames"][::20]              # 1 frame sur 20 suffit
        luma = frames.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
        rows.append({
            "ep": p.stem,
            "luma_mediane": float(np.median(luma)),
            "luma_p10": float(np.percentile(luma, 10)),
            "part_frames_sombres": float((luma < 0.15).mean()),
        })
        if (i + 1) % 40 == 0:
            print(f"{i+1}/{len(files)} episodes audites")

    rows.sort(key=lambda r: -r["luma_p10"])
    out = ROOT / "outputs" / "audit_brightness.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== les 12 plus 'lumineux partout' (p10 eleve = grottes claires ou pas de grottes) ===")
    for r in rows[:12]:
        print(f"  p10={r['luma_p10']:.3f}  med={r['luma_mediane']:.3f}  "
              f"sombre={r['part_frames_sombres']:.0%}  {r['ep']}")
    print("\n=== les 8 plus sombres ===")
    for r in rows[-8:]:
        print(f"  p10={r['luma_p10']:.3f}  med={r['luma_mediane']:.3f}  "
              f"sombre={r['part_frames_sombres']:.0%}  {r['ep']}")
    n_dark = sum(1 for r in rows if r["part_frames_sombres"] > 0.2)
    print(f"\n{n_dark}/{len(rows)} episodes ont >20% de frames sombres "
          f"(luma<0.15) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
