"""
Chiffre les baselines B0 (copie) et B1 (flux optique) sur les episodes npz.

C'est la barre a battre du Go/No-Go (CADRE_DREAMCRAFT.md #5-6) -- calculee
AVANT tout entrainement, comme le veut le protocole.

Sorties :
  - outputs/baselines/results.json   (tous les chiffres)
  - outputs/baselines/exemple_*.png  (contact sheets : frame_t | reel t+1 | B1 | masque)
  - tableau resume dans la console

Lance : .venv\\Scripts\\python.exe scripts\\eval_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import vpt  # noqa: E402
from dreamcraft.eval import baselines, metrics  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "baselines"

# Seuil camera (en unites normalisees [-1,1] ; 0.05 ~ 3 px de souris).
CAM_ACTIVE = 0.05


def action_subsets(a: np.ndarray) -> list[str]:
    """Etiquettes de la transition selon l'action (non exclusives)."""
    subs = []
    if a[6] > 0.5:
        subs.append("taper")
    if abs(a[8]) > CAM_ACTIVE or abs(a[9]) > CAM_ACTIVE:
        subs.append("camera")
    if a[0] > 0.5 or a[1] > 0.5 or a[2] > 0.5 or a[3] > 0.5:
        subs.append("deplacement")
    if not subs:
        subs.append("idle")
    return subs


def main():
    npz_files = sorted(PROC_DIR.glob("*.npz"))
    if not npz_files:
        print("Aucun episode npz. Lance d'abord scripts/build_dataset.py")
        sys.exit(1)

    # accumulateurs: cle -> dict(sums)
    keys = ["tout", "taper", "camera", "deplacement", "idle"]
    acc = {k: {"n": 0, "g_b0": 0.0, "g_b1": 0.0,
               "n_delta": 0, "d_b0": 0.0, "d_b1": 0.0, "coverage": 0.0} for k in keys}
    n_static = 0
    n_gui_skipped = 0
    examples = []  # (coverage, tag, ep_name, t, frame_t, actual, pred_b1, mask)

    for npz in npz_files:
        frames_u8, actions, gui = vpt.load_episode(npz)
        frames = frames_u8.astype(np.float32) / 255.0
        n = frames.shape[0]
        print(f"[{npz.stem}] {n} frames")

        for t in range(1, n - 1):
            # exclusions : GUI ouverte autour de la transition (dynamique de menu)
            if gui[t] or gui[t - 1]:
                n_gui_skipped += 1
                continue

            prev, cur, target = frames[t - 1], frames[t], frames[t + 1]
            pred_b0 = baselines.b0_copy(cur)
            pred_b1 = baselines.b1_optical_flow(prev, cur)

            g_b0 = metrics.mse(pred_b0, target)
            g_b1 = metrics.mse(pred_b1, target)
            d_b0, n_changed = metrics.delta_region_mse(pred_b0, target, cur)
            d_b1, _ = metrics.delta_region_mse(pred_b1, target, cur)

            is_static = np.isnan(d_b0)
            if is_static:
                n_static += 1

            subs = ["tout"] + action_subsets(actions[t])
            for k in subs:
                a = acc[k]
                a["n"] += 1
                a["g_b0"] += g_b0
                a["g_b1"] += g_b1
                if not is_static:
                    a["n_delta"] += 1
                    a["d_b0"] += d_b0
                    a["d_b1"] += d_b1
                    a["coverage"] += n_changed / (64 * 64)

            # candidats pour les visuels (grosses zones de changement)
            if not is_static and n_changed > 200:
                tag = "taper" if "taper" in subs else ("camera" if "camera" in subs else "autre")
                mask = metrics.change_mask(cur, target)
                examples.append((n_changed, tag, npz.stem, t, cur, target, pred_b1, mask))

    # --- agregation ---
    def finalize(a):
        if a["n"] == 0:
            return None
        out = {
            "paires": a["n"],
            "MSE_globale_B0": a["g_b0"] / a["n"],
            "MSE_globale_B1": a["g_b1"] / a["n"],
        }
        if a["n_delta"] > 0:
            out.update({
                "paires_dynamiques": a["n_delta"],
                "MSE_delta_B0": a["d_b0"] / a["n_delta"],
                "MSE_delta_B1": a["d_b1"] / a["n_delta"],
                "couverture_moy": a["coverage"] / a["n_delta"],
            })
        return out

    results = {k: finalize(a) for k, a in acc.items()}
    results["_meta"] = {
        "episodes": [p.stem for p in npz_files],
        "paires_gui_exclues": n_gui_skipped,
        "paires_statiques": n_static,
        "seuil_changement": metrics.CHANGE_THRESHOLD,
        "note": "MSE sur images [0,1]. Delta = restreinte aux pixels qui changent.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- visuels : 1 exemple par tag, les plus 'dynamiques' ---
    examples.sort(key=lambda e: -e[0])
    saved_tags = set()
    for n_changed, tag, ep, t, cur, target, pred_b1, mask in examples:
        if tag in saved_tags:
            continue
        saved_tags.add(tag)
        panels = [cur, target, pred_b1, np.repeat(mask[..., None], 3, axis=-1).astype(np.float32)]
        row = np.concatenate(panels, axis=1)  # (64, 256, 3)
        big = cv2.resize((row * 255).astype(np.uint8), None, fx=4, fy=4,
                         interpolation=cv2.INTER_NEAREST)
        p = OUT_DIR / f"exemple_{tag}_{ep}_t{t}.png"
        cv2.imwrite(str(p), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        print(f"visuel [{tag}] -> {p.name}  (pixels changeants: {n_changed})")
        if len(saved_tags) >= 3:
            break

    # --- tableau console ---
    def fmt(x):
        return f"{x:.5f}" if isinstance(x, float) else str(x)

    print("\n=== BASELINES (la barre a battre) ===")
    print(f"{'subset':<12} {'paires':>7} {'dyn.':>6} | {'gMSE B0':>9} {'gMSE B1':>9} | "
          f"{'dMSE B0':>9} {'dMSE B1':>9} | {'couv.':>6}")
    for k in keys:
        r = results[k]
        if r is None:
            continue
        print(f"{k:<12} {r['paires']:>7} {r.get('paires_dynamiques', 0):>6} | "
              f"{fmt(r['MSE_globale_B0']):>9} {fmt(r['MSE_globale_B1']):>9} | "
              f"{fmt(r.get('MSE_delta_B0', float('nan'))):>9} "
              f"{fmt(r.get('MSE_delta_B1', float('nan'))):>9} | "
              f"{r.get('couverture_moy', 0) * 100:>5.1f}%")
    print(f"\n(exclues: {n_gui_skipped} paires GUI ; statiques: {n_static})")
    print(f"Resultats: {OUT_DIR.relative_to(ROOT)}\\results.json")


if __name__ == "__main__":
    main()
