"""
Superpose plusieurs courbes de portee (results.json de rollout_phase1.py)
sur un seul graphe -> outputs/comparatif_portee.png

Lance : .venv\\Scripts\\python.exe scripts\\compare_curves.py tag1 label1 tag2 label2 ...
        (tag = sous-dossier de outputs/ contenant un results.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    if len(args) < 4 or len(args) % 2:
        print("usage: compare_curves.py tag1 label1 tag2 label2 [...]")
        sys.exit(1)
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    b0_done = False
    for tag, label in pairs:
        r = json.loads((ROOT / "outputs" / tag / "results.json").read_text(encoding="utf-8"))
        hs = np.arange(1, r["H"] + 1)
        axes[0].plot(hs, r["mse_modele_par_h"], lw=2, label=label)
        if "dmse_modele_par_h" in r:
            axes[1].plot(hs, r["dmse_modele_par_h"], lw=2,
                         label=f"{label} (portée {r.get('portee_delta_pas', '?')} pas)")
        if not b0_done:
            axes[0].plot(hs, r["mse_b0_gelee_par_h"], "k--", lw=1.8, label="B0-gelée")
            if "dmse_b0_gelee_par_h" in r:
                axes[1].plot(hs, r["dmse_b0_gelee_par_h"], "k--", lw=1.8, label="B0-gelée")
            b0_done = True
    axes[0].set_title("MSE globale (récompense l'inertie)")
    axes[1].set_title("MSE Δ-region, là où la réalité bouge (l'honnête)")
    for ax in axes:
        ax.set_xlabel("horizon de rêve h (pas de 50 ms)")
        ax.set_ylabel("MSE vs réalité")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Portée du rêve - comparatif des recettes (même épisode, mêmes départs)")
    fig.tight_layout()
    out = ROOT / "outputs" / "comparatif_portee.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
