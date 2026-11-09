"""
Figure de l'arene finale : blocs casses en 30 s par joueur, et part des frappes dans le vide.

Lit outputs/arene/resultats_20261109-191206.json (scores) et les sessions de l'arene
(verite du jeu) ; ecrit docs/figures/figure_arene.png.

Lance : .venv\\Scripts\\python.exe scripts\\figure_arene.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "arene" / "resultats_20261109-191206.json"
DATA = ROOT / "data" / "processed_agent"
OUT = ROOT / "docs" / "figures" / "figure_arene.png"
DEBUT = "20261109-1830"                     # sessions de l'arene finale : apres cette heure
RATEES = (11, 12)                           # manches 12 et 13 (indices) : salles mal construites

FOND, ENCRE, ENCRE2, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLEU, GRIS = "#2a78d6", "#a3a29b"
NOMS = {"appris": "appris\n(verifie + détecteur v3)", "mine_et_balaye": "mine_et_balaye\n(script)",
        "verifie": "verifie\n(détecteur v1)", "hasard": "hasard"}


def frappes_dans_le_vide(joueur):
    tot = vide = 0
    for p in DATA.glob(f"dc_arene_{joueur}_*.npz"):
        if p.stem.split("_")[-1] <= DEBUT:
            continue
        z = np.load(p)
        ph = z["phase"]
        tr = z["truth"][:len(ph)]
        t = (ph == "taper") & (tr[:, 0] >= 0)
        tot += t.sum()
        vide += (t & ~((tr[:, 1] > 0) & (tr[:, 2] <= 45))).sum()
    return vide / tot


def main():
    scores = json.loads(RES.read_text(encoding="utf-8"))["scores"]
    ordre = ["appris", "mine_et_balaye", "verifie", "hasard"]
    plt.rcParams.update({"font.size": 10, "text.color": ENCRE, "axes.labelcolor": ENCRE2,
                         "xtick.color": ENCRE2, "ytick.color": ENCRE2})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [2.2, 1]},
                               facecolor=FOND)
    rng = np.random.default_rng(0)
    for i, nom in enumerate(ordre):
        v = np.array(scores[nom], dtype=float)
        c = BLEU if nom == "appris" else GRIS
        a.bar(i, v.mean(), width=0.6, color=c, alpha=0.35, zorder=2)
        x = i + rng.uniform(-0.18, 0.18, len(v))
        ok = np.ones(len(v), bool)
        ok[list(RATEES)] = False
        a.scatter(x[ok], v[ok], s=22, color=c, zorder=3, edgecolor=FOND, linewidth=0.8)
        a.scatter(x[~ok], v[~ok], s=22, facecolor="none", edgecolor=c, zorder=3, linewidth=1.2)
    a.set_xticks(range(len(ordre)), [f"{NOMS[n]}\nmoyenne {np.mean(scores[n]):.1f}".replace(".", ",")
                                     for n in ordre])
    a.set_ylabel("blocs réellement cassés en 30 s")
    a.set_ylim(0, 45)
    a.set_yticks(range(0, 46, 5))
    a.set_title("Arène finale : 16 manches, mêmes salles pour tous", loc="left", fontsize=11)
    a.text(0.99, 0.97, "point = une manche ; cercle vide = salle ratée (manches 12-13)",
           transform=a.transAxes, ha="right", va="top", fontsize=8, color=ENCRE2)

    parts = [frappes_dans_le_vide("verifie"), frappes_dans_le_vide("appris")]
    b.bar([0, 1], [100 * p for p in parts], width=0.55, color=[GRIS, BLEU], zorder=2)
    for i, p in enumerate(parts):
        b.text(i, 100 * p + 2, f"{100 * p:.0f} %", ha="center", va="bottom", fontweight="bold")
    b.set_xticks([0, 1], ["verifie", "appris"])
    b.set_ylim(0, 100)
    b.set_ylabel("% des frappes sur un bloc hors de portée")
    b.set_title("Frappes dans le vide", loc="left", fontsize=11)

    for ax in (a, b):
        ax.set_facecolor(FOND)
        ax.grid(axis="y", color=GRILLE, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRILLE)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=FOND)
    print(f"-> {OUT}  (frappes dans le vide : verifie {parts[0]:.0%}, appris {parts[1]:.0%})")


if __name__ == "__main__":
    main()
