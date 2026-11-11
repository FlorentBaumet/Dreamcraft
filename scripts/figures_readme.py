"""
Les trois figures de resultats du README, dans le style de figure_arene.py.

  - figure_timidite_donnees.png : timidite par domaine et par modele (chiffres du JOURNAL, 15/07) ;
  - figure_engagement.png       : courbe d'engagement (outputs/diag/engagement.json, figure_engagement.py) ;
  - figure_portee_action.png    : part de l'effet attribuee au bouton (outputs/diag/portee_sep.json,
                                  diag_separation.py).

Aucun calcul ici : on redessine des chiffres deja mesures.

Lance : .venv\\Scripts\\python.exe scripts\\figures_readme.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DIAG = ROOT / "outputs" / "diag"

FOND, ENCRE, ENCRE2, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLEU, BLEU_FONCE, GRIS = "#2a78d6", "#174f91", "#a3a29b"

# JOURNAL, 2026-07-15 : % des coups de pioche ou le reve ne casse jamais le bloc vise
TIMIDITE = {"reference": {"rare": 82, "dense": 66},
            "nourri": {"rare": 79, "dense": 47},
            "gros": {"rare": 81, "dense": 49}}


def style(ax):
    ax.set_facecolor(FOND)
    ax.grid(axis="y", color=GRILLE, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRILLE)
    ax.tick_params(length=0)


def base():
    plt.rcParams.update({"font.size": 10, "text.color": ENCRE, "axes.labelcolor": ENCRE2,
                         "xtick.color": ENCRE2, "ytick.color": ENCRE2, "legend.frameon": False})


def timidite():
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=FOND)
    series = [("reference", "référence, 3,65 M de paramètres", GRIS),
              ("nourri", "nourri en minage, même taille", BLEU),
              ("gros", "nourri en minage, 7 fois plus gros (25,7 M)", BLEU_FONCE)]
    groupes = [("rare", "grottes\nminage rare (~35 % des images)"),
               ("dense", "mine en tranchée\nminage dense (~80 % des images)")]
    w = 0.26
    for i, (cle, nom, c) in enumerate(series):
        for g, (dom, _) in enumerate(groupes):
            v = TIMIDITE[cle][dom]
            ax.bar(g + (i - 1) * w, v, width=w * 0.92, color=c, zorder=2, label=nom if g == 0 else None)
            ax.text(g + (i - 1) * w, v + 1.5, f"{v} %", ha="center", va="bottom", fontsize=9)
    ax.annotate("−19 points : même taille,\nseules les données changent", xy=(1, 54), xytext=(1.13, 66),
                ha="center", va="center", fontsize=9, color=ENCRE2,
                arrowprops={"arrowstyle": "->", "color": ENCRE2, "lw": 0.9})
    ax.set_xticks(range(len(groupes)), [g[1] for g in groupes])
    ax.set_ylim(0, 100)
    ax.set_ylabel("timidité : % des coups où le rêve ne casse pas le bloc")
    ax.set_title("La timidité suit la fréquence du minage dans les données, pas la taille du modèle",
                 loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "figure_timidite_donnees.png", dpi=150, facecolor=FOND)


def engagement():
    r = json.loads((DIAG / "engagement.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, facecolor=FOND)
    titres = {"dense": "Mine en tranchée : minage dense", "rare": "Grottes : minage rare"}
    for ax, dom in zip(axes, ("dense", "rare")):
        d = r[dom]
        hs = np.arange(1, len(d["reel"]) + 1)
        ax.plot(hs, d["reel"], color=ENCRE, lw=2.6, label="réel", zorder=3)
        ax.plot(hs, d["nourri"], color=BLEU, lw=2.4, label="rêve, modèle nourri en minage", zorder=3)
        ax.plot(hs, d["timide"], color=GRIS, lw=2.2, ls="--", label="rêve, modèle de référence", zorder=3)
        ecart = (d["nourri"][-1] - d["timide"][-1]) / (d["reel"][-1] - d["timide"][-1])
        ax.text(0.98, 0.04, f"au 20e pas, le modèle nourri rattrape {ecart:.0%} de l'écart\n"
                f"{d['evenements']} coups de pioche, épisodes exclus de l'entraînement".replace("%", " %"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=ENCRE2)
        ax.set_title(titres[dom], loc="left", fontsize=11)
        ax.set_xlabel("pas du rêve (20 pas par seconde)")
        ax.set_xticks([1, 5, 10, 15, 20])
        style(ax)
    axes[0].set_ylabel("changement du bloc visé depuis le coup\n(0 = la scène reste figée)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "figure_engagement.png", dpi=150, facecolor=FOND)


def portee():
    r = json.loads((DIAG / "portee_sep.json").read_text(encoding="utf-8"))
    hs = r["H"]
    courbes = [("v13b courageux / strip", "nourri en minage, domaine connu", BLEU, "-"),
               ("v13b courageux / nvcave", "nourri en minage, grottes", BLEU, ":"),
               ("v6 timide / strip", "référence, domaine connu", GRIS, "-"),
               ("v6 timide / nvcave", "référence, grottes", GRIS, ":")]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=FOND)
    ax.axvspan(min(hs), 12, color=BLEU, alpha=0.06, zorder=0)
    ax.text(min(hs) + 0.4, -33, "fenêtre utilisable pour décider : H ≤ 12", fontsize=9, color=BLEU_FONCE)
    ax.axhline(0, color=ENCRE2, lw=0.8, zorder=1)
    for cle, nom, c, ls in courbes:
        ax.plot(hs, [100 * x for x in r["sep"][cle]], color=c, ls=ls, lw=2.3, marker="o", ms=4,
                label=nom, zorder=3)
    ax.set_xticks(hs)
    ax.set_ylim(-40, 140)
    ax.set_xlabel("portée du rêve H (pas, 20 par seconde)")
    ax.set_ylabel("part de l'effet réel attribuée au bouton (%)")
    ax.set_title("Plus le rêve est long, plus l'effet de l'action se dilue (épisodes jamais vus)",
                 loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "figure_portee_action.png", dpi=150, facecolor=FOND)


def main():
    base()
    timidite()
    engagement()
    portee()
    print(f"-> {FIG} : figure_timidite_donnees.png, figure_engagement.png, figure_portee_action.png")


if __name__ == "__main__":
    main()
