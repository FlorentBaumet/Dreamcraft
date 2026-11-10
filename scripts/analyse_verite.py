"""
Ce que la verite du jeu dit de nos mesures, sans aucun modele :
  - le temps de cassage mesure, bloc par bloc, contre la regle du jeu,
  - le label sur les pixels ("le centre a change") contre le bloc reellement casse,
  - le decalage entre le journal du jeu et les images.

Lance : .venv\\Scripts\\python.exe scripts\\analyse_verite.py [motif, defaut dc_train*]
"""
from __future__ import annotations

import importlib.util as ilu
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft import verite as V  # noqa: E402

_s = ilu.spec_from_file_location("ep", Path(__file__).parent / "eval_pairs.py")
E = ilu.module_from_spec(_s)
_s.loader.exec_module(E)

Y0, Y1 = E.Y0, E.Y1
TRONCS = [V.bloc_id(n) for n in V.TRONCS]


def coups(path):
    """Par coup : le bloc vise au depart, sa distance, le premier bloc casse et apres combien de ticks."""
    z = np.load(path)
    if "truth" not in z.files:
        return []
    truth, phase = z["truth"], z["phase"]
    fr = E.C.load_episode_u8(path)[0].astype(np.float32) / 255.0
    out = []
    t_a = np.flatnonzero((phase[1:] == "A_taper") & (phase[:-1] == "B_temoin")) + 1
    for s in t_a:
        fin = s
        while fin < len(phase) and phase[fin] == "A_taper":
            fin += 1
        casses = np.flatnonzero(truth[s:fin + 1, 3] > 0)
        rec = {"t": int(s), "bloc": int(truth[s, 1]), "dist": int(truth[s, 2]), "casse": 0, "ticks": None,
               "lag": None, "effet": float(np.abs(fr[min(s + E.HREAL, len(fr) - 1), :, Y0:Y1, Y0:Y1]
                                                  - fr[s, :, Y0:Y1, Y0:Y1]).mean()),
               "casse_12": False, "ticks_ok": bool((truth[s:fin + 1, 0] >= 0).all())}
        # en foret : un tronc (pas une feuille ni l'herbe) casse-t-il en 12 ticks ou moins ?
        troncs = np.flatnonzero(np.isin(truth[s:fin + 1, 3], TRONCS))
        rec["tronc_12"] = bool(len(troncs) and truth[s + troncs[0], 0] - truth[s, 0] <= E.HREAL)
        if len(casses):
            f = s + int(casses[0])
            rec.update(casse=int(truth[f, 3]), ticks=int(truth[f, 0] - truth[s, 0]))
            rec["casse_12"] = rec["ticks"] <= E.HREAL
            # l'image ou le centre change le plus, autour de l'annonce du journal
            lo, hi = max(s, f - 5), min(fin, f + 6)
            if hi - lo >= 2:
                d = np.abs(np.diff(fr[lo:hi][:, :, Y0:Y1, Y0:Y1], axis=0)).mean((1, 2, 3))
                rec["lag"] = int(lo + np.argmax(d) + 1 - f)
        out.append(rec)
    return out


def bruit_chat(path):
    """Pendant le temoin rien ne bouge : un changement en bas a gauche serait le chat sur l'image."""
    z = np.load(path)
    idx = np.flatnonzero(z["phase"][:-1] == "B_temoin")
    idx = idx[z["phase"][idx + 1] == "B_temoin"]
    if not len(idx):
        return None
    fr = E.C.load_episode_u8(path)[0].astype(np.float32) / 255.0
    return float(np.median(np.abs(fr[idx + 1][:, :, 32:, :32] - fr[idx][:, :, 32:, :32]).mean((1, 2, 3))))


def main():
    motif = sys.argv[1] if len(sys.argv) > 1 else "dc_train*"
    files = sorted((ROOT / "data" / "processed_agent").glob(f"{motif}.npz"))
    tous = [c for p in files for c in coups(p)]
    if not tous:
        sys.exit(f"aucun coup annote pour {motif}")
    print(f"{len(tous)} coups annotes, {len(files)} sessions")
    bruits = [b for b in map(bruit_chat, files) if b is not None]
    if bruits:
        verdict = "chat invisible" if max(bruits) < 1e-3 else "ATTENTION : le bas a gauche bouge, chat visible ?"
        print(f"temoin, changement en bas a gauche : {max(bruits):.5f} ({verdict})")
    print()

    temps = defaultdict(list)
    for c in tous:
        if c["casse"] and c["bloc"] == c["casse"] and c["ticks_ok"]:
            temps[V.bloc_nom(c["casse"])].append(c["ticks"])
    table = {}
    print(f"{'bloc':<14}{'coups':>6}{'regle':>7}{'mesure':>8}{'min':>5}{'max':>5}")
    for nom, v in sorted(temps.items(), key=lambda kv: V.ticks_theorie(kv[0])):
        table[nom] = {"n": len(v), "regle": V.ticks_theorie(nom), "mediane": float(np.median(v)),
                      "min": int(min(v)), "max": int(max(v))}
        print(f"{nom:<14}{len(v):>6}{V.ticks_theorie(nom):>7}{np.median(v):>8.1f}{min(v):>5}{max(v):>5}")

    # Le coup etait-il joue a portee d'un bloc ? Et les coups a portee cassent-ils ?
    a_portee = np.array([0 < c["bloc"] and c["dist"] <= V.PORTEE for c in tous])
    casse = np.array([c["casse"] > 0 for c in tous])
    print(f"\nbloc a portee au depart : {a_portee.mean():.0%} | casse pendant le coup : {casse.mean():.0%}")
    print(f"  a portee et casse : {(a_portee & casse).sum()} | a portee, intact : {(a_portee & ~casse).sum()}")
    print(f"  hors de portee et casse : {(~a_portee & casse).sum()} | hors de portee, intact : {(~a_portee & ~casse).sum()}")

    # Le label sur les pixels dit-il la verite ?
    ref = sorted((ROOT / "data" / "processed_agent").glob("dc_train*.npz"))[:E.REF_TRAIN]
    thr = E.otsu(np.concatenate([E.load_session(p)["eff_a"] for p in ref])) if len(ref) else 0.0072
    pix = np.array([c["effet"] > thr for c in tous])
    vrai = np.array([c["casse_12"] for c in tous])
    print(f"\nlabel pixels (seuil {thr:.4f}) contre bloc casse en {E.HREAL} ticks :")
    print(f"  accord {np.mean(pix == vrai):.0%} | pixels oui, rien de casse {(pix & ~vrai).sum()} | "
          f"pixels non, bloc casse {(~pix & vrai).sum()}")

    lags = [c["lag"] for c in tous if c["lag"] is not None]
    if lags:
        print(f"\nimage qui change le plus, par rapport au journal : mediane {np.median(lags):+.0f}, "
              f"de {min(lags):+d} a {max(lags):+d} images")

    out = ROOT / "outputs" / "eval_pairs" / "verite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"temps": table, "a_portee": float(a_portee.mean()),
                               "casse": float(casse.mean()),
                               "accord_pixels": float(np.mean(pix == vrai)), "n": len(tous)},
                              indent=1), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
