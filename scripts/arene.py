"""
Arene : l'agent qui decide avec le detecteur, contre des references, dans des salles identiques.

Chaque manche : un nouvel endroit sous terre, une petite cavite de 3 x 2 x 3 (les murs sont a
portee), copiee sous la carte avant que quiconque joue. Avant chaque joueur, la copie est remise
en place : tout le monde trouve exactement les memes blocs, seul le joueur change.
Autour de la salle, liquides, gravier, sable et grottes voisines sont remplaces par de la pierre ;
une salle ou le bloc vise au depart n'est pas a portee (verite du jeu) est refusee.
Ordre des joueurs tournant d'une manche a l'autre.

Joueurs :
  detecteur      : tape seulement si le detecteur voit un bloc qui cassera vite, sinon cherche
  reveur         : l'ancien agent, qui reve chaque plan avec le world model (v15-paires)
  mine_et_balaye : script ecrit a la main (tape 3/4 du temps, tourne, avance)
  hasard, taper  : references sans intelligence

Score : blocs reellement casses en 30 s, lus dans le journal du jeu.

Regle fixee avant de jouer : test de signe unilateral sur les manches, egalites retirees.
L'agent "detecteur" bat une reference si p < 0,05 (avec 8 manches sans egalite : 7 gagnees).
Le seuil du detecteur reste 0,5, jamais regle sur l'arene.

Monde dc_test ouvert, clique dans le jeu, puis :
  .venv\\Scripts\\python.exe scripts\\arene.py --essai     (1 manche de 5 s par joueur)
  .venv\\Scripts\\python.exe scripts\\arene.py
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import torch


def _charge(nom, fichier):
    s = ilu.spec_from_file_location(nom, Path(__file__).parent / fichier)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


AU = _charge("autonome", "autonome.py")
D = _charge("detecteur", "detecteur.py")
A, V = AU.A, AU.V
ROOT = AU.ROOT
OUT = ROOT / "outputs" / "arene"
REPLAN = 4
ANCRE = "@e[type=marker,tag=dc_ancre,limit=1]"
ICI = f"/execute at {ANCRE} run "
AGENT = "verifie"           # l'agent teste par la regle (arene 2) ; arene 1 : "detecteur"


class DetecteurPolicy:
    """Regarde l'image courante. Si le detecteur dit "ca casse", tape ; sinon cherche :
    tourne en balayant de l'horizon vers le sol, et avance un peu apres 4 s sans rien trouver.

    verifie=True (v2, ajoutee apres l'arene 1) : apres 12 ticks de frappe, regarde si le centre
    de l'image a change. Sinon le bloc etait hors de portee : 1er echec, avancer en tapant 1 s ;
    2e echec d'affilee, tourner franchement ailleurs. Seuil de changement calibre sur dc_train
    seulement (2/255 : 81 % des coups a portee le depassent, 8 % des coups hors portee)."""

    CIBLES = (30, 110, 60, 160, 0)          # inclinaison visee, en pixels souris (+ = vers le bas)
    CHANGE = 2.0

    def __init__(self, device, seuil=0.5, engage=3, verifie=False, modele=None):
        self.m = D.Net().to(device)
        self.m.load_state_dict(torch.load(modele or D.OUT / "model.pt", weights_only=True))
        self.m.eval()
        self.device, self.seuil, self.engage, self.verifie = device, seuil, engage, verifie
        self.new_session()

    def new_session(self):
        self.reste = 0          # decisions de frappe encore engagees : un bloc entame est fini
        self.vide = 0           # decisions de recherche d'affilee
        self.pitch = 0.0
        self.ref = None         # centre de l'image au debut de la frappe
        self.echecs = 0
        self.force = []         # decisions imposees apres un echec

    @torch.no_grad()
    def proba(self, frame_hwc):
        x = D.masque(frame_hwc.transpose(2, 0, 1)[None].copy())
        return float(torch.sigmoid(self.m(torch.from_numpy(x).float().div(255).to(self.device))))

    @staticmethod
    def centre(frame_hwc):
        return frame_hwc[26:38, 26:38].astype(np.float32)

    def chercher(self, h):
        self.vide += 1
        v = np.zeros((h, 10), dtype=np.float32)
        if self.vide % 25 >= 20:            # 1 s de marche toutes les 5 s de recherche vaine
            v[:REPLAN, A.IDX["avancer"]] = 1.0
            v[:REPLAN, A.IDX["sauter"]] = 1.0
            return v, "avancer"
        cible = self.CIBLES[(self.vide // 3) % len(self.CIBLES)]
        dy = float(np.clip((cible - self.pitch) / (REPLAN * A.CAM_CLIP), -0.5, 0.5))
        v[:REPLAN, A.CAM_DX] = 0.4
        v[:REPLAN, A.CAM_DY] = dy
        self.pitch += dy * REPLAN * A.CAM_CLIP
        return v, "chercher"

    def plan(self, ctx_u8, h):
        if self.force:
            quoi = self.force.pop(0)
            if quoi == "approcher":
                return A.plan_vector({"taper": 1.0, "avancer": 1.0}, h), "approcher"
            return self.chercher(h)[0], "fuir"
        if self.reste > 0:
            self.reste -= 1
            return A.plan_vector({"taper": 1.0}, h), "taper"
        if self.verifie and self.ref is not None:          # la frappe vient de finir : a-t-elle servi ?
            change = float(np.abs(self.centre(ctx_u8[-1]) - self.ref).mean())
            self.ref = None
            if change < self.CHANGE:
                self.echecs += 1
                self.force = ["approcher"] * 5 if self.echecs == 1 else ["fuir"] * 6
                if self.echecs >= 2:
                    self.echecs = 0
                return self.plan(ctx_u8, h)
            self.echecs = 0
        if self.proba(ctx_u8[-1]) >= self.seuil:
            self.reste = self.engage - 1
            self.ref = self.centre(ctx_u8[-1])
            self.vide = 0
            return A.plan_vector({"taper": 1.0}, h), "taper"
        return self.chercher(h)


def creuse(env):
    """Nouvel endroit, cavite creusee, copie de sauvegarde 60 blocs plus bas."""
    env.command("/kill @e[type=marker,tag=dc_ancre]")
    env.command("/tp @s ~60 30 ~ 0 0")
    time.sleep(2.0)                          # le terrain doit etre charge, sinon /fill echoue
    env.command('/execute at @s align xyz positioned ~0.5 ~ ~0.5 run summon marker ~ ~ ~ {Tags:["dc_ancre"]}')
    # sous un ocean, la salle serait noyee : sous l'eau on casse 5 fois moins vite.
    # Gravier et sable : ils tomberaient dans la cavite (arene 4, manche 12 : depart dans le gravier).
    for bloc in ("water", "lava", "seagrass", "tall_seagrass", "kelp", "kelp_plant", "gravel", "sand"):
        env.command(ICI + f"fill ~-12 ~-4 ~-12 ~12 ~6 ~12 stone replace {bloc}")
    # une grotte voisine ouvrirait la salle sur le vide (arene 4, manche 13) : on la bouche autour
    for vide in ("air", "cave_air"):
        env.command(ICI + f"fill ~-5 ~-3 ~-5 ~5 ~4 ~5 stone replace {vide}")
    env.command(ICI + "fill ~-1 ~ ~-1 ~1 ~1 ~1 air")
    time.sleep(1.0)
    env.command(ICI + "clone ~-12 ~-4 ~-12 ~12 ~6 ~12 ~-12 ~-60 ~-12")


def verite_fraiche(env, attente=0.5):
    """La verite du jeu, apres avoir verifie qu'elle avance : figee, elle ferait refuser des salles a tort."""
    avant = int(env.read_truth()[0])
    time.sleep(attente)
    t = env.read_truth()
    if t[0] < 0 or t[0] == avant:
        raise RuntimeError("la verite du jeu ne bouge plus (jeu en pause ? journal change ?)")
    return t


def salle_ok(env):
    """Verite du jeu au depart : un bloc connu doit etre vise, a portee, et pas sur la tete du joueur."""
    remet(env)
    tick, bloc, dist = (int(x) for x in verite_fraiche(env)[:3])
    ok = tick >= 0 and 1 <= bloc <= len(V.BLOCS) and 0 < dist <= V.PORTEE
    if not ok:
        print(f"[salle refusee] bloc vise {bloc}, distance {dist / 10:.1f} : on creuse plus loin", flush=True)
    return ok


def nouvelle_manche(env, essais=5):
    for _ in range(essais):
        creuse(env)
        if salle_ok(env):
            return
    raise RuntimeError(f"aucune salle valable en {essais} essais")


def remet(env):
    """La salle exactement comme au debut de la manche, le joueur au centre, regard a l'horizon."""
    env.command("/kill @e[type=!player,type=!marker]")
    env.command(ICI + "clone ~-12 ~-60 ~-12 ~12 ~-50 ~12 ~-12 ~-4 ~-12")
    env.command(ICI + "tp @s ~ ~ ~ 0 0")
    env.command("/clear @s")
    env.command(A.PIOCHE)
    time.sleep(1.0)


def joue(env, pol, steps, nom, tag):
    remet(env)
    if hasattr(pol, "new_session"):
        pol.new_session()
    r = A.run_loop(env, pol, steps, 8, REPLAN)
    r["room"] = "arene"
    A.report_and_save(r, argparse.Namespace(env="live", policy=nom, tag=tag), REPLAN)
    if r["steps"] < steps:
        raise AU.Stop(f"arret apres {r['steps']}/{steps} pas")
    return V.blocs_casses(r["truth"]), r["frames"][0], int(r["truth"][0, 2])


def test_de_signe(a, b):
    """P(au moins autant de victoires par hasard), egalites retirees."""
    gagne = sum(x > y for x, y in zip(a, b))
    n = gagne + sum(x < y for x, y in zip(a, b))
    p = sum(comb(n, k) for k in range(gagne, n + 1)) / 2 ** n if n else 1.0
    return gagne, n, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manches", type=int, default=8)
    ap.add_argument("--pas", type=int, default=600, help="pas par joueur (20 par seconde)")
    ap.add_argument("--essai", action="store_true", help="1 manche de 5 s par joueur, rien de conclu")
    ap.add_argument("--joueurs", default="", help="liste separee par des virgules (defaut : tous)")
    ap.add_argument("--agent", default="verifie", help="joueur teste par la regle")
    ap.add_argument("--attente", type=int, default=10, help="secondes laissees pour cliquer dans le jeu")
    args = ap.parse_args()
    if args.essai:
        args.manches, args.pas = 1, 100
    tag = "dc_arene_essai" if args.essai else "dc_arene"
    AU.attendre_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    joueurs = {
        "verifie": DetecteurPolicy(device, verifie=True),
        "detecteur": DetecteurPolicy(device),
        "reveur": A.ChooserPolicy(str(AU.START_MODEL), str(ROOT / "outputs" / "goal_detector" / "model_best.pt"),
                                  device, 8, "jumeau"),
        "mine_et_balaye": A.ScriptedPolicy(),
        "hasard": A.RandomPolicy(),
        "taper": A.TaperPolicy(),
    }
    v3 = ROOT / "outputs" / "detecteur_v3" / "model.pt"
    if v3.exists():
        joueurs["appris"] = DetecteurPolicy(device, verifie=True, modele=v3)
    if args.joueurs:
        joueurs = {k: joueurs[k] for k in args.joueurs.split(",")}
    global AGENT
    AGENT = args.agent

    try:
        env = A.LiveEnv()
    except RuntimeError as e:
        sys.exit(f"ARRET : {e}")
    print(f"[arene] clique dans le jeu, tu as {args.attente} s...", flush=True)
    t = time.time()
    while not env.focused() and time.time() - t < args.attente:
        time.sleep(0.1)
    if not env.focused():
        sys.exit("Minecraft n'est pas au premier plan. Rien n'a ete envoye.")
    time.sleep(3)

    AU.garde_eveille(True)
    scores = {nom: [] for nom in joueurs}
    ecarts = []
    try:
        env.start_truth()
        env.setup_world()
        for manche in range(args.manches):
            nouvelle_manche(env)
            noms = list(joueurs)
            noms = noms[manche % len(noms):] + noms[:manche % len(noms)]
            premieres, visee = [], []
            for nom in noms:
                print(f"\n=== manche {manche + 1}/{args.manches} : {nom} ===", flush=True)
                n, f0, d0 = joue(env, joueurs[nom], args.pas, nom, tag)
                scores[nom].append(n)
                premieres.append(f0.astype(np.float32))
                visee.append(d0)
            # controle : chaque joueur doit partir de la meme image
            ecarts.append(float(max(np.abs(f - premieres[0]).mean() for f in premieres)))
            print(f"\n[manche {manche + 1}] blocs : { {k: v[-1] for k, v in scores.items()} } "
                  f"| ecart max des images de depart {ecarts[-1]:.1f}/255 | distance visee au depart {visee}", flush=True)
    except (AU.Stop, RuntimeError) as e:
        print(f"\narret : {e}")
    except KeyboardInterrupt:
        print("\narret demande")
    finally:
        env.close()
        AU.garde_eveille(False)

    fini = min(len(v) for v in scores.values())
    scores = {k: v[:fini] for k, v in scores.items()}
    res = {"manches": fini, "pas": args.pas, "essai": args.essai, "scores": scores,
           "ecart_images_depart": ecarts[:fini], "contre": {}}
    print(f"\n{fini} manches completes")
    for nom, v in scores.items():
        print(f"  {nom:<15} {np.mean(v) if v else 0:5.1f} blocs en moyenne  {v}")
    for ref in joueurs:
        if ref == AGENT or not fini:
            continue
        g, n, p = test_de_signe(scores[AGENT], scores[ref])
        res["contre"][ref] = {"gagnees": g, "decisives": n, "p": p, "battu": bool(p < 0.05)}
        print(f"  {AGENT} contre {ref:<15} gagne {g}/{n} manches decisives, p = {p:.3f} -> "
              f"{'BATTU' if p < 0.05 else 'pas prouve'}")
    sortie = OUT / ("essai.json" if args.essai else f"resultats_{time.strftime('%Y%m%d-%H%M%S')}.json")
    sortie.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {sortie}")


if __name__ == "__main__":
    main()
