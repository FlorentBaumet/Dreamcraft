"""
En surface, en foret : couper du bois. Meme principe que l'arene sous terre, adapte aux arbres.

Chaque manche : un nouvel endroit environ 120 blocs plus loin, pose sur la terre ferme par spreadplayers
(le biome unique garde ses oceans), avec un sol sous les pieds et au moins un tronc visible a 8 blocs
ou moins (verite du jeu, en tournant la tete sur 8 directions). Un cube de
25 blocs autour (arbres compris) est copie sous la carte, puis remis a l'identique avant chaque
joueur : tout le monde trouve les memes arbres. Hache en diamant : un tronc casse en 8 ticks.

Deux usages :
  --collecte N : N sessions de sonde (regarder, taper, pivoter, marcher), pour apprendre les troncs
  --manches M  : arene, score = troncs reellement casses ; test de signe, p < 0,05, fixe avant

Monde de foret (type « Biome unique », Foret) avec la verite installee, clique dans le jeu, puis :
  .venv\\Scripts\\python.exe scripts\\foret.py --essai
  .venv\\Scripts\\python.exe scripts\\foret.py --collecte 80
  (detecteur bois : voir l'en-tete de scripts\\detecteur.py)
  .venv\\Scripts\\python.exe scripts\\foret.py --collecte 10 --tag vt_foret --manches 16
Un seul processus pour enchainer : un second script lance a la suite peut trouver le jeu au second plan.
Basculer sur une autre fenetre arrete tout, proprement.
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_s = ilu.spec_from_file_location("arene", Path(__file__).parent / "arene.py")
AR = ilu.module_from_spec(_s)
_s.loader.exec_module(AR)
A, V, AU = AR.A, AR.V, AR.AU
ROOT = AR.ROOT
OUT = ROOT / "outputs" / "foret"
HACHE = "/item replace entity @s weapon.mainhand with minecraft:diamond_axe{Unbreakable:1b}"
ANCRE = "@e[type=marker,tag=dc_foret,limit=1]"
ICI = f"/execute at {ANCRE} run "
TRONCS = [V.bloc_id(n) for n in V.TRONCS]
CUBE = "~-12 ~-2 ~-12 ~12 ~22 ~12"                 # 25 x 25 x 25 blocs, sous la limite de /clone
SAUVE = "~-12 ~-62 ~-12"                           # la copie, 60 blocs sous la surface
DETECTEUR_BOIS = ROOT / "outputs" / "detecteur_bois" / "model.pt"
DETECTEUR_BOIS_V3 = ROOT / "outputs" / "detecteur_bois_v3" / "model.pt"     # reentraine sur ses propres coups
# en foret, les troncs sont a hauteur des yeux et au-dessus : on cherche de l'horizon vers le haut
# (sous terre : de l'horizon vers le sol). Fixe avant toute arene en foret.
CIBLES_FORET = (0, -60, 20, -120, 0)


def pose(env, dx=120, dz=0):
    """Nouvel endroit, les pieds sur le sol (pas sur les feuilles), copie de sauvegarde.
    Le relief garde ses oceans meme en biome unique : spreadplayers ne pose jamais sur l'eau."""
    env.command("/kill @e[type=marker,tag=dc_foret]")
    env.command(f"/execute at @s positioned ~{dx} ~ ~{dz} run spreadplayers ~ ~ 0 150 false @s")
    time.sleep(2.5)                              # le terrain doit etre charge
    env.command("/execute at @s positioned over motion_blocking_no_leaves run tp @s ~ ~ ~ 0 0")
    time.sleep(1.0)
    env.command('/execute at @s align xyz positioned ~0.5 ~ ~0.5 run summon marker ~ ~ ~ {Tags:["dc_foret"]}')
    env.command(ICI + f"clone {CUBE} {SAUVE}")


def remet(env):
    """Les arbres exactement comme au debut de la manche, le joueur au centre, regard a l'horizon."""
    env.command("/kill @e[type=!player,type=!marker]")
    env.command(ICI + f"clone ~-12 ~-62 ~-12 ~12 ~-38 ~12 ~-12 ~-2 ~-12")
    env.command(ICI + "tp @s ~ ~ ~ 0 0")
    env.command("/clear @s")
    env.command(HACHE)
    time.sleep(1.0)


def endroit_ok(env):
    """Les directions (sur 8) qui montrent un tronc a 8 blocs ou moins. Aucune : endroit refuse."""
    remet(env)
    # les pieds doivent etre sur un bloc : sur l'eau, le rayon traverse et ne trouve rien de proche
    env.command(ICI + "tp @s ~ ~ ~ 0 90")
    sol = AR.verite_fraiche(env, 0.4)
    if not (sol[1] > 0 and sol[2] <= 25):
        print(f"[endroit refuse] pas de sol sous les pieds (bloc {sol[1]}, distance {sol[2] / 10:.1f})", flush=True)
        return []
    vues = []
    for yaw in range(0, 360, 45):
        env.command(ICI + f"tp @s ~ ~ ~ {yaw} 0")
        if int(AR.verite_fraiche(env, 0.4)[1]) in TRONCS:
            vues.append(yaw)
    env.command(ICI + "tp @s ~ ~ ~ 0 0")
    return vues


def nouvel_endroit(env, essais=12):
    for k in range(essais):
        # vers l'est ; apres 3 refus d'affilee, grand saut de cote (un ocean peut etre large)
        pose(env, *((120, 0) if k < 3 else (0, 400 * (1 if k % 2 else -1))))
        vues = endroit_ok(env)
        if vues:
            print(f"[endroit] tronc visible dans {len(vues)} directions sur 8", flush=True)
            return vues
        print("[endroit refuse] aucun tronc a 8 blocs, ou pas de sol : on va plus loin", flush=True)
    raise RuntimeError(f"aucun arbre trouve en {essais} essais : monde de foret ?")


def joue(env, pol, steps, nom, tag, yaw=0, replan=AR.REPLAN):
    remet(env)
    if yaw:
        env.command(ICI + f"tp @s ~ ~ ~ {yaw} 0")
    if hasattr(pol, "new_session"):
        pol.new_session()
    r = A.run_loop(env, pol, steps, 8, replan)
    r["room"] = "foret"
    A.report_and_save(r, argparse.Namespace(env="live", policy=nom, tag=tag), replan)
    if r["steps"] < steps:
        raise AU.Stop(f"arret apres {r['steps']}/{steps} pas")
    print(f"troncs casses : {V.troncs_casses(r['truth'])}", flush=True)
    return V.troncs_casses(r["truth"]), r["frames"][0]


def collecte(env, n, steps, essai, tag):
    # coups de 15 ticks (un tronc casse en 8) ; une decision par image, comme la collecte sous terre
    pol = A.ProbePolicy(hold=15, seed=int(time.time()) % 10_000)
    tag = "dc_foret_essai" if essai else tag
    for i in range(n):
        print(f"\n=== collecte {i + 1}/{n} ===", flush=True)
        vues = nouvel_endroit(env)
        # sonde seulement : on part face a un tronc, sinon presque aucun coup ne touche du bois
        joue(env, pol, steps, "probe", tag, yaw=vues[0], replan=1)


def arene(env, joueurs, agent, manches, steps, essai):
    tag = "dc_foret_essai" if essai else "dc_foret_arene"
    scores, ecarts = {nom: [] for nom in joueurs}, []
    try:
        for manche in range(manches):
            nouvel_endroit(env)
            noms = list(joueurs)
            noms = noms[manche % len(noms):] + noms[:manche % len(noms)]
            premieres = []
            for nom in noms:
                print(f"\n=== manche {manche + 1}/{manches} : {nom} ===", flush=True)
                n, f0 = joue(env, joueurs[nom], steps, nom, tag)
                scores[nom].append(n)
                premieres.append(f0.astype(np.float32))
            ecarts.append(float(max(np.abs(f - premieres[0]).mean() for f in premieres)))
            print(f"\n[manche {manche + 1}] troncs : { {k: v[-1] for k, v in scores.items()} } "
                  f"| ecart max des images de depart {ecarts[-1]:.1f}/255", flush=True)
    finally:
        fini = min(len(v) for v in scores.values())
        scores = {k: v[:fini] for k, v in scores.items()}
        res = {"manches": fini, "pas": steps, "essai": essai, "scores": scores,
               "ecart_images_depart": ecarts[:fini], "contre": {}}
        print(f"\n{fini} manches completes")
        for nom, v in scores.items():
            print(f"  {nom:<15} {np.mean(v) if v else 0:5.1f} troncs en moyenne  {v}")
        if agent in scores and fini:
            for ref in scores:
                if ref == agent:
                    continue
                g, n, p = AR.test_de_signe(scores[agent], scores[ref])
                res["contre"][ref] = {"gagnees": g, "decisives": n, "p": p, "battu": bool(p < 0.05)}
                print(f"  {agent} contre {ref:<15} gagne {g}/{n} manches decisives, p = {p:.3f} -> "
                      f"{'BATTU' if p < 0.05 else 'pas prouve'}")
        OUT.mkdir(parents=True, exist_ok=True)
        sortie = OUT / ("essai.json" if essai else f"resultats_{time.strftime('%Y%m%d-%H%M%S')}.json")
        sortie.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"-> {sortie}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collecte", type=int, default=0, help="sessions de sonde")
    ap.add_argument("--manches", type=int, default=0, help="manches d'arene")
    ap.add_argument("--pas", type=int, default=1200, help="pas par session ou par joueur (20 par seconde)")
    ap.add_argument("--tag", default="dc_foret_probe", help="nom des sessions de sonde (vt_foret : test)")
    ap.add_argument("--essai", action="store_true", help="1 session de sonde + 1 manche de 5 s, rien de conclu")
    ap.add_argument("--joueurs", default="", help="liste separee par des virgules (defaut : tous)")
    ap.add_argument("--agent", default="bucheron", help="joueur teste par la regle")
    ap.add_argument("--attente", type=int, default=10, help="secondes laissees pour cliquer dans le jeu")
    args = ap.parse_args()
    if args.essai:
        args.collecte, args.manches, args.pas = 1, 1, 100
    AU.attendre_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    joueurs = {"mine_et_balaye": A.ScriptedPolicy(), "hasard": A.RandomPolicy(), "taper": A.TaperPolicy()}
    if DETECTEUR_BOIS.exists():
        joueurs = {"bucheron": AR.DetecteurPolicy(device, verifie=True, modele=DETECTEUR_BOIS, cibles=CIBLES_FORET),
                   **joueurs}
    if DETECTEUR_BOIS_V3.exists():
        joueurs = {"bucheron_appris": AR.DetecteurPolicy(device, verifie=True, modele=DETECTEUR_BOIS_V3,
                                                         cibles=CIBLES_FORET), **joueurs}
    if args.joueurs:
        joueurs = {k: joueurs[k] for k in args.joueurs.split(",")}

    try:
        env = A.LiveEnv()
    except RuntimeError as e:
        sys.exit(f"ARRET : {e}")
    print(f"[foret] clique dans le jeu, tu as {args.attente} s...", flush=True)
    t = time.time()
    while not env.focused() and time.time() - t < args.attente:
        time.sleep(0.1)
    if not env.focused():
        sys.exit("Minecraft n'est pas au premier plan. Rien n'a ete envoye.")
    time.sleep(3)

    AU.garde_eveille(True)
    try:
        env.start_truth()
        env.setup_world()
        if args.collecte:
            collecte(env, args.collecte, args.pas, args.essai, args.tag)
        if args.manches:
            arene(env, joueurs, args.agent, args.manches, args.pas, args.essai)
    except (AU.Stop, RuntimeError) as e:
        print(f"\narret : {e}")
    except KeyboardInterrupt:
        print("\narret demande")
    finally:
        env.close()
        AU.garde_eveille(False)


if __name__ == "__main__":
    main()
