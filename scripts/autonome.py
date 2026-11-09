"""
La boucle complete, sans intervention : jouer, apprendre, se mesurer, recommencer.

Chaque cycle :
  1. collecte : `--bloc` sessions de sonde, moitie terrain naturel, moitie salles d'un seul bloc.
  2. entrainement : meme programme, depuis le meme modele de depart, avec deux compositions
     au meme nombre de sessions : terrain naturel seul, ou avec les salles.
  3. mesure : eval_pairs.py sur le monde de test, qui ne change jamais et ne choisit rien.
  4. arene : l'agent joue contre trois references, score = blocs reellement casses.
Resultats : outputs/autonome/journal.jsonl, une ligne par cycle.

Le programme est fixe : aucun modele n'est choisi d'apres dc_test, donc le test reste honnete.
Les parties de l'arene ne servent pas a l'entrainement : un seul levier a la fois.

Monde dc_train ouvert, clique dans le jeu, puis :
  .venv\\Scripts\\python.exe scripts\\autonome.py --essai       (2 min, verifie toute la plomberie)
  .venv\\Scripts\\python.exe scripts\\autonome.py --heures 8

Passer sur une autre fenetre arrete tout, proprement.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util as ilu
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft import verite as V  # noqa: E402

_s = ilu.spec_from_file_location("agent", Path(__file__).parent / "agent_loop.py")
A = ilu.module_from_spec(_s)
_s.loader.exec_module(A)

OUT = ROOT / "outputs" / "autonome"
DATA = ROOT / "data" / "processed_agent"
PY = sys.executable
VRAM_MIN_MO = 3000
START_MODEL = ROOT / "outputs" / "v15-paires" / "model_best.pt"


class Stop(Exception):
    """Le jeu a perdu le focus : on arrete sans perdre ce qui est deja sauve."""


def garde_eveille(actif: bool):
    """Empeche la mise en veille tant que le script tourne, sans toucher aux reglages Windows."""
    continuous, system, display = 0x80000000, 0x1, 0x2
    ctypes.windll.kernel32.SetThreadExecutionState(
        continuous | (system | display if actif else 0))


def attendre_gpu():
    """Le GPU est partage : on attend qu'il se libere, on ne touche jamais aux autres calculs."""
    while True:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
        libre = int(r.stdout.split()[0])
        if libre >= VRAM_MIN_MO:
            return
        print(f"GPU occupe ({libre} Mo libres), nouvel essai dans 60 s", flush=True)
        time.sleep(60)


def jouer(env, pol, steps, tag, name, room, replan):
    env.new_room(room)
    if hasattr(pol, "new_session"):
        pol.new_session()
    r = A.run_loop(env, pol, steps, 8, replan)
    r["room"] = room
    A.report_and_save(r, argparse.Namespace(env="live", policy=name, tag=tag), replan)
    if r["steps"] < steps:
        raise Stop(f"arret apres {r['steps']}/{steps} pas")
    return r


def collecte(env, n, steps, tag, debut):
    pol = A.ProbePolicy(seed=debut)
    for i in range(debut, debut + n):
        room = V.salle_de(i, "mixte")
        print(f"\n=== collecte {i - debut + 1}/{n} : {room or 'naturel'} ===", flush=True)
        jouer(env, pol, steps, tag, "probe", room, 1)


def lance(cmd, envir=None):
    """Lance un calcul a part : s'il plante, la collecte de la nuit n'est pas perdue."""
    r = subprocess.run(cmd, env={**os.environ, **(envir or {})}, cwd=ROOT)
    if r.returncode:
        print(f"ECHEC ({r.returncode}) : {' '.join(map(str, cmd))}", flush=True)
    return r.returncode == 0


def sessions_train():
    """Sessions retenues, sans les deux gardees pour le suivi (meme regle que train_pairs.py)."""
    files = sorted(DATA.glob("dc_train*.npz"))
    suivi = files[:8][3::4]
    return [p for p in files if p not in suivi]


def room_of(p):
    with np.load(p) as z:
        return str(z["room"]) if "room" in z.files else "naturel"


def apprendre(cycle):
    """Deux compositions, meme nombre de sessions : seul le contenu differe."""
    files = sessions_train()
    n = sum(room_of(p) == "naturel" for p in files)
    modeles = {}
    for nom, rooms in (("naturel", "naturel"), ("mixte", "tout")):
        tag = f"auto_c{cycle:02d}_{nom}"
        attendre_gpu()
        print(f"\n=== entrainement {nom} : {n} sessions ===", flush=True)
        if lance([PY, "scripts/train_pairs.py"], {"DC_TAG": tag, "DC_ROOMS": rooms, "DC_N": str(n)}):
            modeles[nom] = ROOT / "outputs" / tag / "model_best.pt"
    return n, modeles


def mesurer(cycle, modeles):
    out = OUT / f"eval_c{cycle:02d}.json"
    cmd = [PY, "scripts/eval_pairs.py", "--out", str(out)]
    for nom, path in modeles.items():
        cmd += ["--ckpt", f"{nom}={path}"]
    attendre_gpu()
    if not lance(cmd):
        return {}
    return json.loads(out.read_text(encoding="utf-8"))["auc"]


def arene(env, ckpt, essais, steps, tag, device):
    """Chaque essai fait jouer tout le monde a la suite. Score : blocs casses par minute."""
    agent = A.ChooserPolicy(str(ckpt), str(ROOT / "outputs" / "goal_detector" / "model_best.pt"),
                            device, 8, "jumeau")
    joueurs = {"agent": agent, "mine_et_balaye": A.ScriptedPolicy(), "hasard": A.RandomPolicy(),
               "taper": A.TaperPolicy()}
    scores = {nom: [] for nom in joueurs}
    for essai in range(essais):
        noms = list(joueurs)
        noms = noms[essai % len(noms):] + noms[:essai % len(noms)]     # ordre tournant
        for nom in noms:
            print(f"\n=== arene {essai + 1}/{essais} : {nom} ===", flush=True)
            r = jouer(env, joueurs[nom], steps, tag, nom, None, 4)
            scores[nom].append(V.blocs_casses(r["truth"]) * 60 / (steps / A.FPS))
    return {nom: float(np.mean(v)) for nom, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heures", type=float, default=8.0)
    ap.add_argument("--bloc", type=int, default=24, help="sessions de collecte par cycle")
    ap.add_argument("--pas", type=int, default=2400)
    ap.add_argument("--essais", type=int, default=3, help="manches d'arene par cycle")
    ap.add_argument("--pas-arene", type=int, default=600)
    ap.add_argument("--essai", action="store_true", help="2 min pour tout verifier, rien pour l'entrainement")
    ap.add_argument("--attente", type=int, default=10, help="secondes laissees pour cliquer dans le jeu")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        env = A.LiveEnv()
    except RuntimeError as e:
        sys.exit(f"ARRET : {e}")
    print(f"[auto] clique dans le jeu, tu as {args.attente} s...", flush=True)
    t = time.time()
    while not env.focused() and time.time() - t < args.attente:
        time.sleep(0.1)
    if not env.focused():
        sys.exit("Minecraft n'est pas au premier plan. Rien n'a ete envoye.")
    time.sleep(3)

    garde_eveille(True)
    t0 = time.time()
    try:
        env.start_truth()
        env.setup_world()
        if args.essai:
            pol = A.ProbePolicy()
            jouer(env, pol, 300, "dc_essai", "probe", None, 1)
            jouer(env, pol, 300, "dc_essai", "probe", "obsidian", 1)
            print(arene(env, START_MODEL, 1, 100, "dc_essai", device))
            env.close()
            lance([PY, "scripts/analyse_verite.py", "dc_essai*"])
            return
        cycle, deja = 0, sum(1 for p in DATA.glob("dc_train*.npz") if "truth" in np.load(p).files)
        while time.time() - t0 < args.heures * 3600:
            cycle += 1
            collecte(env, args.bloc, args.pas, "dc_train", deja)
            deja += args.bloc
            n, modeles = apprendre(cycle)
            auc = mesurer(cycle, modeles)
            jeu = arene(env, modeles.get("mixte", START_MODEL), args.essais, args.pas_arene,
                        "dc_agent", device) if modeles else {}
            ligne = {"cycle": cycle, "heures": (time.time() - t0) / 3600, "sessions_par_modele": n,
                     "auc": auc, "blocs_par_minute": jeu}
            with open(OUT / "journal.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ligne) + "\n")
            print(f"\n[cycle {cycle}] {json.dumps(ligne)}", flush=True)
    except (Stop, RuntimeError) as e:
        print(f"\narret : {e}")
    except KeyboardInterrupt:
        print("\narret demande")
    finally:
        env.close()
        garde_eveille(False)


if __name__ == "__main__":
    main()
