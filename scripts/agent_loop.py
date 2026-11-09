"""
Boucle agent : observer -> rever -> choisir -> agir -> recommencer.

Deux usages, un seul code :
  1. COLLECTER des donnees parfaites (labels d'actions EXACTS + domaine EXACT)
     pour ameliorer le modele -> le carburant que les lecons 18-19 reclament.
  2. DEMONTRER le Monde 2 : le modele choisit et joue tout seul.

Politiques interchangeables (--policy) :
  scripted : mine devant soi + balaye la camera. Garantit des donnees RICHES en
             evenements des le 1er run (evite le demarrage a froid : un agent
             timide collecterait des donnees ennuyeuses).
  chooser  : le VRAI agent. Reve les plans candidats, score le but SUR LES REVES,
             joue le meilleur. Horizon glissant (joue REPLAN pas puis re-decide)
             = l'intuition de depart du projet.
  random   : baseline de comparaison.

Backends (--env) :
  replay : A VIDE sur un episode enregistre. Valide la logique + mesure la latence
           SANS lancer le jeu. Les actions sont calculees, pas executees.
  live   : le vrai Minecraft (capture ecran + injection de touches).

Securites (live) : n'agit que si Minecraft est au premier plan, duree bornee,
relache TOUTES les touches en sortant (y compris sur Ctrl+C / erreur).

Test a vide :
  .venv\\Scripts\\python.exe scripts\\agent_loop.py --env replay --policy chooser --steps 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft import game_window as GW  # noqa: E402
from dreamcraft import verite as V  # noqa: E402
from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

import importlib.util as _ilu
_g = _ilu.spec_from_file_location("tgd", Path(__file__).parent / "train_goal_detector.py")
_tgd = _ilu.module_from_spec(_g)
_g.loader.exec_module(_tgd)

OUT = ROOT / "outputs" / "agent"
RES, FPS, CTX = 64, 20.0, 4
IDX = {"avancer": 0, "reculer": 1, "gauche": 2, "droite": 3,
       "sauter": 4, "sneak": 5, "taper": 6, "utiliser": 7}
CAM_DX, CAM_DY = 8, 9
CAM_CLIP = 60.0                      # meme normalisation qu'a l'entrainement

PLANS = {
    "taper":         {"taper": 1.0},
    "taper_avancer": {"taper": 1.0, "avancer": 1.0},
    "avancer":       {"avancer": 1.0},
    "rien":          {},
    "cam_gauche":    {"cam_dx": -0.3},
    "cam_droite":    {"cam_dx": 0.3},
}


# Preparation d'un monde (triche activee). Le retour des commandes est coupe en
# premier : aucun message de chat ne doit apparaitre sur les images collectees.
SETUP_COMMANDS = [
    "/gamerule sendCommandFeedback false",
    "/difficulty peaceful",
    "/gamerule doDaylightCycle false",
    "/time set noon",
    "/weather clear 999999",
    "/gamerule doImmediateRespawn true",
    "/gamerule keepInventory true",
    "/effect give @s minecraft:night_vision 999999 0 true",
    "/effect give @s minecraft:resistance 999999 255 true",
    "/effect give @s minecraft:fire_resistance 999999 0 true",
    "/effect give @s minecraft:water_breathing 999999 0 true",
]
PIOCHE = "/item replace entity @s weapon.mainhand with minecraft:diamond_pickaxe{Unbreakable:1b}"


def plan_vector(spec: dict, h: int) -> np.ndarray:
    v = np.zeros((h, 10), dtype=np.float32)
    for k, val in spec.items():
        if k == "cam_dx":
            v[:, CAM_DX] = val
        else:
            v[:, IDX[k]] = val
    return v


# ---------------------------------------------------------------- politiques
class ScriptedPolicy:
    """Mine devant soi, balaye la camera. Donnees riches en evenements garanties."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def plan(self, ctx_u8, h):
        self.t += 1
        v = np.zeros((h, 10), dtype=np.float32)
        if (self.t // 10) % 4 < 3:          # ~3/4 du temps : minage
            v[:, IDX["taper"]] = 1.0
        else:                                # reorientation
            v[:, CAM_DX] = float(self.rng.choice([-0.35, 0.35]))
            if self.rng.random() < 0.5:
                v[:, IDX["avancer"]] = 1.0
        return v, "scripted"


class ProbePolicy:
    """EXPERIENCE CONTROLEE : la seule politique qui produit des PAIRES.

    Cycle (hold en frames, 30 = 1.5 s a 20 Hz) :
      B. NE RIEN FAIRE pendant `hold`  -> la scene, sans le bouton
      A. TAPER pendant `hold`          -> la MEME scene de depart, avec le bouton
      C. pivot camera aleatoire        -> nouveau point de vue
      D. un cycle sur `walk_every`, marcher -> nouvel endroit

    Le temoin passe AVANT le coup : si le bloc etait casse d'abord, le temoin
    regarderait un trou et la paire ne comparerait plus la meme scene.

    La camera ne bouge ni en A ni en B : le label "ce coup a-t-il servi ?" ne
    peut donc pas etre pollue par le mouvement du joueur, contrairement aux
    videos. Certains coups tombent dans le vide (ciel, bloc hors de portee) :
    sans ces cas negatifs, on ne pourrait pas tester si le modele sait ou taper.
    """

    def __init__(self, hold=30, pivot=10, walk=40, walk_every=6, seed=0):
        self.hold, self.pivot, self.walk, self.walk_every = hold, pivot, walk, walk_every
        self.rng = np.random.default_rng(seed)
        self.pitch = 0.0          # inclinaison cumulee en pixels souris (+ = vers le bas)
        self.cycle = 0
        self.queue: list[tuple[str, np.ndarray]] = []

    def _action(self, **values) -> np.ndarray:
        a = np.zeros(10, dtype=np.float32)
        for key, val in values.items():
            a[{"cam_dx": CAM_DX, "cam_dy": CAM_DY}.get(key, IDX.get(key, -1))] = val
        return a

    def _next_cycle(self) -> None:
        self.queue += [("B_temoin", self._action())] * self.hold
        self.queue += [("A_taper", self._action(taper=1.0))] * self.hold
        # 15 % des visees vers le haut, le reste entre l'horizon et le sol
        target = (-self.rng.uniform(150, 350) if self.rng.random() < 0.15
                  else self.rng.uniform(-40, 180))
        dy = float(np.clip((target - self.pitch) / (self.pivot * CAM_CLIP), -1, 1))
        dx = float(self.rng.uniform(0.15, 0.5) * self.rng.choice([-1, 1]))
        self.pitch += dy * self.pivot * CAM_CLIP
        self.queue += [("C_pivot", self._action(cam_dx=dx, cam_dy=dy))] * self.pivot
        self.cycle += 1
        if self.cycle % self.walk_every == 0:
            self.queue += [("D_marche", self._action(avancer=1.0, sauter=1.0))] * self.walk

    def plan(self, ctx_u8, h):
        if not self.queue:
            self._next_cycle()
        name, a = self.queue.pop(0)
        return np.repeat(a[None], h, axis=0), name

    def new_session(self):
        """Nouvelle salle : le cycle repart de zero, le regard est remis a l'horizon."""
        self.queue.clear()
        self.pitch = 0.0


class TaperPolicy:
    """Reference : taper tout le temps, sans bouger."""

    def plan(self, ctx_u8, h):
        return plan_vector(PLANS["taper"], h), "taper"


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def plan(self, ctx_u8, h):
        name = str(self.rng.choice(list(PLANS)))
        return plan_vector(PLANS[name], h), f"random:{name}"


class ChooserPolicy:
    """Le vrai agent : reve chaque plan, score le but SUR LE REVE, joue le meilleur."""

    def __init__(self, ckpt, goal_ckpt, device, horizon, mode="jumeau"):
        cfg = {"base": 32, "in_frames": 1}
        cf = Path(ckpt).parent / "config.json"
        if cf.exists():
            cfg.update(json.loads(cf.read_text(encoding="utf-8")))
        self.model = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
        self.model.load_state_dict(torch.load(ckpt, weights_only=True))
        self.model.eval()
        self.nctx = cfg["in_frames"]
        self.device = device
        self.h = horizon
        self.mode = mode
        self.names = list(PLANS)
        self.plans = np.stack([plan_vector(PLANS[n], horizon) for n in self.names])
        self.twins = self.plans.copy()          # meme plan, boutons d'interaction coupes
        self.twins[:, :, IDX["taper"]] = 0.0
        self.twins[:, :, IDX["utiliser"]] = 0.0
        self.goal = None
        if goal_ckpt and Path(goal_ckpt).exists():
            self.goal = _tgd.GoalNet().to(device)
            self.goal.load_state_dict(torch.load(goal_ckpt, weights_only=True))
            self.goal.eval()
        cm = torch.zeros(RES, RES, dtype=torch.bool, device=device)
        cm[26:38, 26:38] = True
        self.cm = cm
        self.wgoal = 0.0 if mode == "jumeau" else 1.0
        self.eps = 2e-3          # seuil d effet du bouton juge exploitable

    @torch.no_grad()
    def plan(self, ctx_u8, h):
        """Reve chaque plan ET son jumeau sans interaction. L ecart = l effet du bouton."""
        n = len(self.names)
        recent = ctx_u8[-self.nctx:]
        stack = np.concatenate([f.transpose(2, 0, 1) for f in recent], axis=0)[None]
        both = np.concatenate([self.plans, self.twins])          # (2n, H, 10)
        x = torch.from_numpy(np.repeat(stack, 2 * n, 0)).to(self.device).float() / 255.0
        start = x[:, -3:].clone()
        dreams = []
        for step in range(self.h):
            pred = self.model(x, torch.from_numpy(both[:, step]).to(self.device))
            dreams.append(pred)
            x = C.roll_context(x, pred)
        d = (dreams[-1] - start).abs().mean(1)
        c = d[:, self.cm].mean(1)
        p = d[:, ~self.cm].mean(1)
        if self.mode == "ratio":
            base = (c / (c + p + 1e-6))[:n]
        else:
            base = c[:n] - c[n:]                                 # effet propre du bouton
        score = base.clone()
        goal = torch.zeros_like(base)
        if self.goal is not None:
            seq = torch.stack([start[:n]] + [dd[:n] for dd in dreams], dim=1)
            if seq.shape[1] >= 5:
                wins = torch.stack([seq[:, k:k + 5].flatten(1, 2)
                                    for k in range(seq.shape[1] - 4)], dim=1)
                b_, w = wins.shape[:2]
                probs = torch.sigmoid(self.goal(wins.flatten(0, 1))).view(b_, w)
                goal = probs.topk(min(3, w), dim=1).values.mean(1)
                score = base + self.wgoal * goal
        inter = torch.tensor([self.plans[j, :, IDX["taper"]].max() > 0.5
                              or self.plans[j, :, IDX["utiliser"]].max() > 0.5
                              for j in range(n)], device=score.device)
        if self.mode == "jumeau":
            # etage 1 : taper ne paie que si l effet propre du bouton est net
            if bool(inter.any()) and float(score[inter].max()) > self.eps:
                m = torch.where(inter, score, torch.full_like(score, -1e9))
            else:
                # etage 2 : rien a casser ici -> on se deplace pour trouver mieux
                nav = (c / (c + p + 1e-6))[:n]
                m = torch.where(inter, torch.full_like(nav, -1e9), nav)
            i = int(m.argmax())
        else:
            i = int(score.argmax())
        self.last = {"names": self.names, "ratio": base.tolist(),
                     "goal": goal.tolist(), "score": score.tolist()}
        return self.plans[i], self.names[i]


# ------------------------------------------------------------------ backends
class ReplayEnv:
    """A VIDE : rejoue un episode enregistre. Actions calculees, pas executees."""

    def __init__(self, stem):
        for base in ["processed", "processed_youtube"]:
            p = ROOT / "data" / base / f"{stem}.npz"
            if p.exists():
                self.frames = np.load(p)["frames"]
                break
        else:
            raise FileNotFoundError(stem)
        self.i = CTX

    def observe(self):
        return self.frames[min(self.i, len(self.frames) - 1)]

    def act(self, a):
        self.i += 1              # le monde avance, independamment de nous

    def focused(self):
        return True

    def close(self):
        pass


class LiveEnv:
    """Le vrai jeu : capture mss + touches DirectInput, seulement si Minecraft a le focus."""

    def __init__(self):
        import mss
        found = GW.find_game_window()
        if found is None:
            raise RuntimeError("aucune fenetre Minecraft (jeu lance ? mode FENETRE ?)")
        self.hwnd, title = found
        self.sct = mss.mss()
        self.box = GW.client_box(self.hwnd)
        self.held = set()
        self.journal = None
        w, h = self.box["width"], self.box["height"]
        print(f"[live] '{title}' -> capture {w}x{h}")
        if abs(w / max(h, 1) - 16 / 9) > 0.05:
            print("[live] ATTENTION : fenetre pas en 16:9, l'image sera deformee autrement "
                  "qu'a l'entrainement")

    def focused(self):
        return GW.is_foreground(self.hwnd)

    def observe(self):
        img = np.array(self.sct.grab(self.box))[:, :, :3]
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (RES, RES),
                          interpolation=cv2.INTER_AREA)

    def act(self, a):
        import ctypes

        import pydirectinput
        pydirectinput.PAUSE = 0.0
        if not self.focused():
            self.release_all()
            raise RuntimeError("Minecraft a perdu le focus -> arret de securite")
        keys = {"avancer": "w", "reculer": "s", "gauche": "a", "droite": "d",
                "sauter": "space", "sneak": "shift"}
        for name, key in keys.items():
            want = a[IDX[name]] > 0.5
            if want and key not in self.held:
                pydirectinput.keyDown(key)
                self.held.add(key)
            elif not want and key in self.held:
                pydirectinput.keyUp(key)
                self.held.discard(key)
        for name, btn in [("taper", "left"), ("utiliser", "right")]:
            want = a[IDX[name]] > 0.5
            if want and btn not in self.held:
                pydirectinput.mouseDown(button=btn)
                self.held.add(btn)
            elif not want and btn in self.held:
                pydirectinput.mouseUp(button=btn)
                self.held.discard(btn)
        dx = int(round(float(a[CAM_DX]) * CAM_CLIP))
        dy = int(round(float(a[CAM_DY]) * CAM_CLIP))
        if dx or dy:
            ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)

    def command(self, text):
        """Tape une commande dans le chat. Par copier-coller : '/' et '~' changent de
        touche selon le clavier (AZERTY, QWERTY), pas 't', 'v', Ctrl ni Entree."""
        import subprocess

        import pydirectinput
        if not self.focused():
            raise RuntimeError("Minecraft a perdu le focus -> arret de securite")
        subprocess.run(["clip"], input=text.encode("ascii"), check=True)

        def tap(key):
            pydirectinput.keyDown(key)
            time.sleep(0.05)
            pydirectinput.keyUp(key)
            time.sleep(0.2)

        tap("t")                          # ouvre le chat
        pydirectinput.keyDown("ctrl")
        tap("v")
        pydirectinput.keyUp("ctrl")
        tap("enter")

    def setup_world(self):
        for c in SETUP_COMMANDS:
            self.command(c)

    def new_room(self, block=None):
        """Une salle 60 blocs plus loin, regard a l'horizon, inventaire vide.
        block : murs d'un seul type, assez epais pour ne pas voir le terrain derriere."""
        self.command("/tp @s ~60 30 ~ ~ 0")
        time.sleep(2.0)                   # le temps que le terrain charge, sinon /fill echoue
        if block:
            self.command(f"/fill ~-12 ~-3 ~-12 ~12 ~6 ~12 minecraft:{block}")
        self.command("/fill ~-9 ~ ~-9 ~9 ~3 ~9 air")
        self.command("/clear @s")
        self.command(PIOCHE)
        time.sleep(1.5)

    def start_truth(self):
        """Refuse de collecter sans verite : on le saurait trop tard, apres des heures de jeu."""
        if not V.chat_masque():
            raise RuntimeError("taille du chat differente de 0 : les lignes de verite "
                               "saliraient les images (scripts/installe_verite.py, jeu ferme)")
        self.journal = V.JournalJeu(V.dossier_jeu() / "logs" / "latest.log")
        t0 = time.time()
        while time.time() - t0 < 5:
            if self.read_truth()[0] >= 0:
                return
            time.sleep(0.1)
        raise RuntimeError("aucune ligne de verite dans le journal du jeu : datapack absent "
                           "de ce monde ? (scripts/installe_verite.py, puis rouvrir le monde)")

    def read_truth(self):
        return None if self.journal is None else self.journal.lire()

    def release_all(self):
        try:
            import pydirectinput
            for k in list(self.held):
                if k in ("left", "right"):
                    pydirectinput.mouseUp(button=k)
                else:
                    pydirectinput.keyUp(k)
            self.held.clear()
        except Exception:
            pass

    def close(self):
        self.release_all()


# ----------------------------------------------------------------------- boucle
def run_loop(env, pol, steps, horizon, replan, fps=FPS, verbose=False):
    """Observer -> choisir -> agir. Convention des donnees : actions[t] mene de frames[t] a frames[t+1]."""
    frames, actions, phases, decisions, lat = [], [], [], [], []
    read_truth = getattr(env, "read_truth", lambda: None)
    obs = env.observe()
    truth = [read_truth()]                       # truth[t] : l'etat du jeu a l'image t
    ctx = np.stack([obs] * CTX)
    t0 = time.time()
    step = 0
    try:
        while step < steps:
            d0 = time.perf_counter()
            plan, name = pol.plan(ctx, horizon)
            lat.append((time.perf_counter() - d0) * 1000)
            decisions.append(name)
            if verbose and hasattr(pol, "last") and len(decisions) <= 5:
                L = pol.last
                print(f"  decision {len(decisions)} -> {name}")
                for j, nm in enumerate(L["names"]):
                    print(f"    {nm:<14} ratio {L['ratio'][j]:.3f}  goal {L['goal'][j]:.3f}"
                          f"  = {L['score'][j]:.3f}")
            for k in range(min(replan, len(plan))):
                if step >= steps:
                    break
                frames.append(obs)
                env.act(plan[k])                 # leve une erreur si le jeu perd le focus
                actions.append(plan[k].copy())   # seulement ce qui a vraiment ete envoye
                phases.append(name)
                if fps:
                    time.sleep(max(0.0, t0 + (step + 1) / fps - time.time()))
                obs = env.observe()
                truth.append(read_truth())
                ctx = np.concatenate([ctx[1:], obs[None]])
                step += 1
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\narret : {e}")
    finally:
        env.close()
    if len(frames) == len(actions):
        frames.append(obs)                       # l'image qui suit la derniere action
    truth = np.stack(truth[:len(frames)]) if truth[0] is not None else None
    return {"frames": frames, "actions": actions, "phases": phases, "decisions": decisions,
            "lat": lat, "steps": step, "duration": time.time() - t0, "truth": truth}


def save_episode(path, frames, actions, phases, truth=None, room=None):
    """Meme format que le corpus : N images, N-1 actions, ecriture atomique."""
    if len(frames) != len(actions) + 1:
        raise ValueError(f"{len(frames)} images pour {len(actions)} actions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.part")
    with open(tmp, "wb") as f:
        extra = {} if truth is None else {"truth": truth, "room": np.array(room or "naturel")}
        np.savez_compressed(f, frames=np.stack(frames), actions=np.stack(actions),
                            gui=np.zeros(len(actions), np.uint8), phase=np.array(phases), **extra)
    tmp.replace(path)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["replay", "live"], default="replay")
    ap.add_argument("--policy", choices=["scripted", "chooser", "random", "probe", "taper"],
                    default="chooser")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--replan", type=int, default=4, help="pas joues avant de re-decider")
    ap.add_argument("--episode", default="strip_051")
    ap.add_argument("--ckpt", default=str(ROOT / "outputs" / "v13b-indomain" / "model_best.pt"))
    ap.add_argument("--goal", default=str(ROOT / "outputs" / "goal_detector" / "model_best.pt"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--verbose", action="store_true", help="detail du score par plan")
    ap.add_argument("--score", choices=["jumeau", "ratio"], default="jumeau")
    ap.add_argument("--sessions", type=int, default=0,
                    help="sonde en jeu : prepare le monde et enchaine N sessions, une salle chacune")
    ap.add_argument("--salles", choices=["naturel", "mixte"], default="naturel",
                    help="mixte : une session sur deux dans une salle d'un seul type de bloc")
    ap.add_argument("--attente", type=int, default=10, help="secondes laissees pour cliquer dans le jeu")
    args = ap.parse_args()
    if args.sessions and (args.env != "live" or args.policy != "probe"):
        ap.error("--sessions demande --env live --policy probe")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    pol = {"scripted": lambda: ScriptedPolicy(),
           "random": lambda: RandomPolicy(),
           "probe": lambda: ProbePolicy(),
           "taper": lambda: TaperPolicy(),
           "chooser": lambda: ChooserPolicy(args.ckpt, args.goal, device,
                                            args.horizon, args.score)}[args.policy]()
    try:
        env = ReplayEnv(args.episode) if args.env == "replay" else LiveEnv()
    except RuntimeError as e:
        print(f"ARRET : {e}")
        print("  1. lance Minecraft Java 1.19.x, vanilla, en mode FENETRE")
        print("  2. charge un monde et clique dedans (curseur capture)")
        print("  3. relance cette commande")
        return

    if args.env == "live":
        print(f"[live] clique dans le jeu, tu as {args.attente} s (inventaire et menus fermes)...", flush=True)
        t_wait = time.time()
        while not env.focused() and time.time() - t_wait < args.attente:
            time.sleep(0.1)
        if not env.focused():
            print("ERREUR : Minecraft n'est pas au premier plan. Rien n'a ete envoye.")
            return
        print("[live] demarrage dans 3 s -- passe sur une autre fenetre pour tout arreter.")
        time.sleep(3)

    # la sonde decide a chaque image, pour que ses phases aient la duree exacte
    replan = 1 if args.policy == "probe" else args.replan

    if args.sessions:
        try:
            env.start_truth()
            env.setup_world()
            for i in range(args.sessions):
                room = V.salle_de(i, args.salles)
                print(f"\n=== session {i + 1}/{args.sessions} : {room or 'naturel'} ===")
                env.new_room(room)
                pol.new_session()
                r = run_loop(env, pol, args.steps, args.horizon, replan)
                r["room"] = room
                report_and_save(r, args, replan)
                if r["steps"] < args.steps:
                    break
        except RuntimeError as e:
            print(f"\narret : {e}")
        finally:
            env.close()
        return

    r = run_loop(env, pol, args.steps, args.horizon, replan, verbose=args.verbose)
    report_and_save(r, args, replan)


def report_and_save(r, args, replan):
    actions, lat = r["actions"], r["lat"]
    if not actions:
        print("aucune action envoyee, rien a sauver")
        return

    print(f"\n{r['steps']} pas en {r['duration']:.1f}s = "
          f"{r['steps'] / max(r['duration'], 1e-9):.1f} Hz (cible {FPS:.0f})")
    print(f"latence de decision : mediane {np.median(lat):.1f} ms | max {np.max(lat):.1f} ms "
          f"(budget {1000 * replan / FPS:.0f} ms)")
    print(f"plans choisis : {dict(Counter(r['decisions']).most_common())}")
    print(f"taux taper : {np.mean([a[IDX['taper']] for a in actions]):.0%}")
    truth = r.get("truth")
    if truth is not None:
        print(f"verite : {V.blocs_casses(truth)} blocs casses | butin {truth[-1, 4]} | "
              f"{np.mean(truth[:, 0] >= 0):.0%} des images annotees")

    if args.env == "live" and len(actions) > 50:
        # labels exacts (on a envoye ces touches). Date dans le nom : jamais d'ecrasement.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        p = ROOT / "data" / "processed_agent" / f"{args.tag}_{args.policy}_{stamp}.npz"
        save_episode(p, r["frames"], actions, r["phases"], truth, r.get("room"))
        print(f"donnees collectees -> {p.relative_to(ROOT)} ({len(r['frames'])} frames)")
        return p


if __name__ == "__main__":
    main()
