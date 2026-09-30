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

    Cycle, sans jamais bouger de place :
      A. TAPER pendant `hold` frames   -> le bloc casse
      B. NE RIEN FAIRE pendant `hold`  -> la meme scene, sans le bouton
      C. petit pivot camera -> nouveau bloc, on recommence

    A et B partagent le meme point de vue : la SEULE chose qui differe est
    l appui sur le bouton. C est exactement le signal que le modele n a jamais
    vu proprement (dans YouTube l action est devinee par l IDM et confondue
    avec le mouvement) et qui manque au chooseur.
    """

    def __init__(self, hold=30, pivot=10):
        self.hold, self.pivot = hold, pivot
        self.t = 0

    def plan(self, ctx_u8, h):
        v = np.zeros((h, 10), dtype=np.float32)
        cycle = 2 * self.hold + self.pivot   # hold en FRAMES (30 = 1.5 s a 20 Hz)
        phase = self.t % cycle
        if phase < self.hold:
            v[:, IDX["taper"]] = 1.0
            name = "A_taper"
        elif phase < 2 * self.hold:
            name = "B_temoin"
        else:
            v[:, CAM_DX] = 0.25
            name = "C_pivot"
        self.phase_name = name
        return v, name

    def advance(self, n):
        self.t += n            # compte les pas REELLEMENT joues, pas l horizon


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
        import pygetwindow as gw
        cands = [w for w in gw.getAllWindows()
                 if w.title and "minecraft" in w.title.lower() and w.width > 200]
        if not cands:
            raise RuntimeError("aucune fenetre Minecraft (jeu lance ? mode FENETRE ?)")
        w = max(cands, key=lambda x: x.width * x.height)
        self.gw = gw
        self.sct = mss.mss()
        self.box = {"left": w.left + 8, "top": w.top + 31,
                    "width": max(w.width - 16, 64), "height": max(w.height - 39, 64)}
        self.held = set()
        print(f"[live] '{w.title}' -> capture {self.box['width']}x{self.box['height']}")

    def focused(self):
        a = self.gw.getActiveWindow()
        return a is not None and "minecraft" in (a.title or "").lower()

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


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["replay", "live"], default="replay")
    ap.add_argument("--policy", choices=["scripted", "chooser", "random", "probe"],
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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    pol = {"scripted": lambda: ScriptedPolicy(),
           "random": lambda: RandomPolicy(),
           "probe": lambda: ProbePolicy(),
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
        if not env.focused():
            print("ERREUR : Minecraft n'est pas au premier plan. Clique dans le jeu.")
            return
        print("[live] demarrage dans 3 s -- passe sur une autre fenetre pour tout arreter.")
        time.sleep(3)

    frames, actions, decisions, lat, phases = [], [], [], [], []
    obs = env.observe()
    ctx = np.stack([obs] * CTX)
    t0 = time.time()
    step = 0
    try:
        while step < args.steps:
            d0 = time.perf_counter()
            plan, name = pol.plan(ctx, args.horizon)
            lat.append((time.perf_counter() - d0) * 1000)
            decisions.append(name)
            if args.verbose and hasattr(pol, "last") and len(decisions) <= 5:
                L = pol.last
                print(f"  decision {len(decisions)} -> {name}")
                for j, nm in enumerate(L["names"]):
                    print(f"    {nm:<14} ratio {L['ratio'][j]:.3f}  goal {L['goal'][j]:.3f}"
                          f"  = {L['score'][j]:.3f}")
            for k in range(min(args.replan, len(plan))):
                if step >= args.steps:
                    break
                a = plan[k]
                frames.append(obs)
                actions.append(a.copy())
                phases.append(name)
                env.act(a)
                time.sleep(max(0.0, t0 + (step + 1) / FPS - time.time()))
                obs = env.observe()
                ctx = np.concatenate([ctx[1:], obs[None]])
                step += 1
            if hasattr(pol, "advance"):
                pol.advance(min(args.replan, len(plan)))
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\narret : {e}")
    finally:
        env.close()

    dur = time.time() - t0
    print(f"\n{step} pas en {dur:.1f}s = {step / max(dur, 1e-9):.1f} Hz (cible {FPS:.0f})")
    print(f"latence de decision : mediane {np.median(lat):.1f} ms | max {np.max(lat):.1f} ms "
          f"(budget {1000 * args.replan / FPS:.0f} ms)")
    print(f"plans choisis : {dict(Counter(decisions).most_common())}")
    print(f"taux taper : {np.mean([a[IDX['taper']] for a in actions]):.0%}")

    if args.env == "live" and len(frames) > 50:
        # donnees PARFAITES : labels exacts (on a envoye ces touches) + domaine exact
        arr = np.stack(frames)
        acts = np.stack(actions[1:]) if len(actions) > 1 else np.zeros((0, 10), np.float32)
        p = ROOT / "data" / "processed_agent" / f"{args.tag}_{args.policy}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, frames=arr, actions=acts,
                            gui=np.zeros(len(acts), np.uint8),
                            phase=np.array(phases[:len(acts)]))
        print(f"donnees collectees -> {p.relative_to(ROOT)} ({len(arr)} frames)")


if __name__ == "__main__":
    main()
