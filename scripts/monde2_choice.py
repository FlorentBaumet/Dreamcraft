"""
MONDE 2 -- marche 1 : le CHOIX par le reve, en offline (aucun jeu pilotable requis).

Question mesuree : les reves du modele portent-ils assez d'information pour
DEPARTAGER des plans d'action ?

Protocole :
  a un instant reel t (fenetre "active" : le joueur faisait quelque chose),
  on reve H pas sous 4 plans candidats :
    C0 = le vrai plan (ce que l'humain a fait)
    C1 = camera inversee (dx -> -dx)
    C2 = taper bascule (clic gauche inverse a chaque pas)
    C3 = immobile (deplacements et camera annules)
  puis on compare chaque reve a ce qui s'est VRAIMENT passe.
  Le modele "choisit" le plan dont le reve colle le mieux a la realite.
  S'il retrouve C0, c'est que ses reves discriminent les actions -> on peut
  s'en servir pour choisir. Chance = 25 %.

Deux scoreurs (le protocole gele s'applique aussi au choix !) :
  - global : MSE de la derniere frame revee vs reelle (trichable par l'inertie)
  - delta  : MSE Delta-region moyenne par pas (l'honnete)

Sorties : outputs/monde2/  (accuracy_vs_H.png, results.json, choix_*.png)
Lance   : .venv\\Scripts\\python.exe scripts\\monde2_choice.py [ckpt] [ep1 ep2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.eval import metrics as M  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "monde2"

H_SWEEP = [2, 4, 8, 16, 32]
N_WINDOWS = 192          # fenetres testees par H et par episode
CHUNK = 256              # taille de batch GPU pour les reves
CANDIDATES = ["vrai", "camera_inversee", "taper_bascule", "immobile"]

DEFAULT_CKPT = ROOT / "outputs" / "v3b-fullcorpus" / "model_best.pt"
DEFAULT_EPS = ["Player129-f153ac423f61-20210617-173110",
               "treechop-984393664dfd-20210924-174326"]


def load_model(ckpt: Path, device: str):
    cfg = {"base": 32, "in_frames": 1}
    cfg_file = ckpt.parent / "config.json"
    if cfg_file.exists():
        cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
    model = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model, cfg["in_frames"]


def make_candidates(seq: np.ndarray) -> list[np.ndarray]:
    """seq (H,10) -> les 4 plans candidats (H,10) chacun."""
    c0 = seq.copy()
    c1 = seq.copy()
    c1[:, 8] = -c1[:, 8]                       # camera dx inversee
    c2 = seq.copy()
    c2[:, 6] = 1.0 - c2[:, 6]                  # taper bascule
    c3 = seq.copy()
    c3[:, 0:4] = 0.0                           # immobile : plus de deplacement
    c3[:, 8:10] = 0.0                          # ni de camera
    return [c0, c1, c2, c3]


def window_label(seq: np.ndarray) -> str:
    """Etiquette du comportement dominant de la fenetre (pour le breakdown)."""
    if seq[:, 6].mean() >= 0.5:
        return "taper"
    if np.abs(seq[:, 8:10]).mean() > 0.05:
        return "camera"
    if seq[:, 0:4].max(axis=1).mean() >= 0.5:
        return "deplacement"
    return "autre"


@torch.no_grad()
def dream_batch(model, frames_u8, ctx_starts, plans, ctx, device):
    """Reve les plans. plans: (B, H, 10). Retourne les frames revees (B,H,3,64,64)."""
    B, H = plans.shape[0], plans.shape[1]
    out = np.zeros((B, H, 3, 64, 64), dtype=np.float32)
    for i in range(0, B, CHUNK):
        sl = slice(i, min(i + CHUNK, B))
        x = torch.from_numpy(C.stack_context(frames_u8, ctx_starts[sl], ctx)).to(device).float() / 255.0
        p = torch.from_numpy(plans[sl]).to(device)
        for h in range(H):
            pred = model(x, p[:, h])
            out[sl, h] = pred.cpu().numpy()
            x = C.roll_context(x, pred)
    return out


def score_dreams(dreams, frames, t, H):
    """Scores par fenetre : (global, delta). Plus PETIT = meilleur.
    dreams (B,H,3,64,64) ; t (B,) instants presents."""
    B = dreams.shape[0]
    s_glob = np.zeros(B)
    s_delta = np.zeros(B)
    for b in range(B):
        real_final = frames[t[b] + H]
        s_glob[b] = float(((dreams[b, H - 1] - real_final) ** 2).mean())
        vals = []
        for h in range(1, H + 1):
            real = frames[t[b] + h]
            prev = frames[t[b] + h - 1]
            mask = np.abs(real - prev).max(axis=0) > M.CHANGE_THRESHOLD
            if int(mask.sum()) < M.MIN_CHANGED_PIXELS:
                continue
            vals.append(float(((dreams[b, h - 1] - real) ** 2)[:, mask].mean()))
        s_delta[b] = float(np.mean(vals)) if vals else s_glob[b]
    return s_glob, s_delta


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    eps = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_EPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ctx = load_model(ckpt, device)
    rng = np.random.default_rng(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {"ckpt": str(ckpt), "candidats": CANDIDATES, "episodes": {}}

    for stem in eps:
        frames_u8, actions, ok = C.load_episode_u8(PROC_DIR / f"{stem}.npz")
        frames = frames_u8.astype(np.float32) / 255.0
        res_ep = {}
        print(f"\n=== {stem} ===")

        for H in H_SWEEP:
            cand = C.valid_starts(ok, C.EVAL_CONTEXT_PAD, H)
            t_all = cand + C.EVAL_CONTEXT_PAD - 1
            # fenetres "actives" : il se passe quelque chose a discriminer
            active = []
            for s, tt in zip(cand, t_all):
                seq = actions[tt: tt + H]
                if (np.abs(seq[:, 8:10]).mean() > 0.05 or seq[:, 6].mean() >= 0.5
                        or seq[:, 0:4].max(axis=1).mean() >= 0.5):
                    active.append((s, tt))
            if len(active) < 8:
                print(f"  H={H}: trop peu de fenetres actives, saute")
                continue
            idx = rng.permutation(len(active))[: N_WINDOWS]
            wins = [active[i] for i in idx]

            # construit tous les plans : (B*4, H, 10)
            plans, ctx_starts, t_list, labels = [], [], [], []
            for s, tt in wins:
                seq = actions[tt: tt + H]
                for c in make_candidates(seq):
                    plans.append(c)
                    ctx_starts.append(tt - ctx + 1)
                    t_list.append(tt)
                labels.append(window_label(seq))
            plans = np.stack(plans).astype(np.float32)
            ctx_starts = np.array(ctx_starts)
            t_arr = np.array(t_list)

            dreams = dream_batch(model, frames_u8, ctx_starts, plans, ctx, device)
            s_glob, s_delta = score_dreams(dreams, frames, t_arr, H)

            B = len(wins)
            sg = s_glob.reshape(B, 4)
            sd = s_delta.reshape(B, 4)
            pick_g = sg.argmin(axis=1)
            pick_d = sd.argmin(axis=1)
            acc_g = float((pick_g == 0).mean())
            acc_d = float((pick_d == 0).mean())

            subsets = {}
            lab = np.array(labels)
            for name in ["taper", "camera", "deplacement", "autre"]:
                m = lab == name
                if m.sum() >= 5:
                    subsets[name] = {"n": int(m.sum()),
                                     "acc_globale": float((pick_g[m] == 0).mean()),
                                     "acc_delta": float((pick_d[m] == 0).mean())}
            res_ep[f"H{H}"] = {"n": B, "acc_globale": acc_g, "acc_delta": acc_d,
                               "subsets": subsets}
            print(f"  H={H:2d}: acc globale {acc_g:.2%} | acc delta {acc_d:.2%} "
                  f"(chance 25%, n={B})")

            # visuel pour H=8 : realite + 4 reves, choix delta encadre
            if H == 8:
                for k in range(min(2, B)):
                    tt = t_arr[k * 4]
                    rows = [np.concatenate(
                        [np.transpose(frames[tt + h], (1, 2, 0)) for h in range(1, H + 1)], axis=1)]
                    for ci in range(4):
                        d = dreams[k * 4 + ci]
                        rows.append(np.concatenate(
                            [np.transpose(d[h], (1, 2, 0)) for h in range(H)], axis=1))
                    sheet = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
                    big = cv2.resize(sheet, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                    names = ["REEL"] + CANDIDATES
                    for ri, nm in enumerate(names):
                        y0 = ri * 64 * 3
                        col = (0, 255, 0) if (ri - 1) == pick_d[k] and ri > 0 else (255, 255, 0)
                        cv2.putText(big, nm, (6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, col, 1, cv2.LINE_AA)
                        if (ri - 1) == pick_d[k] and ri > 0:
                            cv2.rectangle(big, (0, y0), (big.shape[1] - 1, y0 + 64 * 3 - 1),
                                          (0, 255, 0), 2)
                    p = OUT_DIR / f"choix_{stem[:20]}_t{tt}.png"
                    cv2.imwrite(str(p), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

        results["episodes"][stem] = res_ep

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # courbe accuracy vs H
    fig, axes = plt.subplots(1, len(results["episodes"]), figsize=(6.5 * len(results["episodes"]), 4.8),
                             squeeze=False)
    for ax, (stem, res_ep) in zip(axes[0], results["episodes"].items()):
        hs = [int(k[1:]) for k in res_ep]
        ax.plot(hs, [res_ep[f"H{h}"]["acc_delta"] * 100 for h in hs], "o-", lw=2,
                label="scoreur Δ-region (honnête)")
        ax.plot(hs, [res_ep[f"H{h}"]["acc_globale"] * 100 for h in hs], "s--", lw=2,
                label="scoreur MSE globale")
        ax.axhline(25, color="gray", ls=":", label="chance (25 %)")
        ax.set_xlabel("portée du rêve H (pas)")
        ax.set_ylabel("% de bons choix (retrouve le vrai plan)")
        ax.set_title(stem[:34])
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Monde 2, marche 1 - choisir le bon plan en rêvant chaque candidat")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_vs_H.png", dpi=130)
    print(f"\n-> {OUT_DIR.relative_to(ROOT)}\\accuracy_vs_H.png + results.json")


if __name__ == "__main__":
    main()
