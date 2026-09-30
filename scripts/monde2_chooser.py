"""
MONDE 2 -- marche 2 : le CHOOSEUR. Le modele choisit son plan vers un BUT VISIBLE.

But : "fais disparaitre ce qui est devant toi" (casser le bloc vise = le bois).
Score de but, calcule SUR LE REVE :
    S = changement persistant du CENTRE de la vue - changement de la PERIPHERIE
  (bloc casse devant : le centre change durablement, le decor autour non ;
   tourner la camera : tout change -> S ~ 0 ; ne rien faire : rien ne change.)

Menu de 6 plans constants sur H pas :
    taper / taper+avancer / avancer / rien / cam_gauche / cam_droite
Le chooseur reve les 6, score chaque reve, choisit le meilleur.

Verification contre la realite (offline, aucun jeu pilotable) :
  - POSITIFS : fenetres ou l'humain TAPAIT et ou le centre a VRAIMENT change
    durablement (il cassait un bloc devant lui). Bon choix attendu : un plan
    contenant "taper" (chance = 2/6 = 33 %).
  - NEGATIFS : fenetres sans taper. Test d'affordance : le score-taper reve
    doit etre plus bas qu'aux positifs (AUC) -- "le modele sait s'il y a
    quelque chose a casser devant".

NB : les baselines copie/flux donnent par construction le MEME score a tous
les plans (leur "reve" ignore l'action) -> cette tache exige un world model.

Sorties : outputs/monde2_chooser/  (results.json, affordance_H*.png, choix_*.png)
Lance   : .venv\\Scripts\\python.exe scripts\\monde2_chooser.py [ckpt] [ep1 ep2 ...]
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
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

import importlib.util as _ilu
_gspec = _ilu.spec_from_file_location("tgd", Path(__file__).parent / "train_goal_detector.py")
_tgd = _ilu.module_from_spec(_gspec)
_gspec.loader.exec_module(_tgd)

PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "monde2_chooser"

H_SWEEP = [8, 16, 24]
N_POS_MAX = 128
N_NEG_MAX = 128
CHUNK = 256
CENTER = 12          # demi-cote du patch central (24x24 sur 64x64)

PLANS = {
    "taper":         {"taper": 1.0},
    "taper_avancer": {"taper": 1.0, "avancer": 1.0},
    "avancer":       {"avancer": 1.0},
    "rien":          {},
    "cam_gauche":    {"cam_dx": -0.3},
    "cam_droite":    {"cam_dx": 0.3},
}
TAPER_PLANS = {0, 1}          # indices des plans contenant "taper"
_IDX = {"avancer": 0, "taper": 6, "cam_dx": 8}

DEFAULT_CKPT = ROOT / "outputs" / "v4-k16-ema" / "model_best.pt"
DEFAULT_EPS = ["Player129-f153ac423f61-20210617-173110",
               "Player871-2e9a64a90d31-20210627-154641"]


def plan_vector(spec: dict, h: int) -> np.ndarray:
    v = np.zeros((h, 10), dtype=np.float32)
    for k, val in spec.items():
        v[:, _IDX[k]] = val
    return v


def load_model(ckpt: Path, device: str):
    cfg = {"base": 32, "in_frames": 1}
    cfg_file = ckpt.parent / "config.json"
    if cfg_file.exists():
        cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


def center_mask(half: int):
    m = np.zeros((64, 64), dtype=bool)
    m[32 - half:32 + half, 32 - half:32 + half] = True
    return m


CM_LARGE = center_mask(CENTER)   # 24x24
CM_CROIX = center_mask(6)        # 12x12 : juste le bloc vise

# 4 fonctions de but candidates, calculees sur les MEMES reves.
# "diff"  = centre - peripherie (v1 : sature dans les grottes sombres)
# "ratio" = centre / (centre + peripherie) : invariant au contraste de la scene
SCORERS = ["diff", "ratio", "diff_croix", "ratio_croix"]


import os

# DC_GAMMA < 1.0 : eclaircit les images AVANT le scoring (idee :
# le probleme des grottes = contraste ecrase dans le sombre ; gamma 0.45 ~
# "full brightness"). N'affecte que la fonction de but, pas le reve.
GAMMA = float(os.environ.get("DC_GAMMA", "1.0"))


def all_scores(img_end: np.ndarray, img_start: np.ndarray) -> dict:
    if GAMMA != 1.0:
        img_end = np.power(np.clip(img_end, 0, 1), GAMMA)
        img_start = np.power(np.clip(img_start, 0, 1), GAMMA)
    d = np.abs(img_end - img_start).mean(axis=0)   # (64,64), moyenne canaux
    cl, pl = float(d[CM_LARGE].mean()), float(d[~CM_LARGE].mean())
    cc, pc = float(d[CM_CROIX].mean()), float(d[~CM_CROIX].mean())
    eps = 1e-6
    return {
        "diff": cl - pl,
        "ratio": cl / (cl + pl + eps),
        "diff_croix": cc - pc,
        "ratio_croix": cc / (cc + pc + eps),
    }


def persistent_score(img_end: np.ndarray, img_start: np.ndarray) -> float:
    """Compat : le scoreur v1 (selection des positifs)."""
    return all_scores(img_end, img_start)["diff"]


@torch.no_grad()
def dream(model, frames_u8, ctx_starts, plans, ctx, device):
    B, H = plans.shape[0], plans.shape[1]
    finals = np.zeros((B, 3, 64, 64), dtype=np.float32)
    all_frames = np.zeros((B, H, 3, 64, 64), dtype=np.float32)
    for i in range(0, B, CHUNK):
        sl = slice(i, min(i + CHUNK, B))
        x = torch.from_numpy(C.stack_context(frames_u8, ctx_starts[sl], ctx)).to(device).float() / 255.0
        p = torch.from_numpy(plans[sl]).to(device)
        for h in range(H):
            pred = model(x, p[:, h])
            all_frames[sl, h] = pred.cpu().numpy()
            x = C.roll_context(x, pred)
        finals[sl] = all_frames[sl, H - 1]
    return finals, all_frames


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    eps = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_EPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ctx = load_model(ckpt, device)
    goalnet = None
    gn_path = ROOT / "outputs" / "goal_detector" / "model_best.pt"
    if gn_path.exists() and os.environ.get("DC_GOALNET", "1") != "0":
        goalnet = _tgd.GoalNet().to(device)
        goalnet.load_state_dict(torch.load(gn_path, weights_only=True))
        goalnet.eval()
        print("scoreur APPRIS actif (goal_detector)")
    rng = np.random.default_rng(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_names = list(PLANS)

    results = {"ckpt": str(ckpt), "plans": plan_names, "episodes": {}}

    # DC_INPUT_GAMMA < 1 : eclaircit les ENTREES avant de rever (mode
    # "full brightness" -- a utiliser avec un modele entraine gamma-robuste).
    input_gamma = float(os.environ.get("DC_INPUT_GAMMA", "1.0"))

    for stem in eps:
        ep_path = PROC_DIR / f"{stem}.npz"
        if not ep_path.exists():   # episodes YouTube pseudo-etiquetes
            ep_path = ROOT / "data" / "processed_youtube" / f"{stem}.npz"
        frames_u8, actions, ok = C.load_episode_u8(ep_path)
        if input_gamma != 1.0:
            frames_u8 = (255.0 * (frames_u8.astype(np.float32) / 255.0) ** input_gamma
                         ).astype(np.uint8)
        frames = frames_u8.astype(np.float32) / 255.0
        res_ep = {}
        print(f"\n=== {stem} ===")

        for H in H_SWEEP:
            cand = C.valid_starts(ok, C.EVAL_CONTEXT_PAD, H)
            t_all = cand + C.EVAL_CONTEXT_PAD - 1

            pos, neg = [], []
            for tt in t_all:
                seq = actions[tt: tt + H]
                real_S = persistent_score(frames[tt + H], frames[tt])
                if seq[:, 6].mean() >= 0.8 and real_S > 0.02:
                    pos.append(tt)          # il tapait ET le centre a change
                elif seq[:, 6].max() < 0.5:
                    neg.append(tt)          # il ne tapait pas du tout
            if len(pos) < 15:
                print(f"  H={H}: {len(pos)} positifs (<15), saute")
                continue
            pos = list(rng.permutation(pos)[:N_POS_MAX])
            neg = list(rng.permutation(neg)[:N_NEG_MAX])
            moments = pos + neg
            is_pos = np.array([1] * len(pos) + [0] * len(neg))

            # reve les 6 plans pour chaque moment
            plans_np, ctx_starts, starts_ref = [], [], []
            for tt in moments:
                for name in plan_names:
                    plans_np.append(plan_vector(PLANS[name], H))
                    ctx_starts.append(tt - ctx + 1)
                    starts_ref.append(tt)
            plans_np = np.stack(plans_np)
            ctx_starts = np.array(ctx_starts)
            starts_ref = np.array(starts_ref)

            finals, dreams = dream(model, frames_u8, ctx_starts, plans_np, ctx, device)

            raw = [all_scores(finals[i], frames[starts_ref[i]]) for i in range(len(finals))]
            S_by = {sc: np.array([r[sc] for r in raw]).reshape(len(moments), len(plan_names))
                    for sc in SCORERS}

            scorers_run = list(SCORERS)
            if goalnet is not None:
                # score appris : max sur les pas du reve de P(un bloc vient de casser),
                # fenetres 5 frames (2 reelles de contexte + le reve)
                B = len(finals)
                s_appris = np.zeros(B, dtype=np.float32)
                with torch.no_grad():
                    for i in range(B):
                        tt = starts_ref[i]
                        seq = np.concatenate([frames[tt - 1][None], frames[tt][None],
                                              dreams[i]], axis=0)   # (H+2,3,64,64)
                        wins = np.stack([seq[c - 2: c + 3].reshape(15, 64, 64)
                                         for c in range(2, seq.shape[0] - 2)])
                        xb = torch.from_numpy(wins).to(device)
                        probs = torch.sigmoid(goalnet(xb))
                        k = min(3, probs.numel())
                        s_appris[i] = float(probs.topk(k).values.mean())
                S_by["appris"] = s_appris.reshape(len(moments), len(plan_names))
                scorers_run.append("appris")
                # ENSEMBLE : moyenne des rangs appris + ratio (les deux juges
                # sont complementaires : net/court vs flou/loin/sombre)
                def _ranks(M):
                    return M.argsort(axis=1).argsort(axis=1).astype(np.float32)
                S_by["ensemble"] = _ranks(S_by["appris"]) + _ranks(S_by["ratio"])
                scorers_run.append("ensemble")

            res_h = {"n_pos": len(pos), "n_neg": len(neg), "chance": 2 / 6, "scoreurs": {}}
            line = f"  H={H:2d}:"
            for sc in scorers_run:
                S = S_by[sc]
                picks_sc = S.argmax(axis=1)
                acc_pos = float(np.isin(picks_sc[is_pos == 1], list(TAPER_PLANS)).mean())
                s_taper = S[:, 0]
                sp, sn = s_taper[is_pos == 1], s_taper[is_pos == 0]
                auc = float((sp[:, None] > sn[None, :]).mean()
                            + 0.5 * (sp[:, None] == sn[None, :]).mean())
                res_h["scoreurs"][sc] = {"choix_taper_sur_positifs": acc_pos,
                                         "auc_affordance": auc}
                line += f"  {sc} {acc_pos:.0%}/{auc:.2f}"
            print(line + f"   (choix/AUC, chance 33%/0.50, n={len(pos)}+{len(neg)})")
            res_ep[f"H{H}"] = res_h

            # picks du scoreur "ratio" pour les visuels
            S = S_by["ratio"]
            picks = S.argmax(axis=1)

            # visuel H=16 : 2 positifs -- reel + 6 reves, elu encadre
            if H == 16:
                for k in range(min(2, len(pos))):
                    tt = pos[k]
                    mi = moments.index(tt)
                    steps = list(range(0, H, 2))
                    rows = [np.concatenate(
                        [np.transpose(frames[tt + 1 + s], (1, 2, 0)) for s in steps], axis=1)]
                    for pi in range(len(plan_names)):
                        d = dreams[mi * len(plan_names) + pi]
                        rows.append(np.concatenate(
                            [np.transpose(d[s], (1, 2, 0)) for s in steps], axis=1))
                    sheet = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
                    big = cv2.resize(sheet, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                    labels = ["REEL"] + [f"{n} S={S[mi, i]:+.3f}" for i, n in enumerate(plan_names)]
                    for ri, lab in enumerate(labels):
                        y0 = ri * 64 * 3
                        chosen = ri > 0 and (ri - 1) == picks[mi]
                        col = (0, 255, 0) if chosen else (255, 255, 0)
                        cv2.putText(big, lab, (6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, col, 1, cv2.LINE_AA)
                        if chosen:
                            cv2.rectangle(big, (0, y0), (big.shape[1] - 1, y0 + 191),
                                          (0, 255, 0), 2)
                    cv2.imwrite(str(OUT_DIR / f"choix_{stem[:16]}_t{tt}.png"),
                                cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

        results["episodes"][stem] = res_ep

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # figure : choix (haut) + AUC (bas), une colonne par episode, 4 scoreurs
    n_ep = len(results["episodes"])
    fig, axes = plt.subplots(2, n_ep, figsize=(6.2 * n_ep, 8.2), squeeze=False)
    for col, (stem, res_ep) in enumerate(results["episodes"].items()):
        hs = [int(k[1:]) for k in res_ep]
        for sc in SCORERS:
            axes[0][col].plot(hs, [res_ep[f"H{h}"]["scoreurs"][sc]["choix_taper_sur_positifs"] * 100
                                   for h in hs], "o-", lw=2, label=sc)
            axes[1][col].plot(hs, [res_ep[f"H{h}"]["scoreurs"][sc]["auc_affordance"]
                                   for h in hs], "o-", lw=2, label=sc)
        axes[0][col].axhline(100 * 2 / 6, color="gray", ls=":", label="chance")
        axes[1][col].axhline(0.5, color="gray", ls=":", label="hasard")
        axes[0][col].set_title(stem[:34])
        axes[0][col].set_ylabel("% choix contenant taper")
        axes[1][col].set_ylabel("AUC affordance")
        axes[0][col].set_ylim(0, 100)
        axes[1][col].set_ylim(0.4, 1.0)
        for r in (0, 1):
            axes[r][col].set_xlabel("portée du rêve H (pas)")
            axes[r][col].legend(fontsize=8)
            axes[r][col].grid(alpha=0.3)
    fig.suptitle("Monde 2, marche 2.5 - quatre fonctions de but comparées sur les mêmes rêves")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chooseur_vs_H.png", dpi=130)
    print(f"\n-> {OUT_DIR.relative_to(ROOT)}\\chooseur_vs_H.png + results.json")


if __name__ == "__main__":
    main()
