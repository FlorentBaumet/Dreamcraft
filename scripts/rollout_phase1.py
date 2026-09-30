"""
Le modele REVE en boucle, et on mesure jusqu'ou il tient.

Compatible mono-frame (Phase 0) et contexte multi-frames (v3+) : lit le
config.json a cote du checkpoint ({"base", "in_frames"}) s'il existe.
L'eligibilite des departs utilise un pad CONSTANT (EVAL_CONTEXT_PAD) pour que
tous les modeles soient mesures sur LES MEMES fenetres.

Metriques par pas h :
  - MSE globale vs realite (reference B0-gelee)  [recompense l'inertie -> a lire avec l'autre]
  - MSE Delta-region : uniquement la ou la realite bouge a ce pas  [l'honnete]
  - portee honnete = dernier pas ou le reve bat la frame gelee en Delta-region.

Sorties : outputs/<tag>/  (courbe_H.png 2 panneaux, reve_*.gif, strip_*.png, results.json)
Lance   : .venv\\Scripts\\python.exe scripts\\rollout_phase1.py [val_stem] [ckpt] [tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
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
DEFAULT_OUT = ROOT / "outputs" / "phase1"
DEFAULT_CKPT = ROOT / "outputs" / "phase0" / "model_best.pt"

import os

VAL_STEM = "Player129-f153ac423f61-20210617-173110"
H = int(os.environ.get("DC_H", "32"))
N_STARTS = 48
N_GIFS = 3
FPS_GIF = 8
UP = 4


def to_img(x: np.ndarray) -> np.ndarray:
    return (np.transpose(np.clip(x, 0, 1), (1, 2, 0)) * 255).astype(np.uint8)


def load_model(ckpt: Path, device: str) -> tuple[ConvPredictor, int]:
    cfg = {"base": 32, "in_frames": 1}
    cfg_file = ckpt.parent / "config.json"
    if cfg_file.exists():
        cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
    model = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model, cfg["in_frames"]


def main():
    val_stem = sys.argv[1] if len(sys.argv) > 1 else VAL_STEM
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CKPT
    out_dir = (ROOT / "outputs" / sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ep_path = PROC_DIR / f"{val_stem}.npz"
    alt = ROOT / "data" / os.environ.get("DC_PROC", "processed") / f"{val_stem}.npz"
    if alt.exists():
        ep_path = alt
    frames_u8, actions, ok = C.load_episode_u8(ep_path)
    frames = frames_u8.astype(np.float32) / 255.0
    res = frames_u8.shape[-1]
    model, ctx = load_model(ckpt, device)
    pad = C.EVAL_CONTEXT_PAD
    assert ctx <= pad, f"contexte {ctx} > pad d'eligibilite {pad}"
    print(f"episode: {val_stem} | ckpt: {ckpt} (contexte {ctx}) | H={H}")

    cand = C.valid_starts(ok, pad, H)
    print(f"{len(cand)} fenetres valides (pad {pad} + {H} pas)")
    starts = cand[np.linspace(0, len(cand) - 1, min(N_STARTS, len(cand))).astype(int)]
    t = starts + pad - 1                     # le "present" de chaque fenetre
    ctx_starts = t - ctx + 1                 # debut du contexte du modele

    err_model = np.zeros((len(starts), H))
    err_frozen = np.zeros((len(starts), H))
    derr_model = np.full((len(starts), H), np.nan)
    derr_frozen = np.full((len(starts), H), np.nan)
    dreams_np = np.zeros((len(starts), H, 3, res, res), dtype=np.float32)

    with torch.no_grad():
        x = torch.from_numpy(C.stack_context(frames_u8, ctx_starts, ctx)).to(device).float() / 255.0
        frozen = frames[t]
        for h in range(H):
            act = torch.from_numpy(actions[t + h]).to(device)
            pred = model(x, act)
            dreams_np[:, h] = pred.cpu().numpy()
            real = frames[t + h + 1]
            real_prev = frames[t + h]
            err_model[:, h] = ((dreams_np[:, h] - real) ** 2).mean(axis=(1, 2, 3))
            err_frozen[:, h] = ((frozen - real) ** 2).mean(axis=(1, 2, 3))
            chg = (np.abs(real - real_prev).max(axis=1) > M.CHANGE_THRESHOLD)
            for s in range(len(starts)):
                m = chg[s]
                if int(m.sum()) < M.MIN_CHANGED_PIXELS:
                    continue
                derr_model[s, h] = ((dreams_np[s, h] - real[s]) ** 2)[:, m].mean()
                derr_frozen[s, h] = ((frozen[s] - real[s]) ** 2)[:, m].mean()
            x = C.roll_context(x, pred)

    m_mean, m_std = err_model.mean(0), err_model.std(0)
    f_mean = err_frozen.mean(0)
    dm_mean = np.nanmean(derr_model, axis=0)
    df_mean = np.nanmean(derr_frozen, axis=0)
    cross = next((h + 1 for h in range(H) if m_mean[h] > f_mean[h]), None)
    portee = cross - 1 if cross else H
    dcross = next((h + 1 for h in range(H) if dm_mean[h] > df_mean[h]), None)
    dportee = dcross - 1 if dcross else H
    print(f"portee (MSE globale)  : {portee} pas (~{portee/20:.2f} s)")
    print(f"portee (Delta-region) : {dportee} pas (~{dportee/20:.2f} s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    hs = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(hs, m_mean, label="modèle (rêve en boucle)", lw=2)
    ax.fill_between(hs, m_mean - m_std, m_mean + m_std, alpha=0.2)
    ax.plot(hs, f_mean, label="B0-gelée", lw=2, ls="--")
    ax.set_title("MSE globale (récompense l'inertie)")
    ax = axes[1]
    ax.plot(hs, dm_mean, label="modèle (rêve en boucle)", lw=2)
    ax.plot(hs, df_mean, label="B0-gelée", lw=2, ls="--")
    if dcross:
        ax.axvline(dcross, color="gray", ls=":", lw=1.5)
    ax.set_title(f"Δ-region, là où la réalité bouge (portée {dportee} pas)")
    for ax in axes:
        ax.set_xlabel("horizon de rêve h (pas de 50 ms)")
        ax.set_ylabel("MSE vs réalité")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Portée du rêve - {len(starts)} départs, épisode jamais vu")
    fig.tight_layout()
    fig.savefig(out_dir / "courbe_H.png", dpi=130)
    print(f"courbe -> {out_dir.relative_to(ROOT)}\\courbe_H.png")

    motion = np.array([
        np.abs(np.diff(frames[tt: tt + H + 1], axis=0)).mean() for tt in t
    ])
    top = np.argsort(-motion)[:N_GIFS]
    for rank, si in enumerate(top):
        tt = int(t[si])
        gif_frames = []
        for h in range(H):
            real = to_img(frames[tt + h + 1])
            dream = to_img(dreams_np[si, h])
            canvas = np.concatenate(
                [real, np.full((res, 2, 3), 255, np.uint8), dream], axis=1)
            big = cv2.resize(canvas, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, f"h={h+1}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(big, "REEL", (6, big.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(big, "REVE", (res * UP + 2 * UP + 6, big.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            gif_frames.append(big)
        imageio.mimsave(out_dir / f"reve_{rank+1}_t{tt}.gif", gif_frames,
                        fps=FPS_GIF, loop=0)

        steps = [1, 2, 4, 8, 16, 32]
        row_r = np.concatenate([to_img(frames[tt + s]) for s in steps], axis=1)
        row_d = np.concatenate([to_img(dreams_np[si, s - 1]) for s in steps], axis=1)
        strip = np.concatenate([row_r, row_d], axis=0)
        big = cv2.resize(strip, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        for i, s in enumerate(steps):
            cv2.putText(big, f"h={s}", (i * res * UP + 6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"strip_{rank+1}_t{tt}.png"),
                    cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print(f"gifs + strips -> {out_dir.relative_to(ROOT)}")

    (out_dir / "results.json").write_text(json.dumps({
        "val_ep": val_stem, "H": H, "n_starts": int(len(starts)),
        "ckpt": str(ckpt), "contexte": ctx,
        "portee_effective_pas": int(portee),
        "croisement_h": cross,
        "portee_delta_pas": int(dportee),
        "croisement_delta_h": dcross,
        "mse_modele_par_h": m_mean.tolist(),
        "mse_b0_gelee_par_h": f_mean.tolist(),
        "dmse_modele_par_h": dm_mean.tolist(),
        "dmse_b0_gelee_par_h": df_mean.tolist(),
    }, indent=2), encoding="utf-8")
    print("results.json ecrit")


if __name__ == "__main__":
    main()
