"""
v3 -- contexte multi-frames + curriculum multi-pas + selection HONNETE.

Ce qui change vs v2 (lecons de l'apres-midi) :
  1. CONTEXTE : le modele voit 4 frames (il percoit enfin le mouvement),
  2. SELECTION : le meilleur checkpoint = meilleure MSE Delta-region moyenne
     sur un reve de 8 pas (la metrique honnete ; la globale selectionne l'inertie),
  3. LR : divise par ~1.7 a chaque montee du curriculum K (les epochs K>1
     oscillaient en v2),
  4. base 48 (~3.7M params) au lieu de 32 (~1.6M).

Sorties : outputs/<DC_TAG=v3-ctx4>/ (model_best.pt, config.json)
Lance   : .venv\\Scripts\\python.exe scripts\\train_v3.py [val_stem] [exclu...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.eval import metrics  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"

CTX = 4
BASE = int(os.environ.get("DC_BASE", "48"))
EPOCHS = 16
K_SCHEDULE = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 4, 8: 4, 9: 4, 10: 4,
              11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 8}
BATCH = 64
LR0 = 3e-4
LR_DECAY_ON_K = 0.6
LAMBDA_CHANGE = 5.0
MAX_STEPS_PER_EPOCH = 1200
VAL_ROLL_H = 8
SEED = 0
GRAD_CKPT = os.environ.get("DC_GRAD_CKPT", "0") == "1"
DEFAULT_VAL = "Player129-f153ac423f61-20210617-173110"


def build_corpus(paths, res=64):
    """Concatene tous les episodes en un seul tableau (frontieres invalidees).

    Pre-alloue le gros tableau (pic RAM ~1x le corpus au lieu de 2x avec
    np.concatenate -- lecon du crash OOM du 2026-07-07). res : 64 ou 96."""
    # passe 1 : tailles seulement (via 'gui', minuscule)
    sizes = [int(np.load(p)["gui"].shape[0]) + 1 for p in paths]
    total = sum(sizes)
    frames = np.empty((total, 3, res, res), dtype=np.uint8)
    actions = np.zeros((total - 1, 10), dtype=np.float32)
    ok = np.zeros(total - 1, dtype=bool)
    # passe 2 : remplissage episode par episode
    off = 0
    for p, n in zip(paths, sizes):
        f, a, o = C.load_episode_u8(p)
        frames[off: off + n] = f
        actions[off: off + n - 1] = a
        ok[off: off + n - 1] = o
        # la transition frontiere (dernier frame -> premier du suivant)
        # tombe a l'index off+n-1 et reste False
        off += n
    return frames, actions, ok


def to_f32(u8_batch, device):
    return torch.from_numpy(u8_batch).to(device).float() / 255.0


@torch.no_grad()
def val_honest_rollout(model, vf, va, vok, device, ctx=CTX, h_max=VAL_ROLL_H, n_starts=200):
    """MSE Delta-region moyenne sur les pas 1..h_max d'un reve (masques reels).
    C'est LA metrique de selection (l'inertie y perd)."""
    model.eval()
    cand = C.valid_starts(vok, ctx, h_max)
    starts = cand[np.linspace(0, len(cand) - 1, min(n_starts, len(cand))).astype(int)]
    t = starts + ctx - 1
    x = to_f32(C.stack_context(vf, starts, ctx), device)
    vals = []
    for h in range(1, h_max + 1):
        act = torch.from_numpy(va[t + h - 1]).to(device)
        pred = model(x, act)
        real = to_f32(vf[t + h], device)
        real_prev = to_f32(vf[t + h - 1], device)
        mask = ((real - real_prev).abs().amax(dim=1) > metrics.CHANGE_THRESHOLD)
        err = ((pred - real) ** 2).mean(dim=1)          # (S,64,64) moyenne canaux
        msum = mask.sum(dim=(1, 2))
        esum = (err * mask).sum(dim=(1, 2))
        valid = msum >= metrics.MIN_CHANGED_PIXELS
        if int(valid.sum()) > 0:
            vals.append(float((esum[valid] / msum[valid]).mean()))
        x = C.roll_context(x, pred)
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = os.environ.get("DC_TAG", "v3-ctx4")
    out_dir = ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_stem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAL
    excl = set(sys.argv[2:]) | {val_stem}
    npz_files = sorted(PROC_DIR.glob("*.npz"))
    train_eps = [p for p in npz_files if p.stem not in excl]
    print(f"device: {device} | train: {len(train_eps)} eps | val: {val_stem}")
    print(f"exclus: {sorted(excl)}")

    frames, actions, ok = build_corpus(train_eps)
    vf, va, vok = C.load_episode_u8(PROC_DIR / f"{val_stem}.npz")
    print(f"corpus: {frames.shape[0]} frames, {int(ok.sum())} transitions valides, "
          f"{frames.nbytes / 1e9:.2f} Go RAM (uint8)")

    model = ConvPredictor(base=BASE, in_frames=CTX).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametres: {n_params/1e6:.2f} M (base {BASE}, contexte {CTX})")
    (out_dir / "config.json").write_text(
        json.dumps({"base": BASE, "in_frames": CTX, "action_dim": 10}), encoding="utf-8")

    lr = LR0
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    run = None
    try:
        import wandb
        run = wandb.init(project="dreamcraft", name=run_name,
                         config={"epochs": EPOCHS, "k_schedule": str(K_SCHEDULE),
                                 "batch": BATCH, "lr0": LR0, "lr_decay_on_k": LR_DECAY_ON_K,
                                 "base": BASE, "context": CTX, "params": n_params,
                                 "lambda_change": LAMBDA_CHANGE,
                                 "n_train_eps": len(train_eps), "val_ep": val_stem,
                                 "selection": "delta_region_rollout@8"},
                         settings=wandb.Settings(init_timeout=30))
        print("wandb: ok")
    except Exception as e:
        print(f"wandb indisponible ({type(e).__name__}) -> on continue sans")

    best = float("inf")
    ckpt = out_dir / "model_best.pt"
    prev_k = 1
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        K = K_SCHEDULE[epoch]
        if K > prev_k:
            lr *= LR_DECAY_ON_K
            for g in opt.param_groups:
                g["lr"] = lr
            print(f"  [K {prev_k} -> {K}] lr -> {lr:.2e}")
            prev_k = K

        starts_all = C.valid_starts(ok, CTX, K)
        starts_all = rng.permutation(starts_all)[: MAX_STEPS_PER_EPOCH * BATCH]
        model.train()
        ep_loss, n_steps = 0.0, 0
        for i in range(0, len(starts_all), BATCH):
            b = starts_all[i: i + BATCH]
            t = b + CTX - 1
            x = to_f32(C.stack_context(frames, b, CTX), device)
            loss = 0.0
            for h in range(1, K + 1):
                act = torch.from_numpy(actions[t + h - 1]).to(device)
                if GRAD_CKPT:
                    pred = torch.utils.checkpoint.checkpoint(
                        model, x, act, use_reentrant=False)
                else:
                    pred = model(x, act)
                tgt = to_f32(frames[t + h], device)
                real_prev = to_f32(frames[t + h - 1], device)
                err = (pred - tgt) ** 2
                mask = ((tgt - real_prev).abs().amax(dim=1, keepdim=True)
                        > metrics.CHANGE_THRESHOLD).float()
                loss = loss + err.mean() \
                    + LAMBDA_CHANGE * (err * mask).sum() / (mask.sum() * 3 + 1e-8)
                x = C.roll_context(x, pred)
            loss = loss / K
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            n_steps += 1

        v_honest = val_honest_rollout(model, vf, va, vok, device)
        marker = ""
        if v_honest < best:
            best = v_honest
            torch.save(model.state_dict(), ckpt)
            marker = "  <- best, sauve"
        print(f"epoch {epoch:2d}/{EPOCHS} K={K} lr={lr:.1e}  loss {ep_loss/max(n_steps,1):.5f}  "
              f"val dRoll@{VAL_ROLL_H} {v_honest:.5f}{marker}", flush=True)
        if run:
            run.log({"epoch": epoch, "K": K, "lr": lr,
                     "train_loss": ep_loss / max(n_steps, 1),
                     "val_delta_rollout": v_honest})

    print(f"entrainement: {time.time()-t0:.0f}s | best val dRoll@{VAL_ROLL_H}: {best:.5f}")
    if run:
        run.summary["best_val_delta_rollout"] = best
        run.finish()


if __name__ == "__main__":
    main()
