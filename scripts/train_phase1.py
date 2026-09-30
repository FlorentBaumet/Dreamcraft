"""
Phase 1 v2 -- entrainement MULTI-PAS : le modele s'entraine sur ses propres reves.

Le probleme mesure en v1 : entraine uniquement a 1 pas depuis des vraies frames,
le modele n'a jamais vu ses propres sorties en entree -> en rollout, son flou
se compose et le reve fond (portee ~ 0).

Le remede (standard en video prediction) : derouler K pas PENDANT l'entrainement
en reinjectant les predictions (gradient a travers tout le rollout), avec K qui
grandit au fil des epochs (curriculum 1 -> 2 -> 4 -> 8). Le modele apprend a
corriger sa propre derive.

Sorties : outputs/phase1_multistep/ (checkpoint, log) ; courbe via rollout_phase1.py
Lance   : .venv\\Scripts\\python.exe scripts\\train_phase1.py [val_stem] [exclu...]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import vpt  # noqa: E402
from dreamcraft.eval import metrics  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"

EPOCHS = 16
# curriculum : epoch -> horizon K d'entrainement
K_SCHEDULE = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 4, 8: 4, 9: 4, 10: 4,
              11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 8}
BATCH = 64
LR = 3e-4
LAMBDA_CHANGE = 5.0
SEED = 0
VAL_ROLL_H = 8       # horizon du rollout de validation (selection du best)
DEFAULT_VAL = "Player129-f153ac423f61-20210617-173110"


def load_ep(npz_path):
    frames_u8, actions, gui = vpt.load_episode(npz_path)
    frames = np.transpose(frames_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))
    n = frames.shape[0]
    ok = np.ones(n - 1, dtype=bool)
    ok[gui == 1] = False
    ok[0] = False
    return frames, actions, ok


def windows(ok: np.ndarray, k: int) -> np.ndarray:
    """t0 tels que les k transitions t0..t0+k-1 sont toutes valides."""
    conv = np.convolve(ok.astype(int), np.ones(k, dtype=int), mode="valid")
    return np.flatnonzero(conv == k)


@torch.no_grad()
def val_rollout_mse(model, frames, actions, ok, device, h=VAL_ROLL_H, n_starts=64):
    """MSE globale moyenne au pas h d'un reve en boucle (proxy de la portee)."""
    model.eval()
    cand = windows(ok, h)
    starts = cand[np.linspace(0, len(cand) - 1, min(n_starts, len(cand))).astype(int)]
    cur = torch.from_numpy(frames[starts]).to(device)
    for i in range(h):
        act = torch.from_numpy(actions[starts + i]).to(device)
        cur = model(cur, act)
    real = torch.from_numpy(frames[starts + h]).to(device)
    return float(((cur - real) ** 2).mean())


@torch.no_grad()
def val_1step_dmse(model, frames, actions, ok, device, stride=10):
    model.eval()
    idx = np.flatnonzero(ok)[::stride]
    total, count = 0.0, 0
    for i in range(0, len(idx), 256):
        b = idx[i: i + 256]
        cur = torch.from_numpy(frames[b]).to(device)
        act = torch.from_numpy(actions[b]).to(device)
        tgt = torch.from_numpy(frames[b + 1]).to(device)
        pred = model(cur, act)
        mask = ((tgt - cur).abs().amax(dim=1, keepdim=True) > metrics.CHANGE_THRESHOLD)
        for j in range(len(b)):
            m = mask[j, 0]
            if int(m.sum()) < metrics.MIN_CHANGED_PIXELS:
                continue
            total += float(((pred[j] - tgt[j]) ** 2)[:, m].mean())
            count += 1
    return total / max(count, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = os.environ.get("DC_TAG", "phase1-multistep")
    out_dir = ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_stem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAL
    excl = set(sys.argv[2:]) | {val_stem}
    npz_files = sorted(PROC_DIR.glob("*.npz"))
    train_eps = [p for p in npz_files if p.stem not in excl]
    print(f"device: {device} | train: {len(train_eps)} eps | val: {val_stem}")
    print(f"exclus du train: {sorted(excl)}")

    tr = [load_ep(p) for p in train_eps]
    vf, va, vok = load_ep(PROC_DIR / f"{val_stem}.npz")
    n_pairs = sum(int(ok.sum()) for _, _, ok in tr)
    print(f"{n_pairs} transitions d'entrainement")

    model = ConvPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    run = None
    try:
        import wandb
        run = wandb.init(project="dreamcraft", name=run_name,
                         config={"epochs": EPOCHS, "k_schedule": str(K_SCHEDULE),
                                 "batch": BATCH, "lr": LR, "params": n_params,
                                 "lambda_change": LAMBDA_CHANGE,
                                 "n_train_eps": len(train_eps), "val_ep": val_stem,
                                 "val_roll_h": VAL_ROLL_H},
                         settings=wandb.Settings(init_timeout=30))
        print("wandb: ok")
    except Exception as e:
        print(f"wandb indisponible ({type(e).__name__}) -> on continue sans")

    best = float("inf")
    ckpt = out_dir / "model_best.pt"
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        K = K_SCHEDULE[epoch]
        model.train()
        ep_loss, n_steps = 0.0, 0
        for frames, actions, ok in tr:
            w = windows(ok, K)
            if len(w) == 0:
                continue
            w = rng.permutation(w)
            for i in range(0, len(w), BATCH):
                b = w[i: i + BATCH]
                cur = torch.from_numpy(frames[b]).to(device)
                loss = 0.0
                for h in range(K):
                    act = torch.from_numpy(actions[b + h]).to(device)
                    pred = model(cur, act)
                    tgt = torch.from_numpy(frames[b + h + 1]).to(device)
                    real_prev = torch.from_numpy(frames[b + h]).to(device)
                    err = (pred - tgt) ** 2
                    mask = ((tgt - real_prev).abs().amax(dim=1, keepdim=True)
                            > metrics.CHANGE_THRESHOLD).float()
                    loss = loss + err.mean() \
                        + LAMBDA_CHANGE * (err * mask).sum() / (mask.sum() * 3 + 1e-8)
                    cur = pred          # <- le reve devient l'entree (gradient inclus)
                loss = loss / K
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach())
                n_steps += 1

        v_roll = val_rollout_mse(model, vf, va, vok, device)
        v_1 = val_1step_dmse(model, vf, va, vok, device)
        marker = ""
        if v_roll < best:
            best = v_roll
            torch.save(model.state_dict(), ckpt)
            marker = "  <- best, sauve"
        print(f"epoch {epoch:2d}/{EPOCHS} K={K}  loss {ep_loss/max(n_steps,1):.5f}  "
              f"val roll@{VAL_ROLL_H} {v_roll:.5f}  val 1-pas dMSE {v_1:.5f}{marker}")
        if run:
            run.log({"epoch": epoch, "K": K, "train_loss": ep_loss / max(n_steps, 1),
                     f"val_rollout_mse_h{VAL_ROLL_H}": v_roll, "val_delta_mse": v_1})

    print(f"entrainement: {time.time()-t0:.0f}s | best val roll@{VAL_ROLL_H}: {best:.5f}")
    print(f"checkpoint: {ckpt.relative_to(ROOT)}")
    if run:
        run.summary["best_val_rollout_mse"] = best
        run.finish()


if __name__ == "__main__":
    main()
