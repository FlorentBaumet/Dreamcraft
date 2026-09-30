"""
Entraine l'IDM sur nos 234 episodes VPT etiquetes (frames + touches connues).

Fenetre : frames t-2..t+2 -> action de la transition t -> t+1 (+ flag GUI).
Val (qualite du pseudo-etiquetage) : les 4 episodes d'eval habituels, jamais vus.

Sorties : outputs/idm/ (idm_best.pt, metrics par bouton)
Lance   : .venv\\Scripts\\python.exe scripts\\train_idm.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import vpt  # noqa: E402
from dreamcraft.models.idm import IDM  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "idm"

WIN = 5          # frames t-2..t+2
EPOCHS = 6
BATCH = 128
LR = 3e-4
STEPS_PER_EPOCH = 1500
CAM_W = 5.0      # poids de la loss camera
SEED = 0
HOLDOUT = {"Player129-f153ac423f61-20210617-173110",
           "treechop-984393664dfd-20210924-174326",
           "Player871-2e9a64a90d31-20210627-154641",
           "Player309-dcc21a4f8784-20210721-163058"}
BTN_NAMES = vpt.ACTION_NAMES[:8]


def load_ep(p):
    """frames uint8 (N,3,64,64), actions (N-1,10), gui (N-1,)"""
    frames_u8, actions, gui = vpt.load_episode(p)
    frames = np.ascontiguousarray(np.transpose(frames_u8, (0, 3, 1, 2)))
    return frames, actions.astype(np.float32), gui.astype(np.float32)


def build(paths):
    sizes = [int(np.load(p)["gui"].shape[0]) + 1 for p in paths]
    total = sum(sizes)
    frames = np.empty((total, 3, 64, 64), dtype=np.uint8)
    offs = list(np.cumsum([0] + sizes[:-1]))
    fs, acts, guis = [], [], []
    for p, s, n in zip(paths, offs, sizes):
        f, a, g = load_ep(p)
        frames[s: s + n] = f
        fs.append(None); acts.append(a); guis.append(g)
        del f
    actions = np.zeros((total - 1, 10), dtype=np.float32)
    gui = np.zeros(total - 1, dtype=np.float32)
    valid = np.zeros(total - 1, dtype=bool)
    for a, g, s, n in zip(acts, guis, offs, sizes):
        actions[s:s + n - 1] = a
        gui[s:s + n - 1] = g
        valid[s:s + n - 1] = True
        valid[s + n - 2] = False  # frontiere
    # centres t valides : t-2..t+2 dans le meme episode
    ok_center = valid.copy()
    for shift in (1, 2):
        ok_center[shift:] &= valid[:-shift]
        ok_center[:-shift] &= valid[shift:]
    ok_center[:2] = False
    ok_center[-2:] = False
    return frames, actions, gui, np.flatnonzero(ok_center)


def windows_batch(frames, centers):
    """(B,) centres -> (B, 15, 64, 64) uint8 (t-2..t+2)."""
    cols = [frames[centers + d] for d in (-2, -1, 0, 1, 2)]
    return np.concatenate(cols, axis=1)


@torch.no_grad()
def evaluate(model, frames, actions, gui, centers, device, n=4000):
    idx = centers[np.linspace(0, len(centers) - 1, min(n, len(centers))).astype(int)]
    P = {k: [0, 0, 0] for k in BTN_NAMES}   # tp, fp, fn
    cam_err, gui_ok, count = 0.0, 0, 0
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        x = torch.from_numpy(windows_batch(frames, b)).to(device).float() / 255.0
        lb, cam, lg = model(x)
        pb = (torch.sigmoid(lb) > 0.5).float().cpu().numpy()
        tb = actions[b][:, :8]
        for j, k in enumerate(BTN_NAMES):
            P[k][0] += int(((pb[:, j] == 1) & (tb[:, j] == 1)).sum())
            P[k][1] += int(((pb[:, j] == 1) & (tb[:, j] == 0)).sum())
            P[k][2] += int(((pb[:, j] == 0) & (tb[:, j] == 1)).sum())
        cam_err += float(np.abs(cam.cpu().numpy() - actions[b][:, 8:10]).sum())
        gui_ok += int(((torch.sigmoid(lg[:, 0]) > 0.5).float().cpu().numpy()
                       == gui[b]).sum())
        count += len(b)
    f1 = {}
    for k, (tp, fp, fn) in P.items():
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1[k] = round(2 * prec * rec / max(prec + rec, 1e-9), 3)
    return f1, cam_err / (count * 2), gui_ok / count


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npz = sorted(PROC_DIR.glob("*.npz"))
    train_p = [p for p in npz if p.stem not in HOLDOUT]
    val_p = [p for p in npz if p.stem in HOLDOUT]
    print(f"IDM train: {len(train_p)} eps | val: {len(val_p)} eps | device {device}")

    frames, actions, gui, centers = build(train_p)
    vframes, vactions, vgui, vcenters = build(val_p)
    print(f"{len(centers)} fenetres train, {len(vcenters)} val, "
          f"{frames.nbytes/1e9:.1f} Go RAM")

    model = IDM().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # boutons rares -> pos_weight equilibre
    pos = actions[centers][:, :8].mean(axis=0)
    pw = torch.tensor(((1 - pos) / np.maximum(pos, 1e-3)).clip(1, 30),
                      dtype=torch.float32, device=device)
    print("pos_weight:", [f"{k}:{v:.1f}" for k, v in zip(BTN_NAMES, pw.tolist())])

    run = None
    try:
        import wandb
        run = wandb.init(project="dreamcraft", name="idm",
                         config={"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                                 "win": WIN, "n_train_eps": len(train_p)},
                         settings=wandb.Settings(init_timeout=30))
    except Exception as e:
        print(f"wandb indisponible ({type(e).__name__})")

    best = -1.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sel = rng.permutation(centers)[: STEPS_PER_EPOCH * BATCH]
        tot, ns = 0.0, 0
        for i in range(0, len(sel), BATCH):
            b = sel[i:i + BATCH]
            x = torch.from_numpy(windows_batch(frames, b)).to(device).float() / 255.0
            ta = torch.from_numpy(actions[b]).to(device)
            tg = torch.from_numpy(gui[b]).to(device)
            lb, cam, lg = model(x)
            loss = (F.binary_cross_entropy_with_logits(lb, ta[:, :8], pos_weight=pw)
                    + CAM_W * F.smooth_l1_loss(cam, ta[:, 8:10])
                    + F.binary_cross_entropy_with_logits(lg[:, 0], tg))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); ns += 1

        f1, cam_mae, gui_acc = evaluate(model, vframes, vactions, vgui, vcenters, device)
        score = float(np.mean([f1["avancer"], f1["taper"]])) - cam_mae
        marker = ""
        if score > best:
            best = score
            torch.save(model.state_dict(), OUT_DIR / "idm_best.pt")
            (OUT_DIR / "val_metrics.json").write_text(
                json.dumps({"f1": f1, "cam_mae": cam_mae, "gui_acc": gui_acc},
                           indent=2), encoding="utf-8")
            marker = "  <- best, sauve"
        print(f"epoch {epoch}/{EPOCHS}  loss {tot/max(ns,1):.4f}  "
              f"F1 avancer {f1['avancer']:.2f} taper {f1['taper']:.2f} "
              f"| cam MAE {cam_mae:.3f} (x60px={cam_mae*60:.1f}) | gui {gui_acc:.2%}{marker}",
              flush=True)
        if run:
            run.log({"epoch": epoch, "loss": tot / max(ns, 1), "cam_mae": cam_mae,
                     "gui_acc": gui_acc, **{f"f1_{k}": v for k, v in f1.items()}})

    print(f"IDM: {time.time()-t0:.0f}s | F1/MAE detail: outputs/idm/val_metrics.json")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
