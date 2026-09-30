"""
v4 -- fine-tune de v3b, cible : rever PLUS LOIN.

Diagnostic (JOURNAL 2026-07-05) : la decision decroit au-dela de H=8 et la
portee casse a 56 pas -- or v3b n'a jamais ete entraine a rever plus de 8 pas.
Remede :
  1. curriculum pousse a K=16 (warm-start depuis v3b, LR encore reduit),
  2. EMA des poids (moyenne glissante, decay 0.999) -- stabilisateur quasi gratuit,
  3. selection honnete a profondeur 16 (val Delta-rollout@16, sur les poids EMA).

Sorties : outputs/<DC_TAG=v4-k16-ema>/ (model_best.pt = EMA, config.json)
Lance   : .venv\\Scripts\\python.exe scripts\\train_v4_finetune.py [val_stem] [exclu...]
"""
from __future__ import annotations

import importlib.util
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

# reutilise build_corpus / val_honest_rollout de train_v3
_spec = importlib.util.spec_from_file_location("tv3", Path(__file__).parent / "train_v3.py")
tv3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv3)

PROC_DIR = ROOT / "data" / "processed"
SRC_CKPT = Path(os.environ.get("DC_SRC",
                str(ROOT / "outputs" / "v3b-fullcorpus" / "model_best.pt")))

CTX, BASE = 4, 48
BATCH = 64
LAMBDA_CHANGE = 5.0
SEED = 0
VAL_H = 16
EMA_DECAY = 0.999
# (K, lr, n_epochs, max_steps)
STAGES = [(16, 3e-5, 4, 700, 48)]  # stage K=8 deja fait (model_stage1) ;
# batch 48 au lieu de 64 : les activations K=16 debordaient la VRAM (crawl WDDM)
DEFAULT_VAL = "Player129-f153ac423f61-20210617-173110"
DEFAULT_EXCL = ["treechop-984393664dfd-20210924-174326",
                "Player871-2e9a64a90d31-20210627-154641",
                "Player309-dcc21a4f8784-20210721-163058"]


class Ema:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = os.environ.get("DC_TAG", "v4-k16-ema")
    out_dir = ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_stem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAL
    excl = set(sys.argv[2:]) if len(sys.argv) > 2 else set(DEFAULT_EXCL)
    excl |= {val_stem}
    npz_files = sorted(PROC_DIR.glob("*.npz"))
    yt_dir = ROOT / "data" / "processed_youtube"
    if yt_dir.exists():
        npz_files += sorted(yt_dir.glob("*.npz"))
    train_eps = [p for p in npz_files if p.stem not in excl]
    print(f"device: {device} | train: {len(train_eps)} eps | val: {val_stem}")

    frames, actions, ok = tv3.build_corpus(train_eps)
    vf, va, vok = C.load_episode_u8(PROC_DIR / f"{val_stem}.npz")
    print(f"corpus: {frames.shape[0]} frames ({frames.nbytes/1e9:.2f} Go uint8)")

    model = ConvPredictor(base=BASE, in_frames=CTX).to(device)
    model.load_state_dict(torch.load(SRC_CKPT, weights_only=True))
    print(f"warm-start depuis {SRC_CKPT.relative_to(ROOT)}")
    (out_dir / "config.json").write_text(
        json.dumps({"base": BASE, "in_frames": CTX, "action_dim": 10,
                    "warm_start": str(SRC_CKPT)}), encoding="utf-8")

    ema = Ema(model, EMA_DECAY)
    eval_model = ConvPredictor(base=BASE, in_frames=CTX).to(device)

    # reference avant fine-tune (v3b tel quel, metrique @16)
    ref = tv3.val_honest_rollout(model, vf, va, vok, device, ctx=CTX, h_max=VAL_H)
    print(f"reference v3b : val dRoll@{VAL_H} = {ref:.5f}")

    run = None
    try:
        import wandb
        run = wandb.init(project="dreamcraft", name=run_name,
                         config={"stages": str(STAGES), "ema": EMA_DECAY,
                                 "warm_start": "v3b", "val_h": VAL_H,
                                 "n_train_eps": len(train_eps)},
                         settings=wandb.Settings(init_timeout=30))
        print("wandb: ok")
    except Exception as e:
        print(f"wandb indisponible ({type(e).__name__})")

    best = ref
    ckpt = out_dir / "model_best.pt"
    torch.save(model.state_dict(), ckpt)  # depart = v3b (on ne peut que faire mieux)
    opt = torch.optim.AdamW(model.parameters(), lr=STAGES[0][1])
    t0 = time.time()
    ep_num = 0

    for K, lr, n_ep, max_steps, batch in STAGES:
        for g in opt.param_groups:
            g["lr"] = lr
        print(f"[stage] K={K} lr={lr:.1e} x{n_ep} epochs (cap {max_steps} steps)")
        for _ in range(n_ep):
            ep_num += 1
            starts_all = C.valid_starts(ok, CTX, K)
            starts_all = rng.permutation(starts_all)[: max_steps * batch]
            model.train()
            ep_loss, n_steps = 0.0, 0
            for i in range(0, len(starts_all), batch):
                b = starts_all[i: i + batch]
                t = b + CTX - 1
                x = tv3.to_f32(C.stack_context(frames, b, CTX), device)
                loss = 0.0
                for h in range(1, K + 1):
                    act = torch.from_numpy(actions[t + h - 1]).to(device)
                    pred = model(x, act)
                    tgt = tv3.to_f32(frames[t + h], device)
                    real_prev = tv3.to_f32(frames[t + h - 1], device)
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
                ema.update(model)
                ep_loss += float(loss.detach())
                n_steps += 1

            ema.copy_to(eval_model)
            v_ema = tv3.val_honest_rollout(eval_model, vf, va, vok, device,
                                           ctx=CTX, h_max=VAL_H)
            marker = ""
            if v_ema < best:
                best = v_ema
                torch.save(eval_model.state_dict(), ckpt)
                marker = "  <- best (EMA), sauve"
            print(f"epoch {ep_num} K={K}  loss {ep_loss/max(n_steps,1):.5f}  "
                  f"val dRoll@{VAL_H} (EMA) {v_ema:.5f}{marker}", flush=True)
            if run:
                run.log({"epoch": ep_num, "K": K, "train_loss": ep_loss / max(n_steps, 1),
                         "val_delta_rollout16_ema": v_ema})

    ema.copy_to(eval_model)
    torch.save(eval_model.state_dict(), out_dir / "model_last.pt")
    print(f"fine-tune: {time.time()-t0:.0f}s | ref {ref:.5f} -> best {best:.5f} "
          f"({(ref-best)/ref*100:+.1f}% d'amelioration)")
    if run:
        run.summary["ref_v3b"] = ref
        run.summary["best"] = best
        run.finish()


if __name__ == "__main__":
    main()
