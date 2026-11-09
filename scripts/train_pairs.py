"""
Reentrainement sur les paires collectees en jeu.

Un seul levier : les donnees. Meme modele, meme loss qu'avant, on change ce qu'il voit.
Une paire partage la meme scene de depart : le modele voit cette scene evoluer avec et
sans le bouton, avec des touches exactes. Il n'y a pas de loss speciale pour les paires :
il suffit d'en mettre beaucoup dans chaque batch.
  - 40 % des departs : un coup commence pendant le reve,
  - 30 % : le temoin, scene immobile sans le bouton,
  - 30 % : n'importe ou (pivots, marche).

Le monde dc_test n'est jamais lu ici. Deux sessions de dc_train sont gardees de cote pour
suivre l'apprentissage, toujours les memes. Le programme d'entrainement est fixe : pas de
selection de checkpoint, donc pas de biais de selection.

Pour comparer des quantites ou des compositions de donnees, a programme egal :
  DC_ROOMS=naturel   seulement le terrain naturel (defaut : tout)
  DC_N=12            seulement les 12 premieres sessions retenues (defaut : toutes)

Sorties : outputs/<DC_TAG>/model_best.pt (dernier etat EMA), lisible par eval_pairs.py
Lance   : .venv\\Scripts\\python.exe scripts\\train_pairs.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.eval import metrics  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("eval_pairs")
tv3 = _load("train_v3")

TAG = os.environ.get("DC_TAG", "v15-paires")
SRC = Path(os.environ.get("DC_SRC", str(ROOT / "outputs" / "v13b-indomain" / "model_best.pt")))
STAGES = json.loads(os.environ.get("DC_STAGES",
                                   "[[4, 3e-5, 1500, 32], [8, 3e-5, 1500, 24], [12, 2e-5, 1000, 16]]"))
ROOMS = os.environ.get("DC_ROOMS", "tout")
N_MAX = int(os.environ.get("DC_N", "0"))
SHARE_A, SHARE_B = 0.4, 0.3
CTX, LAMBDA_CHANGE, EMA_DECAY, SEED = 4, 5.0, 0.999, 0


def room_of(path):
    with np.load(path) as z:
        return str(z["room"]) if "room" in z.files else "naturel"


def val_report(model, val, thr, device):
    """Suivi seulement : AUC du score jumeau sur les sessions gardees de cote."""
    scores = np.concatenate([E.U.twin_scores(model, CTX, s["fr"], s["acts"], s["starts"], device)
                             for s in val])
    label = np.concatenate([s["eff_a"] for s in val]) > thr
    return float(E.U.auc(scores, label)), len(label)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "outputs" / TAG
    out.mkdir(parents=True, exist_ok=True)

    files = sorted((ROOT / "data" / "processed_agent").glob("dc_train*.npz"))
    ref_files = files[:E.REF_TRAIN]
    val_files = ref_files[3::4]
    train_files = [p for p in files if p not in val_files
                   and (ROOMS == "tout" or room_of(p) == "naturel")]
    if N_MAX:
        train_files = train_files[:N_MAX]
    rooms = dict(Counter(room_of(p) for p in train_files))
    print(f"train : {len(train_files)} sessions {rooms} | suivi : {[p.stem[-6:] for p in val_files]}")

    # seuil appris sur les memes sessions que dans eval_pairs.py
    sessions = [E.load_session(p) for p in ref_files]
    thr = E.otsu(np.concatenate([s["eff_a"] for s in sessions]))
    val = [s for p, s in zip(ref_files, sessions) if p in val_files]
    for s in val:
        keep = s["eff_b"] <= thr
        s["starts"] = [t for t, k in zip(s["starts"], keep) if k]
        s["eff_a"], s["eff_b"] = s["eff_a"][keep], s["eff_b"][keep]

    frames, actions, ok = tv3.build_corpus(train_files, res=64)
    phase = np.concatenate([np.append(np.load(p)["phase"], "") for p in train_files])
    assert len(phase) == len(frames)
    t_a = np.flatnonzero((phase[1:] == "A_taper") & (phase[:-1] == "B_temoin")) + 1
    t_b = np.flatnonzero(phase == "B_temoin")
    print(f"{len(frames)} images | {len(t_a)} coups | {len(t_b)} images de temoin")

    model = ConvPredictor(base=48, in_frames=CTX).to(device)
    model.load_state_dict(torch.load(SRC, weights_only=True))
    ema_model = ConvPredictor(base=48, in_frames=CTX).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    (out / "config.json").write_text(json.dumps(
        {"base": 48, "in_frames": CTX, "action_dim": 10, "warm_start": str(SRC.parent.name),
         "stages": STAGES, "share_a": SHARE_A, "share_b": SHARE_B,
         "sessions": [p.stem for p in train_files]}), encoding="utf-8")

    auc0, n = val_report(ema_model, val, thr, device)
    print(f"depart : AUC jumeau sur le suivi {auc0:.1%} ({n} paires)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=STAGES[0][1])
    t0 = time.time()
    history = []
    for stage, (K, lr, steps, batch) in enumerate(STAGES):
        for g in opt.param_groups:
            g["lr"] = lr
        vs = C.valid_starts(ok, CTX, K)
        valid = np.zeros(len(frames), dtype=bool)
        valid[vs] = True
        n_a, n_b = int(batch * SHARE_A), int(batch * SHARE_B)
        total = 0.0
        for step in range(1, steps + 1):
            # le coup tombe a l'interieur des K pas reves
            s_a = rng.choice(t_a, n_a * 3) - rng.integers(0, max(K - 1, 1), n_a * 3) - CTX + 1
            s_b = rng.choice(t_b, n_b * 3) - CTX + 1
            s_a = s_a[(s_a >= 0) & valid[np.clip(s_a, 0, len(valid) - 1)]][:n_a]
            s_b = s_b[(s_b >= 0) & valid[np.clip(s_b, 0, len(valid) - 1)]][:n_b]
            s_r = rng.choice(vs, batch - len(s_a) - len(s_b))
            b = np.concatenate([s_a, s_b, s_r])
            t = b + CTX - 1
            x = tv3.to_f32(C.stack_context(frames, b, CTX), device)
            loss = 0.0
            for h in range(1, K + 1):
                pred = model(x, torch.from_numpy(actions[t + h - 1]).to(device))
                tgt = tv3.to_f32(frames[t + h], device)
                prev = tv3.to_f32(frames[t + h - 1], device)
                err = (pred - tgt) ** 2
                mask = ((tgt - prev).abs().amax(dim=1, keepdim=True)
                        > metrics.CHANGE_THRESHOLD).float()
                loss = loss + err.mean() + LAMBDA_CHANGE * (err * mask).sum() / (mask.sum() * 3 + 1e-8)
                x = C.roll_context(x, pred)
            loss = loss / K
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
                for be, bm in zip(ema_model.buffers(), model.buffers()):
                    be.copy_(bm)
            total += float(loss.detach())
            if step % 500 == 0 or step == steps:
                a, _ = val_report(ema_model, val, thr, device)
                print(f"K={K} pas {step}/{steps}  loss {total / step:.5f}  AUC suivi {a:.1%}  "
                      f"{time.time() - t0:.0f}s", flush=True)
                history.append({"K": K, "step": step, "loss": total / step, "auc_suivi": a})
        torch.save(ema_model.state_dict(), out / f"stage{stage}_K{K}.pt")

    torch.save(ema_model.state_dict(), out / "model_best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"termine en {time.time() - t0:.0f}s -> {out / 'model_best.pt'}")


if __name__ == "__main__":
    main()
