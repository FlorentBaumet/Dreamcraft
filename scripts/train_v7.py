"""
v7 -- ANTI-TIMIDITE : entrainement pondere-evenements (lecon 10).

Diagnostic (batterie regles du jeu) : le reveur connait les lois d'etat mais
refuse les transitions d'evenement (81 % des blocs jamais casses, 90 % des
chutes sans atterrissage). Cause : les moments d'engagement sont rares dans
la loss -> le biais MSE "que rien n'advienne" gagne.

Remede v7 (UN seul levier, attribution propre) :
  50 % de chaque batch est seede sur un EVENEMENT D'ENGAGEMENT mine
  (outputs/mine_events/breaks.json, 2304 debuts de 'taper maintenu'),
  avec jitter pour que l'evenement tombe DANS les K pas reves.

Selection : taux d'engagement reve sur les evenements du val (le reve
casse-t-il ?) SOUS GARDE de fidelite (dRoll@16 <= 1.15x la reference).
K plafonne a 12 (la zone K=16 declenchait le crawl VRAM WDDM, mystere ouvert).

Configuration (variables d environnement, toutes optionnelles) :
  DC_TAG            nom du run -> outputs/<DC_TAG>/        (defaut v7-antitimidite)
  DC_SRC            checkpoint de depart (warm start)     (defaut outputs/v6-nvcave)
  DC_RES            resolution 64 | 96                    (defaut 64)
  DC_BASE           largeur du reseau (48 = 3,65 M params)
  DC_STAGES         curriculum JSON [[K, lr, epochs, steps, batch], ...]
  DC_COMMIT         1 = active la commit-loss (penalise les reves figes)
  DC_LAMBDA_COMMIT  poids de la commit-loss               (defaut 2.0)
  DC_GRAD_CKPT      1 = gradient checkpointing (VRAM constante en K)
  DC_VAL_ENG        episode de validation d engagement    (exclu du train)
  DC_EXCL_EXTRA     episodes supplementaires a exclure, separes par des virgules
  DC_DATA           dossiers de donnees en plus, ex. processed_agent
  DC_ONLY_DATA      1 = n utiliser QUE les dossiers de DC_DATA

Sorties : outputs/<DC_TAG=v7-antitimidite>/
Lance   : .venv\\Scripts\\python.exe scripts\\train_v7.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.eval import metrics  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

_spec = importlib.util.spec_from_file_location("tv3", Path(__file__).parent / "train_v3.py")
tv3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv3)

RES = int(os.environ.get("DC_RES", "64"))
if RES == 64:
    PROC = ROOT / "data" / "processed"
    PROC_YT = ROOT / "data" / "processed_youtube"
else:   # corpus re-ingere a une autre resolution (un seul dossier)
    PROC = ROOT / "data" / f"processed_{RES}"
    PROC_YT = PROC
# dossiers de donnees supplementaires (ex : DC_DATA=processed_agent pour les
# donnees collectees en jouant, labels d actions exacts + domaine exact)
EXTRA_DIRS = [ROOT / "data" / d.strip()
              for d in os.environ.get("DC_DATA", "").split(",") if d.strip()]
ONLY_EXTRA = os.environ.get("DC_ONLY_DATA", "0") == "1"
SRC_CKPT = Path(os.environ.get("DC_SRC",
                str(ROOT / "outputs" / "v6-nvcave" / "model_best.pt")))

CTX, BASE = 4, int(os.environ.get("DC_BASE", "48"))
LAMBDA_CHANGE = 5.0
SEED = 0
GRAD_CKPT = os.environ.get("DC_GRAD_CKPT", "0") == "1"
EVENT_SHARE = 0.5
STAGES = json.loads(os.environ.get("DC_STAGES",
    "[[8, 4e-5, 3, 1000, 64], [12, 3e-5, 3, 800, 48]]"))
EMA_DECAY = 0.999
VAL_H = 16
ENG_H = 14            # horizon du test d'engagement
PATCH = (26 * RES // 64, 38 * RES // 64)
THETA_BREAK = 0.10
VAL_STEM = "Player129-f153ac423f61-20210617-173110"
VAL_ENG_STEM = os.environ.get("DC_VAL_ENG", "nvcave_012")  # val engagement (in-domain)
COMMIT = os.environ.get("DC_COMMIT", "0") == "1"
LAMBDA_COMMIT = float(os.environ.get("DC_LAMBDA_COMMIT", "2.0"))
Y0, Y1 = PATCH
EXCL = {VAL_STEM, "treechop-984393664dfd-20210924-174326",
        "Player871-2e9a64a90d31-20210627-154641",
        "Player309-dcc21a4f8784-20210721-163058", "nvcave_008", "nvcave_026"}
EXCL |= {s for s in os.environ.get("DC_EXCL_EXTRA", "").split(",") if s}
EXCL.add(VAL_ENG_STEM)   # le val engagement doit etre hors du train


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


def find_break_events_local(actions, ok, hold=6):
    tap = actions[:, 6] > 0.5
    ev = []
    for t in range(6, len(tap) - 20):
        if tap[t] and not tap[t - 1] and tap[t:t + hold].all() and ok[t - 5:t + ENG_H].all():
            ev.append(t)
    return np.array(ev, dtype=int)


@torch.no_grad()
def val_engagement(model, vf, va, vok, ev, device):
    """% des evenements du val ou le reve CASSE (patch croix change durablement)."""
    if len(ev) == 0:
        return float("nan")
    model.eval()
    frames = vf.astype(np.float32) / 255.0
    ok_n = 0
    for i in range(0, len(ev), 128):
        b = ev[i:i + 128]
        x = torch.from_numpy(C.stack_context(vf, b - CTX + 1, CTX)).to(device).float() / 255.0
        acts = np.stack([va[t: t + ENG_H] for t in b]).astype(np.float32)
        dreams = np.zeros((len(b), ENG_H, 3, RES, RES), dtype=np.float32)
        for h in range(ENG_H):
            pred = model(x, torch.from_numpy(acts[:, h]).to(device))
            dreams[:, h] = pred.cpu().numpy()
            x = C.roll_context(x, pred)
        y0, y1 = PATCH
        for j, t in enumerate(b):
            ref = frames[t][:, y0:y1, y0:y1]
            diffs = np.abs(dreams[j][:, :, y0:y1, y0:y1] - ref).mean(axis=(1, 2, 3))
            for h in range(len(diffs) - 3):
                if diffs[h] > THETA_BREAK and diffs[h:h + 3].min() > THETA_BREAK * 0.8:
                    ok_n += 1
                    break
    model.train()
    return ok_n / len(ev)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = os.environ.get("DC_TAG", "v7-antitimidite")
    out_dir = ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = [] if ONLY_EXTRA else sorted(PROC.glob("*.npz"))
    if PROC_YT != PROC and not ONLY_EXTRA:
        npz_files += sorted(PROC_YT.glob("*.npz"))
    for d in EXTRA_DIRS:
        if not d.is_dir():
            raise FileNotFoundError(f"DC_DATA : dossier introuvable {d}")
        found = sorted(d.glob("*.npz"))
        print(f"v7: + {len(found)} episodes depuis {d.name}")
        npz_files += found
    train_eps = [p for p in npz_files if p.stem not in EXCL]
    frames, actions, ok = tv3.build_corpus(train_eps, res=RES)
    print(f"corpus: {frames.shape[0]} frames ({frames.nbytes/1e9:.2f} Go)")

    # offsets globaux pour mapper les evenements mines
    sizes = [int(np.load(p)["gui"].shape[0]) + 1 for p in train_eps]
    offset = {}
    off = 0
    for p, nsz in zip(train_eps, sizes):
        offset[p.stem] = off
        off += nsz

    breaks = json.loads((ROOT / "outputs" / "mine_events" / "breaks.json").read_text(encoding="utf-8"))
    ev_by_ep = defaultdict(list)
    for e in breaks:
        if e["ep"] in offset:
            ev_by_ep[e["ep"]].append(e["t"])
    print(f"{sum(len(v) for v in ev_by_ep.values())} evenements d'engagement dans le train")

    vf, va, vok = C.load_episode_u8(PROC / f"{VAL_STEM}.npz")
    # engagement mesure sur un episode CLAIR (le val sombre etait aveugle : lecon v7)
    ef, ea, eok = C.load_episode_u8(PROC_YT / f"{VAL_ENG_STEM}.npz")
    vev = find_break_events_local(ea, np.concatenate([eok, [False]])[:len(ea)])
    print(f"val engagement ({VAL_ENG_STEM}, clair): {len(vev)} evenements | commit-loss: {COMMIT}")

    model = ConvPredictor(base=BASE, in_frames=CTX).to(device)
    model.load_state_dict(torch.load(SRC_CKPT, weights_only=True))
    (out_dir / "config.json").write_text(
        json.dumps({"base": BASE, "in_frames": CTX, "action_dim": 10,
                    "warm_start": str(SRC_CKPT), "event_share": EVENT_SHARE}), encoding="utf-8")
    ema = Ema(model, EMA_DECAY)
    eval_model = ConvPredictor(base=BASE, in_frames=CTX).to(device)

    ref_roll = tv3.val_honest_rollout(model, vf, va, vok, device, ctx=CTX, h_max=VAL_H)
    ref_eng = val_engagement(model, ef, ea, eok, vev, device)
    guard = ref_roll * 1.15
    print(f"references v6-lite : engagement {ref_eng:.0%} | dRoll@{VAL_H} {ref_roll:.5f} "
          f"(garde <= {guard:.5f})")

    run = None
    try:
        import wandb
        run = wandb.init(project="dreamcraft", name=run_name,
                         config={"stages": str(STAGES), "event_share": EVENT_SHARE,
                                 "warm_start": "v6-lite", "ref_engagement": ref_eng,
                                 "ref_droll16": ref_roll},
                         settings=wandb.Settings(init_timeout=30))
        print("wandb: ok")
    except Exception as e:
        print(f"wandb indisponible ({type(e).__name__})")

    best_eng = ref_eng
    ckpt = out_dir / "model_best.pt"
    torch.save(model.state_dict(), ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=STAGES[0][1])
    t0 = time.time()
    ep_num = 0

    for K, lr, n_ep, max_steps, batch in STAGES:
        for g in opt.param_groups:
            g["lr"] = lr
        # pool d'evenements mappe en indices globaux, avec jitter dans [0, K-2]
        ev_pool = []
        for stem, ts in ev_by_ep.items():
            base_off = offset[stem]
            for t in ts:
                ev_pool.append(base_off + t)
        ev_pool = np.array(ev_pool, dtype=np.int64)
        valid_set = np.zeros(len(ok) + 1, dtype=bool)
        vs = C.valid_starts(ok, CTX, K)
        valid_set[vs] = True
        print(f"[stage] K={K} lr={lr:.1e} x{n_ep} (batch {batch}) | pool evenements {len(ev_pool)}")

        for _ in range(n_ep):
            ep_num += 1
            rand_starts = rng.permutation(vs)[: max_steps * batch]
            model.train()
            ep_loss, n_steps = 0.0, 0
            for i in range(0, len(rand_starts), batch):
                nb = min(batch, len(rand_starts) - i)
                n_ev = int(nb * EVENT_SHARE)
                # seeds evenement : l'evenement tombe au pas jitter+1 du reve
                picks = ev_pool[rng.integers(0, len(ev_pool), n_ev * 3)]
                jit = rng.integers(0, max(K - 1, 1), len(picks))
                s_ev = picks - CTX + 1 - jit
                s_ev = s_ev[(s_ev >= 0) & (s_ev < len(valid_set))]
                s_ev = s_ev[valid_set[s_ev]][:n_ev]
                b = np.concatenate([s_ev, rand_starts[i: i + nb - len(s_ev)]])
                if len(b) == 0:
                    continue
                t = b + CTX - 1
                x = tv3.to_f32(C.stack_context(frames, b, CTX), device)
                loss = 0.0
                for h in range(1, K + 1):
                    act = torch.from_numpy(actions[t + h - 1]).to(device)
                    if GRAD_CKPT:
                        # activations jetees et recalculees au backward :
                        # memoire O(1) pas au lieu de O(K), cout ~1.5-2x temps
                        pred = torch.utils.checkpoint.checkpoint(
                            model, x, act, use_reentrant=False)
                    else:
                        pred = model(x, act)
                    tgt = tv3.to_f32(frames[t + h], device)
                    prev = tv3.to_f32(frames[t + h - 1], device)
                    err = (pred - tgt) ** 2
                    mask = ((tgt - prev).abs().amax(dim=1, keepdim=True)
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
            eng = val_engagement(eval_model, ef, ea, eok, vev, device)
            roll = tv3.val_honest_rollout(eval_model, vf, va, vok, device, ctx=CTX, h_max=VAL_H)
            marker = ""
            if eng > best_eng and roll <= guard:
                best_eng = eng
                torch.save(eval_model.state_dict(), ckpt)
                marker = "  <- best (engagement+garde), sauve"
            print(f"epoch {ep_num} K={K}  loss {ep_loss/max(n_steps,1):.5f}  "
                  f"engagement {eng:.0%}  dRoll@{VAL_H} {roll:.5f}{marker}", flush=True)
            if run:
                run.log({"epoch": ep_num, "K": K, "train_loss": ep_loss / max(n_steps, 1),
                         "val_engagement": eng, "val_droll16": roll})

    ema.copy_to(eval_model)
    torch.save(eval_model.state_dict(), out_dir / "model_last.pt")
    print(f"v7: {time.time()-t0:.0f}s | engagement {ref_eng:.0%} -> {best_eng:.0%}")
    if run:
        run.summary.update({"ref_engagement": ref_eng, "best_engagement": best_eng})
        run.finish()


if __name__ == "__main__":
    main()
