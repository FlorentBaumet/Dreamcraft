"""
Phase 0 -- LE test : un mini modele bat-il les baselines, evalue honnetement ?

Protocole (gele dans CADRE_DREAMCRAFT.md #5-6) :
  - split par episode : 2 episodes d'entrainement, 1 episode jamais vu (val),
  - verdict sur la MSE Delta-region (pixels qui changent) vs B0 (copie) et
    B1 (flux optique), recalcules sur LE MEME episode held-out,
  - + test contrefactuel qualitatif (meme frame, actions differentes).

Sorties : outputs/phase0/  (checkpoint, results.json, GO_NOGO.md, visuels)
Lance   : .venv\\Scripts\\python.exe scripts\\train_phase0.py [val_stem] [exclu1 exclu2 ...]

val_stem : episode held-out (defaut : dernier par ordre alphabetique).
exclus   : episodes retires de l'entrainement SANS servir de val ici
           (ex. un val de generalisation garde pour plus tard).
Variables d'env : DC_TAG (nom du run wandb / sous-dossier de sortie).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import vpt  # noqa: E402
from dreamcraft.eval import baselines, metrics  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "phase0"

EPOCHS = 15
BATCH = 64
LR = 3e-4
LAMBDA_CHANGE = 5.0   # poids de la loss sur les pixels changeants
SEED = 0


def load_pairs(npz_path):
    """Retourne (frames float32 (N,3,64,64), actions (N-1,10), ok (N-1,) bool)."""
    frames_u8, actions, gui = vpt.load_episode(npz_path)
    frames = frames_u8.astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))  # NHWC -> NCHW
    n = frames.shape[0]
    ok = np.ones(n - 1, dtype=bool)
    ok[gui[: n - 1] == 1] = False
    ok[1:] &= gui[: n - 2] == 0  # exclut aussi si GUI a t-1 (coherent avec l'eval)
    ok[0] = False                # B1 a besoin de t-1 : on aligne les indices evalues
    return frames, actions, ok


def batches(idx, size, rng=None):
    if rng is not None:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), size):
        yield idx[i: i + size]


@torch.no_grad()
def val_delta_mse(model, frames, actions, ok, device, stride=10):
    """MSE Delta-region du modele sur l'episode val (1 paire sur `stride`)."""
    model.eval()
    idx = np.flatnonzero(ok)[::stride]
    total, count = 0.0, 0
    for b in batches(idx, 256):
        cur = torch.from_numpy(frames[b]).to(device)
        act = torch.from_numpy(actions[b]).to(device)
        tgt = torch.from_numpy(frames[b + 1]).to(device)
        pred = model(cur, act)
        mask = ((tgt - cur).abs().amax(dim=1, keepdim=True) > metrics.CHANGE_THRESHOLD)
        for j in range(len(b)):
            m = mask[j, 0]
            if int(m.sum()) < metrics.MIN_CHANGED_PIXELS:
                continue
            err = ((pred[j] - tgt[j]) ** 2)[:, m].mean()
            total += float(err)
            count += 1
    return total / max(count, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    import os
    run_name = os.environ.get("DC_TAG", "phase0-conv64")
    out_dir = OUT_DIR if run_name == "phase0-conv64" else ROOT / "outputs" / run_name
    npz_files = sorted(PROC_DIR.glob("*.npz"))
    if len(sys.argv) > 1:
        val_stem = sys.argv[1]
        excl = set(sys.argv[2:]) | {val_stem}
        val_ep = PROC_DIR / f"{val_stem}.npz"
        train_eps = [p for p in npz_files if p.stem not in excl]
    else:
        train_eps, val_ep = npz_files[:-1], npz_files[-1]
    print(f"train: {len(train_eps)} episodes")
    for p in train_eps:
        print(f"   - {p.stem}")
    print(f"val  : {val_ep.stem}  (jamais vu a l'entrainement)")

    tr = [load_pairs(p) for p in train_eps]
    vf, va, vok = load_pairs(val_ep)

    model = ConvPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parametres: {n_params/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    run = None
    try:
        import wandb
        run = wandb.init(project="dreamcraft", name=run_name,
                         config={"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                                 "lambda_change": LAMBDA_CHANGE, "params": n_params,
                                 "train_eps": [p.stem for p in train_eps],
                                 "val_ep": val_ep.stem},
                         settings=wandb.Settings(init_timeout=30))
        print("wandb: ok")
    except Exception as e:
        print(f"wandb indisponible ({type(e).__name__}) -> on continue sans")

    best = float("inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model_best.pt"

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, n_steps = 0.0, 0
        for frames, actions, ok in tr:
            idx = np.flatnonzero(ok)
            for b in batches(idx, BATCH, rng):
                cur = torch.from_numpy(frames[b]).to(device)
                act = torch.from_numpy(actions[b]).to(device)
                tgt = torch.from_numpy(frames[b + 1]).to(device)
                pred = model(cur, act)
                err = (pred - tgt) ** 2
                mask = ((tgt - cur).abs().amax(dim=1, keepdim=True)
                        > metrics.CHANGE_THRESHOLD).float()
                loss_global = err.mean()
                loss_change = (err * mask).sum() / (mask.sum() * 3 + 1e-8)
                loss = loss_global + LAMBDA_CHANGE * loss_change
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss)
                n_steps += 1

        vd = val_delta_mse(model, vf, va, vok, device)
        marker = ""
        if vd < best:
            best = vd
            torch.save(model.state_dict(), ckpt)
            marker = "  <- best, sauve"
        print(f"epoch {epoch:2d}/{EPOCHS}  loss {ep_loss/n_steps:.5f}  "
              f"val dMSE {vd:.5f}{marker}")
        if run:
            run.log({"epoch": epoch, "train_loss": ep_loss / n_steps, "val_delta_mse": vd})

    print(f"entrainement: {time.time()-t0:.0f}s  best val dMSE {best:.5f}")

    # ================= EVAL FINALE (protocole gele, episode held-out) =========
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    idx = np.flatnonzero(vok)
    agg = {k: {"n": 0, "g": {"B0": 0.0, "B1": 0.0, "M": 0.0},
               "nd": 0, "d": {"B0": 0.0, "B1": 0.0, "M": 0.0}}
           for k in ["tout", "taper", "camera", "deplacement", "idle"]}
    examples = []

    with torch.no_grad():
        for b in batches(idx, 256):
            cur_t = torch.from_numpy(vf[b]).to(device)
            act_t = torch.from_numpy(va[b]).to(device)
            preds = model(cur_t, act_t).cpu().numpy()
            for j, t in enumerate(b):
                cur = np.transpose(vf[t], (1, 2, 0))
                prev = np.transpose(vf[t - 1], (1, 2, 0))
                tgt = np.transpose(vf[t + 1], (1, 2, 0))
                pm = np.transpose(preds[j], (1, 2, 0))
                p0 = baselines.b0_copy(cur)
                p1 = baselines.b1_optical_flow(prev, cur)

                g = {"B0": metrics.mse(p0, tgt), "B1": metrics.mse(p1, tgt),
                     "M": metrics.mse(pm, tgt)}
                d0, nc = metrics.delta_region_mse(p0, tgt, cur)
                d1, _ = metrics.delta_region_mse(p1, tgt, cur)
                dm, _ = metrics.delta_region_mse(pm, tgt, cur)
                static = np.isnan(d0)

                a = va[t]
                subs = ["tout"]
                if a[6] > 0.5:
                    subs.append("taper")
                if abs(a[8]) > 0.05 or abs(a[9]) > 0.05:
                    subs.append("camera")
                if a[:4].max() > 0.5:
                    subs.append("deplacement")
                if len(subs) == 1:
                    subs.append("idle")
                for k in subs:
                    agg[k]["n"] += 1
                    for name, v in g.items():
                        agg[k]["g"][name] += v
                    if not static:
                        agg[k]["nd"] += 1
                        for name, v in {"B0": d0, "B1": d1, "M": dm}.items():
                            agg[k]["d"][name] += v
                if not static and nc > 200:
                    tag = ("taper" if "taper" in subs
                           else "camera" if "camera" in subs else "autre")
                    examples.append((nc, tag, t, cur, tgt, pm, p1))

    results = {}
    for k, a in agg.items():
        if a["n"] == 0:
            continue
        r = {"paires": a["n"],
             "globale": {m: a["g"][m] / a["n"] for m in a["g"]}}
        if a["nd"]:
            r["paires_dynamiques"] = a["nd"]
            r["delta"] = {m: a["d"][m] / a["nd"] for m in a["d"]}
        results[k] = r

    dm_all = results["tout"]["delta"]
    go = dm_all["M"] < dm_all["B0"] and dm_all["M"] < dm_all["B1"]
    verdict = "GO" if go else "NO-GO"

    # visuels : modele vs reel vs B1
    examples.sort(key=lambda e: -e[0])
    seen = set()
    for nc, tag, t, cur, tgt, pm, p1 in examples:
        if tag in seen:
            continue
        seen.add(tag)
        row = np.concatenate([cur, tgt, pm, p1], axis=1)
        big = cv2.resize((row * 255).astype(np.uint8), None, fx=4, fy=4,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"pred_{tag}_t{t}.png"),
                    cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        if len(seen) >= 3:
            break

    # test contrefactuel : meme frame, 5 actions differentes
    probe_actions = {
        "rien": np.zeros(10, dtype=np.float32),
        "cam_gauche": np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0], dtype=np.float32),
        "cam_droite": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0], dtype=np.float32),
        "avancer": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "taper": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    }
    probe_ts = idx[[len(idx) // 4, len(idx) // 2]]
    cf_summary = {}
    with torch.no_grad():
        for t in probe_ts:
            cur_t = torch.from_numpy(vf[t][None]).to(device).repeat(len(probe_actions), 1, 1, 1)
            acts = torch.from_numpy(np.stack(list(probe_actions.values()))).to(device)
            preds = model(cur_t, acts).cpu().numpy()
            base_pred = preds[0]
            row_p, row_d, diffs = [], [], {}
            for i, name in enumerate(probe_actions):
                p = np.transpose(preds[i], (1, 2, 0))
                row_p.append(p)
                diff = np.abs(preds[i] - base_pred).max(axis=0)
                diffs[name] = float(diff.mean())
                row_d.append(np.repeat(np.clip(diff * 8, 0, 1)[..., None], 3, axis=-1))
            sheet = np.concatenate([np.concatenate(row_p, axis=1),
                                    np.concatenate(row_d, axis=1)], axis=0)
            big = cv2.resize((sheet * 255).astype(np.uint8), None, fx=4, fy=4,
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_dir / f"contrefactuel_t{t}.png"),
                        cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
            cf_summary[f"t{t}"] = diffs

    out = {"verdict": verdict, "resultats": results, "contrefactuel": cf_summary,
           "meta": {"params": n_params, "epochs": EPOCHS, "seed": SEED,
                    "val_ep": val_ep.stem, "lambda_change": LAMBDA_CHANGE}}
    (out_dir / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        f"# Phase 0 - verdict : **{verdict}**",
        "",
        f"Episode held-out : `{val_ep.stem}` - {results['tout']['paires']} paires, "
        f"{results['tout']['paires_dynamiques']} dynamiques. Modele : {n_params/1e6:.2f} M params.",
        "",
        "| | MSE globale | MSE Δ-region (l'honnête) |",
        "|---|---|---|",
        f"| B0 copie | {results['tout']['globale']['B0']:.5f} | {dm_all['B0']:.5f} |",
        f"| B1 flux optique | {results['tout']['globale']['B1']:.5f} | {dm_all['B1']:.5f} |",
        f"| **Modèle** | **{results['tout']['globale']['M']:.5f}** | **{dm_all['M']:.5f}** |",
        "",
        "## Par type d'action (MSE Δ-region)",
        "| subset | B0 | B1 | Modèle |",
        "|---|---|---|---|",
    ]
    for k in ["taper", "camera", "deplacement", "idle"]:
        if k in results and "delta" in results[k]:
            d = results[k]["delta"]
            md.append(f"| {k} | {d['B0']:.5f} | {d['B1']:.5f} | {d['M']:.5f} |")
    md += ["", "## Contrefactuel (variation moyenne de la prediction vs action nulle)",
           "```", json.dumps(cf_summary, indent=2), "```", ""]
    (out_dir / "GO_NOGO.md").write_text("\n".join(md), encoding="utf-8")

    if run:
        run.summary["verdict"] = verdict
        run.summary["delta_mse_model"] = dm_all["M"]
        run.summary["delta_mse_b0"] = dm_all["B0"]
        run.summary["delta_mse_b1"] = dm_all["B1"]
        run.finish()

    print(f"\n=== VERDICT : {verdict} ===")
    print(f"dMSE  B0 {dm_all['B0']:.5f} | B1 {dm_all['B1']:.5f} | modele {dm_all['M']:.5f}")
    print(f"Sorties: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
