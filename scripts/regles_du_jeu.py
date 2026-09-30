"""
REGLES DU JEU -- batterie de sondes "physique de Minecraft" (mes regles).

Chaque sonde = une loi du jeu enoncee par un joueur expert, compilee en test
automatique. Protocole en 2 temps :
  1. CALIBRATION sur la realite : la loi doit y tenir (sinon le detecteur est faux),
  2. MESURE sur les reves : taux de respect de la loi par le world model.

Sondes v1 :
  R1  temps de cassage : sous 'taper' maintenu, un bloc met PLUSIEURS pas a
      casser (jamais instantane) -- regle #1.
  R2  persistance de destruction : un bloc casse ne reapparait pas.
  R3  invariance du HUD : hotbar/coeurs/faim ne fondent pas pendant un reve.
  R4  continuite du ciel : pas de saut de couleur du ciel en un pas.

Lance : .venv\\Scripts\\python.exe scripts\\regles_du_jeu.py <ep1> [ep2 ...] --models v4=<ckpt> v6=<ckpt>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

OUT_DIR = ROOT / "outputs" / "regles_du_jeu"
H = 32
MAX_EVENTS = 120          # evenements de cassage par episode
import os
RES_RJ = int(os.environ.get("DC_RES", "64"))
PATCH = (26 * RES_RJ // 64, 38 * RES_RJ // 64)   # croix proportionnelle
HUD_ROWS = (52 * RES_RJ // 64, RES_RJ)       # barres coeurs/faim + hotbar
SKY_ROWS = (0, 14 * RES_RJ // 64)
THETA_BREAK = 0.10        # changement fort du patch = cassage
THETA_BACK = 0.055        # retour pres de l'etat initial = reapparition


def load_ep(stem):
    if RES_RJ != 64:
        p = ROOT / "data" / f"processed_{RES_RJ}" / f"{stem}.npz"
        if p.exists():
            return C.load_episode_u8(p)
    p = ROOT / "data" / "processed" / f"{stem}.npz"
    if not p.exists():
        p = ROOT / "data" / "processed_youtube" / f"{stem}.npz"
    return C.load_episode_u8(p)


def load_model(ckpt: Path, device):
    cfg = {"base": 32, "in_frames": 1}
    cfg_file = ckpt.parent / "config.json"
    if cfg_file.exists():
        cfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


def patch_diff(frames, ref):
    """diff moyenne du patch croix vs ref. frames (...,3,64,64) float."""
    y0, y1 = PATCH
    return np.abs(frames[..., :, y0:y1, y0:y1] - ref[..., :, y0:y1, y0:y1]).mean(axis=(-3, -2, -1))


def find_break_events(actions, ok, ctx_pad=4, hold_min=8):
    """Debuts de 'taper maintenu' : taper passe 0->1 et reste >= hold_min pas."""
    tap = actions[:, 6] > 0.5
    starts = []
    for t in range(ctx_pad, len(tap) - H - 1):
        if tap[t] and not tap[t - 1] and tap[t:t + hold_min].all() and ok[t - ctx_pad + 1: t + H].all():
            starts.append(t)
    return np.array(starts, dtype=int)


def break_step(diffs):
    """Premier pas ou le patch change fort ET durablement (>=3 pas). diffs (H,)."""
    for h in range(len(diffs) - 3):
        if diffs[h] > THETA_BREAK and diffs[h:h + 3].min() > THETA_BREAK * 0.8:
            return h + 1
    return None


@torch.no_grad()
def dream(model, ctx_frames_u8, actions_seq, device):
    """ctx (B,3*ctx,64,64) u8 ; actions_seq (B,H,10) -> reves (B,H,3,64,64) f32."""
    x = torch.from_numpy(ctx_frames_u8).to(device).float() / 255.0
    res_d = ctx_frames_u8.shape[-1]
    out = np.zeros((x.shape[0], actions_seq.shape[1], 3, res_d, res_d), dtype=np.float32)
    for h in range(actions_seq.shape[1]):
        act = torch.from_numpy(actions_seq[:, h]).to(device)
        pred = model(x, act)
        out[:, h] = pred.cpu().numpy()
        x = C.roll_context(x, pred)
    return out


def main():
    args = sys.argv[1:]
    sep = args.index("--models")
    eps = args[:sep]
    models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args[sep + 1:]:
        name, ck = spec.split("=", 1)
        models[name] = load_model(Path(ck), device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {"episodes": eps, "H": H, "regles": {}}
    agg = {name: {"r1_inst": [], "r1_jamais": [], "r1_duree": [], "r2_viol": [],
                  "r3_viol": [], "r4_viol": []} for name in models}
    real_stats = {"r1_duree": [], "r1_inst": 0, "r1_total": 0,
                  "r3_thr": [], "r4_thr": []}

    for stem in eps:
        frames_u8, actions, ok = load_ep(stem)
        frames = frames_u8.astype(np.float32) / 255.0
        n = frames.shape[0]

        # --- calibration realite : seuils HUD / ciel (percentiles reels) ---
        steps = np.flatnonzero(ok)[:4000]
        hud = np.abs(frames[steps + 1][:, :, HUD_ROWS[0]:HUD_ROWS[1]]
                     - frames[steps][:, :, HUD_ROWS[0]:HUD_ROWS[1]]).mean(axis=(1, 2, 3))
        sky = np.abs(frames[steps + 1][:, :, SKY_ROWS[0]:SKY_ROWS[1]]
                     - frames[steps][:, :, SKY_ROWS[0]:SKY_ROWS[1]]).mean(axis=(1, 2, 3))
        thr_hud = float(np.percentile(hud, 99))
        thr_sky = float(np.percentile(sky, 99))
        real_stats["r3_thr"].append(thr_hud)
        real_stats["r4_thr"].append(thr_sky)

        # --- evenements de cassage reels ---
        ev = find_break_events(actions, ok)
        if len(ev) > MAX_EVENTS:
            ev = ev[np.linspace(0, len(ev) - 1, MAX_EVENTS).astype(int)]
        print(f"[{stem}] {len(ev)} evenements 'taper maintenu'")
        if len(ev) < 10:
            continue

        # realite : duree de cassage
        for t in ev:
            ref = frames[t]
            diffs = patch_diff(frames[t + 1: t + 1 + H], ref)
            b = break_step(diffs)
            real_stats["r1_total"] += 1
            if b is not None:
                real_stats["r1_duree"].append(b)
                if b <= 2:
                    real_stats["r1_inst"] += 1

        # reves : chaque modele, contexte reel, actions reelles (taper maintenu)
        ctx_starts = ev - 3
        acts_seq = np.stack([actions[t: t + H] for t in ev]).astype(np.float32)
        for name, (model, ctx) in models.items():
            ctxs = C.stack_context(frames_u8, ev - ctx + 1, ctx)
            dreams = dream(model, ctxs, acts_seq, device)
            for j, t in enumerate(ev):
                ref = frames[t]
                diffs = patch_diff(dreams[j], ref)
                b = break_step(diffs)
                if b is None:
                    agg[name]["r1_jamais"].append(1)
                else:
                    agg[name]["r1_jamais"].append(0)
                    agg[name]["r1_duree"].append(b)
                    agg[name]["r1_inst"].append(1 if b <= 2 else 0)
                    # R2 : reapparition apres cassage ?
                    post = diffs[b + 2:]
                    agg[name]["r2_viol"].append(
                        1 if (len(post) > 2 and (post < THETA_BACK).sum() >= 2) else 0)
                # R3/R4 : HUD et ciel dans le reve (pas a pas)
                dh = np.abs(np.diff(np.concatenate([frames[t][None], dreams[j]]), axis=0))
                agg[name]["r3_viol"].append(
                    float((dh[:, :, HUD_ROWS[0]:HUD_ROWS[1]].mean(axis=(1, 2, 3)) > thr_hud).mean()))
                agg[name]["r4_viol"].append(
                    float((dh[:, :, SKY_ROWS[0]:SKY_ROWS[1]].mean(axis=(1, 2, 3)) > thr_sky).mean()))

    # --- table ---
    rd = real_stats["r1_duree"]
    results["realite"] = {
        "r1_duree_mediane_pas": float(np.median(rd)) if rd else None,
        "r1_pct_instantane": real_stats["r1_inst"] / max(real_stats["r1_total"], 1),
        "r1_evenements": real_stats["r1_total"],
    }
    print("\n=== REALITE (calibration) ===")
    print(f"cassage : duree mediane {results['realite']['r1_duree_mediane_pas']} pas | "
          f"instantane {results['realite']['r1_pct_instantane']:.0%} "
          f"(doit etre ~0) | n={real_stats['r1_total']}")

    print("\n=== REVES - taux de respect des regles ===")
    print(f"{'modele':<8} {'R1 casse-en-temps':>18} {'R1 jamais-casse':>16} "
          f"{'R1 duree med':>13} {'R2 pas-de-retour':>17} {'R3 HUD stable':>14} {'R4 ciel continu':>15}")
    for name, a in agg.items():
        inst = np.mean(a["r1_inst"]) if a["r1_inst"] else float("nan")
        jam = np.mean(a["r1_jamais"]) if a["r1_jamais"] else float("nan")
        dur = np.median(a["r1_duree"]) if a["r1_duree"] else float("nan")
        r2 = 1 - np.mean(a["r2_viol"]) if a["r2_viol"] else float("nan")
        r3 = 1 - np.mean(a["r3_viol"]) if a["r3_viol"] else float("nan")
        r4 = 1 - np.mean(a["r4_viol"]) if a["r4_viol"] else float("nan")
        print(f"{name:<8} {1-inst:>17.0%} {jam:>15.0%} {dur:>12.1f} "
              f"{r2:>16.0%} {r3:>13.0%} {r4:>14.0%}")
        results["regles"][name] = {
            "r1_respect_duree": float(1 - inst), "r1_jamais_casse": float(jam),
            "r1_duree_mediane": float(dur), "r2_pas_de_reapparition": float(r2),
            "r3_hud_stable": float(r3), "r4_ciel_continu": float(r4),
        }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_DIR.relative_to(ROOT)}\\results.json")


if __name__ == "__main__":
    main()
