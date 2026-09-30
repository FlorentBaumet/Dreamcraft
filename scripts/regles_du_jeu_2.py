"""
REGLES DU JEU v2 -- sondes #5 (degats de chute) et #10 (manger -> faim).

#5 CHUTE (regle : tomber de >3 blocs -> degats proportionnels)
   - chute reelle = flux optique vertical fort et soutenu (la scene "monte"),
   - atterrissage = arret brutal du flux,
   - degats = flash rouge plein ecran juste apres.
   Calibration realite d'abord (l'eau annule les degats -> le taux reel
   n'est pas 100 %), puis reves seedes en pleine chute : le modele
   atterrit-il ? et inflige-t-il les degats ?

#10 FAIM (regle : clic droit maintenu avec nourriture -> barre remonte)
   - manger reel = 'utiliser' maintenu >=24 pas sans taper,
   - la zone faim du HUD (au-dessus de la hotbar, a droite) doit changer.
   Calibration : le changement est-il seulement DETECTABLE a 64px ?
   Si non -> verdict honnete "non testable a cette resolution".

Lance : .venv\\Scripts\\python.exe scripts\\regles_du_jeu_2.py <ep1> [...] --models n=ckpt ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

OUT_DIR = ROOT / "outputs" / "regles_du_jeu"

FALL_DY = -1.2        # px/frame : la scene monte -> le joueur tombe
FALL_MIN = 8          # pas consecutifs (>3 blocs de chute)
FLASH_DELTA = 0.03    # rouge relatif au-dessus de la baseline
EAT_HOLD = 24
HUNGER = (slice(52, 56), slice(33, 50))   # zone faim (rows, cols)
H_FALL = 16
H_EAT = 32


def load_ep(stem):
    p = ROOT / "data" / "processed" / f"{stem}.npz"
    if not p.exists():
        p = ROOT / "data" / "processed_youtube" / f"{stem}.npz"
    return C.load_episode_u8(p)


def load_model(ckpt: Path, device):
    cfg = {"base": 32, "in_frames": 1}
    cf = ckpt.parent / "config.json"
    if cf.exists():
        cfg.update(json.loads(cf.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


def flow_dy(prev_g, cur_g):
    f = cv2.calcOpticalFlowFarneback(prev_g, cur_g, None, 0.5, 2, 9, 3, 5, 1.1, 0)
    return float(f[8:48, 8:56, 1].mean())


def grayscale_seq(frames_f32):
    """(N,3,64,64) float -> liste de gris uint8."""
    g = (frames_f32.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    return [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in g]


def redness(frame_f32):
    """Indice de flash rouge (hors HUD)."""
    r, g, b = frame_f32[0, :50], frame_f32[1, :50], frame_f32[2, :50]
    return float((r - (g + b) / 2).mean())


@torch.no_grad()
def dream(model, ctx_u8, acts, device):
    x = torch.from_numpy(ctx_u8).to(device).float() / 255.0
    out = np.zeros((x.shape[0], acts.shape[1], 3, 64, 64), dtype=np.float32)
    for h in range(acts.shape[1]):
        a = torch.from_numpy(acts[:, h]).to(device)
        pred = model(x, a)
        out[:, h] = pred.cpu().numpy()
        x = C.roll_context(x, pred)
    return out


def main():
    args = sys.argv[1:]
    sep = args.index("--models")
    eps = args[:sep]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for spec in args[sep + 1:]:
        name, ck = spec.split("=", 1)
        models[name] = load_model(Path(ck), device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = {"episodes": eps, "chute": {"realite": {}, "reves": {}},
           "faim": {"realite": {}, "reves": {}}}
    fall_real = {"n": 0, "flash": 0}
    fall_dream = {n: {"n": 0, "atterrit": 0, "flash": 0} for n in models}
    eat_real = {"n": 0, "delta": [], "delta_null": []}
    eat_dream = {n: {"n": 0, "delta": []} for n in models}

    for stem in eps:
        frames_u8, actions, ok = load_ep(stem)
        frames = frames_u8.astype(np.float32) / 255.0
        n = frames.shape[0]
        grays = grayscale_seq(frames)
        print(f"[{stem}] flux optique...", flush=True)
        dys = np.array([flow_dy(grays[t], grays[t + 1]) for t in range(n - 1)])

        # ---------- #5 CHUTE ----------
        falling = dys < FALL_DY
        t = 0
        fall_events = []
        while t < len(falling) - FALL_MIN - H_FALL:
            if falling[t: t + FALL_MIN].all():
                end = t + FALL_MIN
                while end < len(falling) and falling[end]:
                    end += 1
                if end + 4 < n and ok[max(t - 4, 0): min(end + 4, len(ok))].all():
                    fall_events.append((t, end))     # (debut, atterrissage)
                t = end + 10
            else:
                t += 1
        for (t0, land) in fall_events:
            base = np.median([redness(frames[j]) for j in range(max(t0 - 10, 0), t0)])
            post = max(redness(frames[j]) for j in range(land, min(land + 4, n)))
            fall_real["n"] += 1
            if post - base > FLASH_DELTA:
                fall_real["flash"] += 1
        print(f"  chutes reelles: {len(fall_events)}")

        # reves seedes a mi-chute
        seeds = [(t0 + 4, land) for (t0, land) in fall_events
                 if t0 + 4 + H_FALL < n and land - t0 - 4 < H_FALL - 2]
        if seeds:
            acts = np.stack([actions[s: s + H_FALL] for s, _ in seeds]).astype(np.float32)
            for name, (model, ctx) in models.items():
                starts = np.array([s for s, _ in seeds])
                ctxs = C.stack_context(frames_u8, starts - ctx + 1, ctx)
                dr = dream(model, ctxs, acts, device)
                for j, (s, land) in enumerate(seeds):
                    dg = grayscale_seq(dr[j])
                    ddys = [flow_dy(dg[h], dg[h + 1]) for h in range(len(dg) - 1)]
                    # atterrissage reve : le flux vertical s'arrete
                    landed = None
                    for h in range(1, len(ddys)):
                        if ddys[h - 1] < FALL_DY and ddys[h] > FALL_DY / 2:
                            landed = h
                            break
                    fall_dream[name]["n"] += 1
                    if landed is not None:
                        fall_dream[name]["atterrit"] += 1
                        base = redness(frames[s - 1])
                        post = max(redness(dr[j][h]) for h in range(landed, min(landed + 4, len(dg))))
                        if post - base > FLASH_DELTA:
                            fall_dream[name]["flash"] += 1

        # ---------- #10 FAIM ----------
        use = (actions[:, 7] > 0.5) & (actions[:, 6] < 0.5)
        d = np.diff(np.concatenate([[0], use.astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        eats = [(s, e) for s, e in zip(starts, ends)
                if e - s >= EAT_HOLD and s > 4 and e + 4 < n and ok[s - 4:e].all()]
        rng = np.random.default_rng(0)
        for (s, e) in eats:
            pre = frames[s - 2][:, HUNGER[0], HUNGER[1]]
            post = frames[min(e + 2, n - 1)][:, HUNGER[0], HUNGER[1]]
            eat_real["n"] += 1
            eat_real["delta"].append(float(np.abs(post - pre).mean()))
            # null : meme duree, moment aleatoire sans use
            t0 = int(rng.integers(5, n - EAT_HOLD - 5))
            eat_real["delta_null"].append(float(np.abs(
                frames[t0 + EAT_HOLD][:, HUNGER[0], HUNGER[1]]
                - frames[t0][:, HUNGER[0], HUNGER[1]]).mean()))
        print(f"  repas reels: {len(eats)}")

        if eats:
            seeds = [s for s, e in eats if s + H_EAT < n]
            acts = np.stack([actions[s: s + H_EAT] for s in seeds]).astype(np.float32)
            for name, (model, ctx) in models.items():
                ctxs = C.stack_context(frames_u8, np.array(seeds) - ctx + 1, ctx)
                dr = dream(model, ctxs, acts, device)
                for j, s in enumerate(seeds):
                    pre = frames[s - 1][:, HUNGER[0], HUNGER[1]]
                    post = dr[j][-1][:, HUNGER[0], HUNGER[1]]
                    eat_dream[name]["n"] += 1
                    eat_dream[name]["delta"].append(float(np.abs(post - pre).mean()))

    # ---------- verdicts ----------
    print("\n=== #5 CHUTE ===")
    pr = fall_real["flash"] / max(fall_real["n"], 1)
    print(f"realite : {fall_real['n']} chutes, flash rouge apres atterrissage : {pr:.0%} "
          f"(pas 100% attendu : atterrissages dans l'eau)")
    res["chute"]["realite"] = {"n": fall_real["n"], "taux_flash": pr}
    for name, s in fall_dream.items():
        at = s["atterrit"] / max(s["n"], 1)
        fl = s["flash"] / max(s["atterrit"], 1)
        print(f"{name:8s}: atterrit dans le reve {at:.0%} | flash si atterri {fl:.0%} (n={s['n']})")
        res["chute"]["reves"][name] = {"n": s["n"], "atterrit": at, "flash_si_atterri": fl}

    print("\n=== #10 FAIM ===")
    dm = float(np.median(eat_real["delta"])) if eat_real["delta"] else float("nan")
    dn = float(np.median(eat_real["delta_null"])) if eat_real["delta_null"] else float("nan")
    detectable = dm > 1.5 * dn
    print(f"realite : {eat_real['n']} repas | delta zone faim {dm:.4f} vs null {dn:.4f} "
          f"-> {'DETECTABLE' if detectable else 'NON DETECTABLE a 64px (verdict honnete)'}")
    res["faim"]["realite"] = {"n": eat_real["n"], "delta": dm, "delta_null": dn,
                              "detectable_64px": bool(detectable)}
    for name, s in eat_dream.items():
        dd = float(np.median(s["delta"])) if s["delta"] else float("nan")
        print(f"{name:8s}: delta zone faim dans les reves {dd:.4f} (n={s['n']})")
        res["faim"]["reves"][name] = {"n": s["n"], "delta": dd}

    (OUT_DIR / "results_v2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_DIR.relative_to(ROOT)}\\results_v2.json")


if __name__ == "__main__":
    main()
