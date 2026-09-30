"""
Le reve reagit-il au fait d'appuyer sur taper ?

Aucun score, aucune heuristique. Au moment ou le joueur reel donne son premier
coup de pioche, on fait rever le modele DEUX FOIS depuis le meme contexte :
   - une fois en tapant,
   - une fois en ne faisant rien.
Puis on regarde le bloc vise.

  reel        = ce qui se passe vraiment (le bloc casse)
  reve taper  = ce que le modele imagine s'il tape
  reve rien   = ce qu'il imagine s'il ne fait rien
  SEPARATION  = (reve taper - reve rien) / (reel - reve rien)

SEPARATION = 1 -> le modele attribue au bouton taper tout l'effet reel : un
chooseur peut decider. SEPARATION = 0 -> le bouton ne change rien dans le reve :
AUCUN score ne peut faire un choix. C'est le plafond dur du Monde 2.

Lance : .venv\Scripts\python.exe scripts\diag_separation.py
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

H, PATCH, MAXN = 16, (26, 38), 90
HS = [2, 4, 6, 8, 12, 16, 20, 24]
MODELS = {"v6 timide": ROOT / "outputs" / "v6-nvcave" / "model_best.pt",
          "v7b commit-loss": ROOT / "outputs" / "v7b-commitloss" / "model_best.pt",
          "v13b courageux": ROOT / "outputs" / "v13b-indomain" / "model_best.pt"}
PANELS = [("strip (dense, in-domain)", ["strip_051", "strip_052"]),
          ("nvcave (rare, out-domain)", ["nvcave_008", "nvcave_026"])]


def load_ep(stem):
    for base in ["processed", "processed_youtube"]:
        p = ROOT / "data" / base / f"{stem}.npz"
        if p.exists():
            return C.load_episode_u8(p)
    raise FileNotFoundError(stem)


def load_model(ckpt, device):
    cfg = {"base": 32, "in_frames": 1}
    cf = Path(ckpt).parent / "config.json"
    if cf.exists():
        cfg.update(json.loads(cf.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


@torch.no_grad()
def dream_center(model, nctx, fr_u8, acts, starts, device, force, h=None):
    """Changement moyen du bloc vise apres H pas, en forcant le bit taper a `force`."""
    y0, y1 = PATCH
    h = h or H
    out = []
    for i in range(0, len(starts), 24):
        b = np.array(starts[i:i + 24])
        x = torch.from_numpy(C.stack_context(fr_u8, b - nctx + 1, nctx)).to(device).float() / 255.
        ref = x[:, -3:, y0:y1, y0:y1].clone()
        a = np.stack([acts[t:t + h] for t in b]).astype(np.float32).copy()
        a[:, :, 6] = force                              # 1 = je tape, 0 = je ne tape pas
        if force == 0.0:
            a[:, :, 7] = 0.0                            # ni utiliser
        for k in range(h):
            pred = model(x, torch.from_numpy(a[:, k]).to(device))
            x = C.roll_context(x, pred)
        out.append((x[:, -3:, y0:y1, y0:y1] - ref).abs().mean((1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def real_center(fr_u8, starts, h=None):
    y0, y1 = PATCH
    h = h or H
    f = fr_u8.astype(np.float32) / 255.
    return np.array([np.abs(f[t + h][:, y0:y1, y0:y1] - f[t][:, y0:y1, y0:y1]).mean()
                     for t in starts])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {n: load_model(p, device) for n, p in MODELS.items() if p.exists()}
    print(f"{'':<18}" + "".join(f"{t.split(' (')[0]:>26}" for t, _ in PANELS))
    print(f"{'modele':<18}" + "".join(f"{'taper / rien / reel  SEP':>26}" for _ in PANELS))
    print("-" * (18 + 26 * len(PANELS)))
    res = {}
    for name, (m, nctx) in loaded.items():
        line = f"{name:<18}"
        for title, stems in PANELS:
            T, R, V = [], [], []
            for stem in stems:
                fr, acts, ok = load_ep(stem)
                tap = acts[:, 6] > 0.5
                ev = [t for t in range(8, len(acts) - H - 2)
                      if tap[t] and not tap[t - 1] and tap[t:t + 6].all() and ok[t - 8:t + H + 1].all()]
                if len(ev) > MAXN:
                    ev = [ev[i] for i in np.linspace(0, len(ev) - 1, MAXN).astype(int)]
                if not ev:
                    continue
                T.append(dream_center(m, nctx, fr, acts, ev, device, 1.0))
                R.append(dream_center(m, nctx, fr, acts, ev, device, 0.0))
                V.append(real_center(fr, ev))
            t, r, v = (np.concatenate(z).mean() for z in (T, R, V))
            sep = (t - r) / max(v - r, 1e-6)
            res[(name, title)] = sep
            line += f"{t:.3f} /{r:.3f} /{v:.3f}  {sep:+.0%}".rjust(26)
        print(line)
    print("\nSEP = part de l'effet reel que le modele attribue au bouton taper.")
    print("0 % -> le bouton n'a aucun effet dans le reve : le chooseur est aveugle,")
    print("       et c'est LA le plafond que l'agent doit faire sauter.")


def sweep():
    """La PORTEE optimale : a quel horizon de reve l'action est-elle la plus lisible ?"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {n: load_model(p, device) for n, p in MODELS.items()
              if p.exists() and n != "v7b commit-loss"}
    hmax = max(HS)
    cache = {}
    for title, stems in PANELS:
        for stem in stems:
            fr, acts, ok = load_ep(stem)
            tap = acts[:, 6] > 0.5
            ev = [t for t in range(8, len(acts) - hmax - 2)
                  if tap[t] and not tap[t - 1] and tap[t:t + 6].all()
                  and ok[t - 8:t + hmax + 1].all()]
            if len(ev) > MAXN:
                ev = [ev[i] for i in np.linspace(0, len(ev) - 1, MAXN).astype(int)]
            cache[stem] = (fr, acts, ev)
    print()
    print("PORTEE H : part de l effet reel attribuee au bouton taper (SEP)")
    print(f"{'modele / domaine':<28}" + "".join(f"H={h:<5}" for h in HS))
    print("-" * (28 + 7 * len(HS)))
    res = {}
    for name, (m, nctx) in loaded.items():
        for title, stems in PANELS:
            key = f"{name} / {title.split(' (')[0]}"
            res[key] = []
            line = f"{name + ' / ' + title.split(' (')[0]:<28}"
            for h in HS:
                T, R, V = [], [], []
                for stem in stems:
                    fr, acts, ev = cache[stem]
                    if not ev:
                        continue
                    T.append(dream_center(m, nctx, fr, acts, ev, device, 1.0, h))
                    R.append(dream_center(m, nctx, fr, acts, ev, device, 0.0, h))
                    V.append(real_center(fr, ev, h))
                t, r, v = (np.concatenate(z).mean() for z in (T, R, V))
                sep = (t - r) / max(v - r, 1e-6)
                res[key].append(float(sep))
                line += f"{sep:>5.0%}  "
            print(line)
    print()
    print("un pic = la portee ou l evenement tient dans le reve sans derive")
    out = ROOT / "outputs" / "diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / "portee_sep.json").write_text(json.dumps({"H": HS, "sep": res}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    styles = {"v6 timide / strip": ("tab:gray", "--"),
              "v13b courageux / strip": ("tab:green", "-"),
              "v6 timide / nvcave": ("tab:gray", ":"),
              "v13b courageux / nvcave": ("tab:olive", ":")}
    for k, v in res.items():
        c, ls = styles.get(k, ("k", "-"))
        ax.plot(HS, [100 * x for x in v], color=c, ls=ls, lw=2.4, marker="o", ms=4, label=k)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(min(HS), 12, color="tab:green", alpha=0.07)
    ax.text(3, -30, "fenetre utilisable H <= 12", fontsize=9, color="tab:green")
    ax.set_ylim(-40, 140)
    ax.set_xlabel("portee du reve H (pas a 20 Hz)")
    ax.set_ylabel("part de l effet reel attribuee au bouton (%)")
    ax.set_title("En domaine connu (strip), plus le reve est long, plus l effet de l action se dilue\n"
                 "hors domaine (nvcave) il reste faible et plat  |  episodes jamais vus", fontsize=10.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figure_portee_action.png", dpi=130)
    print("-> outputs/figure_portee_action.png")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    else:
        main()
