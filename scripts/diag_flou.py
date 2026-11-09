"""
La portee honnete resiste-t-elle a une baseline floue ?

En MSE, predire une image floue coute moins cher que predire une image nette
mais decalee (double penalite). Or notre portee compare le reve a la frame
gelee NETTE. Un reve qui ne fait que flouter pourrait donc "battre" la copie
sans rien avoir appris de la dynamique.

Test : memes departs, meme masque Delta-region que rollout_phase1.py, et on
ajoute des baselines sans modele : frame gelee floutee a sigma fixe.
On mesure aussi la nettete du reve : part d'energie spectrale en haute frequence,
rapportee a celle de la realite (1 = aussi net que le reel).

Lance : .venv\\Scripts\\python.exe scripts\\diag_flou.py
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
from dreamcraft.eval import metrics as M  # noqa: E402
from dreamcraft.models.conv_predictor import ConvPredictor  # noqa: E402

H, N_STARTS, N_BOOT = 96, 48, 1000
SIGMAS = [1.0, 2.0, 4.0, 8.0]
SHOW = [1, 4, 8, 16, 32, 64, 96]
MODELS = {"v4": ROOT / "outputs" / "v4-k16-ema" / "model_best.pt",
          "v7b": ROOT / "outputs" / "v7b-commitloss" / "model_best.pt"}
# Player129 sert aussi a selectionner les checkpoints ; les 3 autres jamais.
EPISODES = ["Player129-f153ac423f61-20210617-173110",
            "treechop-984393664dfd-20210924-174326",
            "Player871-2e9a64a90d31-20210627-154641",
            "Player309-dcc21a4f8784-20210721-163058"]


def load_model(ckpt, device):
    cfg = {"base": 32, "in_frames": 1}
    cf = Path(ckpt).parent / "config.json"
    if cf.exists():
        cfg.update(json.loads(cf.read_text(encoding="utf-8")))
    m = ConvPredictor(base=cfg["base"], in_frames=cfg["in_frames"]).to(device)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, cfg["in_frames"]


def blur(img_chw, sigma):
    return cv2.GaussianBlur(img_chw.transpose(1, 2, 0), (0, 0), sigma).transpose(2, 0, 1)


def hf_energy(batch_chw):
    """Part de l'energie au-dessus de 0.25 cycle/pixel, par image."""
    g = batch_chw.mean(1)
    g = g - g.mean((1, 2), keepdims=True)
    p = np.abs(np.fft.rfft2(g)) ** 2
    fy = np.fft.fftfreq(g.shape[1])[:, None]
    fx = np.fft.rfftfreq(g.shape[2])[None, :]
    hf = np.sqrt(fx ** 2 + fy ** 2) > 0.25
    return p[:, hf].sum(1) / (p.sum((1, 2)) + 1e-9)


def portee(ref, base):
    """Dernier pas ou la moyenne du modele reste sous celle de la baseline."""
    cross = next((h for h in range(len(ref)) if ref[h] > base[h]), None)
    return len(ref) if cross is None else cross


def marges(curves, fenetres=((8, 24), (48, 80))):
    """Moyenne sur une fenetre d'horizons, moins bruitee que le premier croisement.
    Le meilleur flou est choisi apres coup, ce qui avantage la baseline."""
    lignes = []
    for a, b in fenetres:
        w = slice(a - 1, b)
        r, g = np.mean(curves["reve"][w]), np.mean(curves["gelee"][w])
        f = min(np.mean(curves[f"flou{s:g}"][w]) for s in SIGMAS)
        lignes.append(f"h={a}-{b} : reve {r:.4f}  gelee {g:.4f}  meilleur flou {f:.4f}  "
                      f"part du gain captee par le flou {(g - f) / (g - r):.0%}  "
                      f"marge du reve {(f - r) / f:+.0%}")
    return lignes


@torch.no_grad()
def run_episode(stem, model, nctx, device):
    fr_u8, actions, ok = C.load_episode_u8(ROOT / "data" / "processed" / f"{stem}.npz")
    frames = fr_u8.astype(np.float32) / 255.0
    pad = C.EVAL_CONTEXT_PAD
    cand = C.valid_starts(ok, pad, H)
    starts = cand[np.linspace(0, len(cand) - 1, min(N_STARTS, len(cand))).astype(int)]
    t = starts + pad - 1
    x = torch.from_numpy(C.stack_context(fr_u8, t - nctx + 1, nctx)).to(device).float() / 255.0
    frozen = frames[t]
    blurred = {s: np.stack([blur(f, s) for f in frozen]) for s in SIGMAS}
    keys = ["reve", "gelee"] + [f"flou{s:g}" for s in SIGMAS]
    err = {k: np.full((len(t), H), np.nan) for k in keys}
    sharp = np.zeros((len(t), H))
    for h in range(H):
        pred = model(x, torch.from_numpy(actions[t + h].astype(np.float32)).to(device))
        p = pred.cpu().numpy()
        real, prev = frames[t + h + 1], frames[t + h]
        chg = np.abs(real - prev).max(axis=1) > M.CHANGE_THRESHOLD
        sharp[:, h] = hf_energy(p) / (hf_energy(real) + 1e-9)
        for s in range(len(t)):
            m = chg[s]
            if int(m.sum()) < M.MIN_CHANGED_PIXELS:
                continue
            err["reve"][s, h] = ((p[s] - real[s]) ** 2)[:, m].mean()
            err["gelee"][s, h] = ((frozen[s] - real[s]) ** 2)[:, m].mean()
            for sg in SIGMAS:
                err[f"flou{sg:g}"][s, h] = ((blurred[sg][s] - real[s]) ** 2)[:, m].mean()
        x = C.roll_context(x, pred)
    return err, sharp


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    out = {}
    for mname, ckpt in MODELS.items():
        model, nctx = load_model(ckpt, device)
        print(f"\n===== {mname} (contexte {nctx}, H={H}, {N_STARTS} departs par episode) =====")
        for stem in EPISODES:
            err, sharp = run_episode(stem, model, nctx, device)
            mean = {k: np.nanmean(v, axis=0) for k, v in err.items()}
            print(f"\n{stem[:24]}")
            print(f"  {'':<8}" + "".join(f"h={h:<7}" for h in SHOW))
            for k, v in mean.items():
                print(f"  {k:<8}" + "".join(f"{v[h - 1]:<9.4f}" for h in SHOW))
            print(f"  {'nettete':<8}" + "".join(f"{np.nanmean(sharp[:, h - 1]):<9.2f}" for h in SHOW)
                  + "  (energie HF reve / reel)")
            line = "  portee du reve contre :"
            out_ep = {}
            for k in mean:
                if k == "reve":
                    continue
                pt = portee(mean["reve"], mean[k])
                out_ep[k] = pt
                # IC bootstrap sur les departs
                n = err["reve"].shape[0]
                boot = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, n, n)
                    boot.append(portee(np.nanmean(err["reve"][i], axis=0),
                                       np.nanmean(err[k][i], axis=0)))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                line += f"  {k} {pt} [{lo:.0f}-{hi:.0f}]"
            print(line)
            for ligne in marges(mean):
                print("  " + ligne)
            out[f"{mname}/{stem}"] = {"portee": out_ep,
                                      "curves": {k: v.tolist() for k, v in mean.items()},
                                      "nettete": np.nanmean(sharp, axis=0).tolist()}
    d = ROOT / "outputs" / "diag"
    d.mkdir(parents=True, exist_ok=True)
    (d / "flou.json").write_text(json.dumps(out), encoding="utf-8")
    print("\n-> outputs/diag/flou.json")


if __name__ == "__main__":
    main()
