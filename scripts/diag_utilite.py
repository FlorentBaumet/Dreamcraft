"""
La decision de l'agent : "taper ICI, ca sert a quelque chose ?"

C est la seule question que l agent doit trancher a chaque instant. On la teste
sur des episodes JAMAIS VUS, sans aucune annotation humaine : la realite tranche
toute seule.

  - on prend chaque coup de pioche reel du joueur,
  - la REALITE dit s il etait PRODUCTIF (le bloc vise bouge dans les 12 frames)
    ou STERILE (rien ne bouge : on tape dans le vide, ou trop peu longtemps),
  - on demande au modele, AVANT, ce que vaut le bouton taper a cet instant
    (score jumeau : reve avec le bouton moins reve du meme plan sans le bouton),
  - AUC = probabilite qu il donne un score plus haut a un coup productif qu a un
    coup sterile. 50 % = pile ou face. 100 % = voyant parfait.

C est la metrique cible de la boucle agent : les donnees collectees en jouant
doivent la faire monter.

Lance : .venv\Scripts\python.exe scripts\diag_utilite.py
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

HDREAM, HREAL, PATCH, MAXN = 8, 12, (26, 38), 250
MODELS = {"v6 timide": ROOT / "outputs" / "v6-nvcave" / "model_best.pt",
          "v7b commit-loss": ROOT / "outputs" / "v7b-commitloss" / "model_best.pt",
          "v13b courageux": ROOT / "outputs" / "v13b-indomain" / "model_best.pt"}
PANELS = [("strip (in-domain)", ["strip_050", "strip_051", "strip_052", "strip_053"]),
          ("nvcave (out-domain)", ["nvcave_008", "nvcave_026"])]


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


def auc(scores, labels):
    """Probabilite qu un positif soit mieux note qu un negatif (Mann-Whitney)."""
    o = np.argsort(scores, kind="mergesort")
    r = np.empty(len(scores)); r[o] = np.arange(1, len(scores) + 1)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg)


@torch.no_grad()
def twin_scores(model, nctx, fr_u8, acts, starts, device):
    """Score jumeau : effet propre du bouton taper sur le bloc vise, a HDREAM pas."""
    y0, y1 = PATCH
    out = []
    for i in range(0, len(starts), 24):
        b = np.array(starts[i:i + 24])
        ctx = C.stack_context(fr_u8, b - nctx + 1, nctx)
        x = torch.from_numpy(np.concatenate([ctx, ctx])).to(device).float() / 255.0
        ref = x[:, -3:, y0:y1, y0:y1].clone()
        a = np.stack([acts[t:t + HDREAM] for t in b]).astype(np.float32)
        a = np.concatenate([a, a.copy()])
        a[len(b):, :, 6] = 0.0                       # le jumeau ne tape pas
        a[len(b):, :, 7] = 0.0
        for k in range(HDREAM):
            x = C.roll_context(x, model(x, torch.from_numpy(a[:, k]).to(device)))
        d = (x[:, -3:, y0:y1, y0:y1] - ref).abs().mean((1, 2, 3)).cpu().numpy()
        out.append(d[:len(b)] - d[len(b):])
    return np.concatenate(out)


@torch.no_grad()
def ratio_scores(model, nctx, fr_u8, acts, starts, device):
    """Baseline : le score ratio_croix utilise jusqu ici."""
    y0, y1 = PATCH
    cm = torch.zeros(64, 64, dtype=torch.bool, device=device); cm[y0:y1, y0:y1] = True
    out = []
    for i in range(0, len(starts), 24):
        b = np.array(starts[i:i + 24])
        x = torch.from_numpy(C.stack_context(fr_u8, b - nctx + 1, nctx)).to(device).float() / 255.
        ref = x[:, -3:].clone()
        a = np.stack([acts[t:t + HDREAM] for t in b]).astype(np.float32)
        for k in range(HDREAM):
            x = C.roll_context(x, model(x, torch.from_numpy(a[:, k]).to(device)))
        d = (x[:, -3:] - ref).abs().mean(1)
        c, p = d[:, cm].mean(1), d[:, ~cm].mean(1)
        out.append((c / (c + p + 1e-6)).cpu().numpy())
    return np.concatenate(out)


def auc_strat(scores, labels, strat, k=3):
    """AUC A TEXTURE COMPARABLE : on compare des coups dont le bloc se ressemble.
    Neutralise le biais 'un patch texture change forcement plus de pixels'."""
    qs = np.quantile(strat, np.linspace(0, 1, k + 1)[1:-1])
    bins = np.digitize(strat, qs)
    vals, ws = [], []
    for b in range(k):
        m = bins == b
        if m.sum() < 20 or labels[m].sum() in (0, m.sum()):
            continue
        vals.append(auc(scores[m], labels[m])); ws.append(m.sum())
    return float(np.average(vals, weights=ws)) if vals else float("nan")


def b0_passe(fr_u8, starts):
    """SANS MODELE : meme ratio, mais sur les images DEJA VUES (t-4 -> t)."""
    y0, y1 = PATCH
    f = fr_u8.astype(np.float32) / 255.
    out = []
    for t in starts:
        d = np.abs(f[t] - f[t - 4]).mean(0)
        c = d[y0:y1, y0:y1].mean()
        tot = d.mean()
        out.append(c / (tot + 1e-6))
    return np.array(out)


def b0_texture(fr_u8, starts):
    """SANS MODELE : contraste local du bloc vise sur la seule image courante."""
    y0, y1 = PATCH
    f = fr_u8.astype(np.float32) / 255.
    return np.array([f[t][:, y0:y1, y0:y1].std() for t in starts])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {n: load_model(p, device) for n, p in MODELS.items() if p.exists()}
    y0, y1 = PATCH
    data = {}
    for title, stems in PANELS:
        starts, labels, eps = [], [], []
        for stem in stems:
            try:
                fr, acts, ok = load_ep(stem)
            except FileNotFoundError:
                continue
            tap = acts[:, 6] > 0.5
            ev = [t for t in range(8, len(acts) - max(HREAL, HDREAM) - 2)
                  if tap[t] and not tap[t - 1] and ok[t - 8:t + HREAL + 1].all()]
            if len(ev) > MAXN:
                ev = [ev[i] for i in np.linspace(0, len(ev) - 1, MAXN).astype(int)]
            if not ev:
                continue
            f = fr.astype(np.float32) / 255.
            eff = np.array([np.abs(f[t + HREAL][:, y0:y1, y0:y1]
                                   - f[t][:, y0:y1, y0:y1]).mean() for t in ev])
            eps.append((stem, fr, acts, ev, eff))
            starts += ev; labels.append(eff)
        allf = np.concatenate(labels)
        thr = np.median(allf)
        data[title] = (eps, thr)
        print(f"{title:<22} {len(allf)} coups de pioche | seuil productif = {thr:.3f}")

    print()
    print(f"{'modele':<18}" + "".join(f"{t:>34}" for t, _ in PANELS))
    print(f"{'':<18}" + "".join(f"{'AUC jumeau / AUC ratio (base)':>34}" for _ in PANELS))
    print("-" * (18 + 34 * len(PANELS)))
    for name, (m, nctx) in loaded.items():
        line = f"{name:<18}"
        for title, _ in PANELS:
            eps, thr = data[title]
            sj, sr, lb = [], [], []
            for stem, fr, acts, ev, eff in eps:
                sj.append(twin_scores(m, nctx, fr, acts, ev, device))
                sr.append(ratio_scores(m, nctx, fr, acts, ev, device))
                lb.append(eff > thr)
            sj, sr, lb = np.concatenate(sj), np.concatenate(sr), np.concatenate(lb)
            line += f"{auc(sj, lb):.1%}  /  {auc(sr, lb):.1%}".rjust(34)
        print(line)
    print()
    print("=== A TEXTURE COMPARABLE (AUC stratifiee par contraste du bloc) ===")
    for title, _ in PANELS:
        eps, thr = data[title]
        lb = np.concatenate([eff > thr for *_, eff in eps])
        tx = np.concatenate([b0_texture(fr, ev) for _, fr, _, ev, _ in eps])
        rows = [("ratio sur le passe (sans modele)",
                 np.concatenate([b0_passe(fr, ev) for _, fr, _, ev, _ in eps]))]
        for name, (m, nctx) in loaded.items():
            rows.append((f"reve {name} : ratio",
                         np.concatenate([ratio_scores(m, nctx, fr, acts, ev, device)
                                         for _, fr, acts, ev, _ in eps])))
            rows.append((f"reve {name} : jumeau",
                         np.concatenate([twin_scores(m, nctx, fr, acts, ev, device)
                                         for _, fr, acts, ev, _ in eps])))
        print(f"  {title}")
        for nm, sc in rows:
            print(f"    {nm:<36} AUC brute {auc(sc, lb):.1%}   stratifiee {auc_strat(sc, lb, tx):.1%}")
    print()
    print("--- baselines SANS AUCUN MODELE (la barre a battre) ---")
    for title, _ in PANELS:
        eps, thr = data[title]
        for nm, fn in [("ratio sur le passe", b0_passe), ("contraste du bloc", b0_texture)]:
            sc = np.concatenate([fn(fr, ev) for _, fr, _, ev, _ in eps])
            lb = np.concatenate([eff > thr for *_, eff in eps])
            print(f"  {title:<22} {nm:<20} AUC {auc(sc, lb):.1%}")
    print()
    print("50 % = le modele ne sait pas. Au-dessus, il voit venir l utilite du coup.")
    print("si une baseline sans modele egale le reve, le reve n apporte rien ICI.")


if __name__ == "__main__":
    main()
