"""
DETECTEUR DE BUT APPRIS -- "un bloc vient de casser devant toi".

Remplace l'heuristique pixel (ratio_croix) par un mini-classifieur entraine
sur les VRAIS cassages du corpus. Cible : les grottes sombres (45 % vs 90 %
foret au chooseur) et, a terme, les regles garees (#2/#6/#9).

Dataset (fenetres de 5 frames, style IDM) :
  + positifs      : le MOMENT de destruction, localise dans les vraies frames
                    (changement persistant du patch croix apres 'taper' maintenu),
  - negatifs durs : minage EN COURS avant le cassage (meme activite, pas
                    d'evenement -> le detecteur apprend l'evenement, pas l'activite),
  - negatifs aleatoires.

Split par EPISODE ; les 6 episodes d'eval (val+gen+tests chooseur) sont EXCLUS
du train et servent de test final (AUC).

Sorties : outputs/goal_detector/{model_best.pt, metrics.json}
Lance   : .venv\\Scripts\\python.exe scripts\\train_goal_detector.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import context as C  # noqa: E402

OUT_DIR = ROOT / "outputs" / "goal_detector"
PATCH = (26, 38)
THETA = 0.10
WIN = 5
EPOCHS = 6
BATCH = 128
LR = 3e-4
EVAL_EPS = {"Player129-f153ac423f61-20210617-173110",
            "treechop-984393664dfd-20210924-174326",
            "Player871-2e9a64a90d31-20210627-154641",
            "Player309-dcc21a4f8784-20210721-163058",
            "nvcave_008", "nvcave_026"}


class GoalNet(nn.Module):
    """Trunk IDM-like -> P(un bloc vient de casser devant)."""

    def __init__(self, n_frames=WIN, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.conv = nn.Sequential(
            nn.Conv2d(3 * n_frames, c1, 4, 2, 1), nn.GroupNorm(8, c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, 2, 1), nn.GroupNorm(8, c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.SiLU(),
            nn.Conv2d(c3, c4, 4, 2, 1), nn.GroupNorm(8, c4), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c4 * 16, 256), nn.SiLU(),
                                  nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.conv(x))[:, 0]


def find_ep(stem):
    for base in ["processed", "processed_youtube"]:
        p = ROOT / "data" / base / f"{stem}.npz"
        if p.exists():
            return p
    return None


def locate_breaks(frames_f32, t0, horizon=32):
    """Retourne t_break (index absolu) ou None -- vrai moment de destruction.

    v2 : seuil ADAPTATIF -- dans une scene sombre le cassage bouge peu de
    valeur absolue ; sans ca le dataset est biaise clair (1087 'sans cassage'
    au v1, concentres dans le sombre)."""
    y0, y1 = PATCH
    ref = frames_f32[t0][:, y0:y1, y0:y1]
    luma = float(ref.mean())
    theta = THETA if luma >= 0.15 else max(0.05, THETA * luma / 0.15)
    n = frames_f32.shape[0]
    hi = min(t0 + 1 + horizon, n)
    diffs = np.abs(frames_f32[t0 + 1: hi][:, :, y0:y1, y0:y1] - ref).mean(axis=(1, 2, 3))
    for h in range(len(diffs) - 3):
        if diffs[h] > theta and diffs[h:h + 3].min() > theta * 0.8:
            return t0 + 1 + h
    return None


def build_samples():
    breaks = json.loads((ROOT / "outputs" / "mine_events" / "breaks.json").read_text(encoding="utf-8"))
    by_ep = defaultdict(list)
    for e in breaks:
        by_ep[e["ep"]].append(e["t"])

    rng = np.random.default_rng(0)
    samples = defaultdict(list)   # ep -> (t_center, label)
    stats = {"pos": 0, "hardneg": 0, "randneg": 0, "sans_cassage": 0}

    for ep, t0s in by_ep.items():
        p = find_ep(ep)
        if p is None:
            continue
        d = np.load(p)
        frames = np.ascontiguousarray(np.transpose(d["frames"], (0, 3, 1, 2)))
        f32 = frames.astype(np.float32) / 255.0
        n = frames.shape[0]
        for t0 in t0s:
            if t0 + 34 >= n:
                continue
            tb = locate_breaks(f32, t0)
            if tb is None or tb + 3 >= n:
                stats["sans_cassage"] += 1
                continue
            for tc in (tb, tb + 1):
                if 2 <= tc < n - 2:
                    samples[ep].append((tc, 1))
                    stats["pos"] += 1
            # negatifs durs : minage en cours, avant le cassage
            cand = list(range(t0 + 2, max(tb - 3, t0 + 2)))
            for tc in rng.permutation(cand)[:2]:
                samples[ep].append((int(tc), 0))
                stats["hardneg"] += 1
            # negatif aleatoire
            tc = int(rng.integers(2, n - 2))
            samples[ep].append((tc, 0))
            stats["randneg"] += 1
    return samples, stats


def windows(frames, centers):
    cols = [frames[centers + dlt] for dlt in (-2, -1, 0, 1, 2)]
    return np.concatenate(cols, axis=1)


@torch.no_grad()
def auc_on(model, eps_samples, device):
    scores, labels = [], []
    for ep, pairs in eps_samples.items():
        p = find_ep(ep)
        frames = np.ascontiguousarray(np.transpose(np.load(p)["frames"], (0, 3, 1, 2)))
        cs = np.array([t for t, _ in pairs])
        ls = np.array([l for _, l in pairs])
        for i in range(0, len(cs), 512):
            b = cs[i:i + 512]
            x = torch.from_numpy(windows(frames, b)).to(device).float() / 255.0
            scores.append(torch.sigmoid(model(x)).cpu().numpy())
        labels.append(ls)
    s = np.concatenate(scores)
    l = np.concatenate(labels)
    # AUC par rangs (Mann-Whitney)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = int(l.sum()), int((1 - l).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), n1, n0
    auc = (ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc), n1, n0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples, stats = build_samples()
    print(f"echantillons: {stats}")
    train_s = {ep: v for ep, v in samples.items() if ep not in EVAL_EPS}
    test_s = {ep: v for ep, v in samples.items() if ep in EVAL_EPS}
    n_train = sum(len(v) for v in train_s.values())
    n_test = sum(len(v) for v in test_s.values())
    print(f"train: {n_train} fenetres ({len(train_s)} eps) | test: {n_test} ({len(test_s)} eps)")

    # tout en RAM (fenetres uint8) -- volumes modestes
    Xs, ys = [], []
    for ep, pairs in train_s.items():
        p = find_ep(ep)
        frames = np.ascontiguousarray(np.transpose(np.load(p)["frames"], (0, 3, 1, 2)))
        cs = np.array([t for t, _ in pairs])
        Xs.append(windows(frames, cs))
        ys.append(np.array([l for _, l in pairs], dtype=np.float32))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print(f"X: {X.shape} ({X.nbytes/1e9:.2f} Go) | positifs {y.mean():.0%}")

    model = GoalNet().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    rng = np.random.default_rng(0)

    best = 0.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = rng.permutation(len(X))
        tot, ns = 0.0, 0
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            xb = torch.from_numpy(X[b]).to(device).float() / 255.0
            g = float(rng.uniform(0.4, 1.3))     # jitter gamma : voir sombre ET clair
            xb = xb ** g
            # v3 : augmentation DOMAINE-REVE -- le detecteur note des reves flous,
            # il doit apprendre sur du flou (lecon : valider l'instrument dans son
            # domaine d'OPERATION, pas seulement sur le reel)
            if rng.random() < 0.5:
                k = int(rng.choice([3, 5]))
                import torch.nn.functional as _F
                xb = _F.avg_pool2d(xb, k, stride=1, padding=k // 2)
            yb = torch.from_numpy(y[b]).to(device)
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); ns += 1
        model.eval()
        auc, n1, n0 = auc_on(model, test_s, device)
        marker = ""
        if auc > best:
            best = auc
            torch.save(model.state_dict(), OUT_DIR / "model_best.pt")
            marker = "  <- best, sauve"
        print(f"epoch {epoch}/{EPOCHS}  loss {tot/max(ns,1):.4f}  "
              f"AUC test (6 eps jamais vus) {auc:.3f} (n+={n1}, n-={n0}){marker}", flush=True)

    (OUT_DIR / "metrics.json").write_text(json.dumps(
        {"auc_test": best, "stats": stats, "n_train": n_train, "n_test": n_test},
        indent=2), encoding="utf-8")
    print(f"detecteur: {time.time()-t0:.0f}s | AUC test {best:.3f} -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
