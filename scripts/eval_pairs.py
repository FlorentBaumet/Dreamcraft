"""
La mesure qui juge le reentrainement, fixee AVANT de collecter les donnees.

Sur le monde de test, jamais vu a l'entrainement, on reprend chaque paire de la sonde :
meme scene sans le bouton (B), puis avec le bouton (A). La camera ne bouge pas, donc
le label ne peut pas venir du mouvement du joueur.

  - label     : le coup a-t-il servi ? Le centre de l'image a-t-il change en 12 images ?
                Le seuil vient du monde d'entrainement (Otsu), jamais du test.
  - AUC       : avant le coup, le score jumeau du modele separe-t-il les coups utiles
                des coups dans le vide ? 50 % = il ne sait pas.
  - SEP       : sur les coups utiles, part de l'effet reel attribuee au bouton.
  - barre     : contraste du bloc et ratio sur le passe, sans aucun modele. Le modele
                n'a rien apporte s'il ne les bat pas a contraste egal.
  - exclusion : paires dont le temoin depasse le seuil (la scene bouge toute seule).

Lance :
  .venv\\Scripts\\python.exe scripts\\eval_pairs.py --ckpt avant=outputs/v13b-indomain/model_best.pt
        --ckpt apres=outputs/<run>/model_best.pt
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft.data import context as C  # noqa: E402

_s = ilu.spec_from_file_location("du", Path(__file__).parent / "diag_utilite.py")
U = ilu.module_from_spec(_s)
_s.loader.exec_module(U)

HREAL, N_BOOT = U.HREAL, 2000
HS_SEP = (2, 4, 8, 12, 16, 20, 24)
Y0, Y1 = U.PATCH
MIN_PAIRS = 50
REF_TRAIN = 8        # le seuil et le sens des baselines restent appris sur ces sessions : la
                     # mesure ne doit pas bouger quand on ajoute des donnees d'entrainement


def load_session(path):
    """Une session de sonde : ses images, ses actions et le debut de chaque paire complete."""
    fr, acts, _ = C.load_episode_u8(path)
    phase = np.load(path)["phase"]
    last = max(HREAL, max(HS_SEP))
    starts = [s for s in range(HREAL, len(phase) - last)
              if phase[s] == "A_taper" and phase[s - 1] == "B_temoin"
              and (phase[s - HREAL:s] == "B_temoin").all()
              and (phase[s:s + last] == "A_taper").all()]
    f = fr.astype(np.float32) / 255.0
    patch = lambda t: f[t][:, Y0:Y1, Y0:Y1]  # noqa: E731
    eff_a = np.array([np.abs(patch(s + HREAL) - patch(s)).mean() for s in starts])
    eff_b = np.array([np.abs(patch(s) - patch(s - HREAL)).mean() for s in starts])
    return {"name": Path(path).stem, "fr": fr, "acts": acts, "starts": starts,
            "eff_a": eff_a, "eff_b": eff_b}


def otsu(values):
    """Le seuil qui separe le mieux deux groupes, sans parametre a regler.
    En echelle log : les coups dans le vide valent ~0,002, les coups utiles de 0,02 a 0,15.
    En echelle lineaire, le seuil tomberait au milieu des coups utiles."""
    v = np.sort(np.log(values + 1e-4))
    best, thr = -1.0, float(v[len(v) // 2])
    for i in range(1, len(v)):
        w0, w1 = i / len(v), 1 - i / len(v)
        between = w0 * w1 * (v[:i].mean() - v[i:].mean()) ** 2
        if between > best:
            best, thr = between, float((v[i - 1] + v[i]) / 2)
    return float(np.exp(thr) - 1e-4)


@torch.no_grad()
def sep_curves(model, nctx, sess, device):
    """Par paire et par horizon : changement au centre, reve avec et sans le bouton."""
    fr, acts, starts = sess["fr"], sess["acts"], np.array(sess["starts"])
    tap, none = [], []
    hmax = max(HS_SEP)
    for i in range(0, len(starts), 24):
        b = starts[i:i + 24]
        ctx = C.stack_context(fr, b - nctx + 1, nctx)
        x = torch.from_numpy(np.concatenate([ctx, ctx])).to(device).float() / 255.0
        ref = x[:, -3:, Y0:Y1, Y0:Y1].clone()
        a = np.stack([acts[t:t + hmax] for t in b]).astype(np.float32)
        a = np.concatenate([a, a.copy()])
        a[len(b):, :, 6] = 0.0
        rows = []
        for k in range(hmax):
            x = C.roll_context(x, model(x, torch.from_numpy(a[:, k]).to(device)))
            if k + 1 in HS_SEP:
                rows.append((x[:, -3:, Y0:Y1, Y0:Y1] - ref).abs().mean((1, 2, 3)).cpu().numpy())
        rows = np.stack(rows, axis=1)
        tap.append(rows[:len(b)])
        none.append(rows[len(b):])
    return np.concatenate(tap), np.concatenate(none)


def real_curves(sess):
    f = sess["fr"].astype(np.float32) / 255.0
    return np.array([[np.abs(f[s + h][:, Y0:Y1, Y0:Y1] - f[s][:, Y0:Y1, Y0:Y1]).mean()
                      for h in HS_SEP] for s in sess["starts"]])


def resamples(groups, rng):
    """Indices de bootstrap, par session s'il y en a assez : les paires d'une session se ressemblent."""
    ids = np.unique(groups)
    if len(ids) >= 5:
        where = {g: np.flatnonzero(groups == g) for g in ids}
        for _ in range(N_BOOT):
            yield np.concatenate([where[g] for g in rng.choice(ids, len(ids))])
    else:
        for _ in range(N_BOOT):
            yield rng.integers(0, len(groups), len(groups))


def auc_ic(score, label, groups, rng):
    vals = [U.auc(score[i], label[i]) for i in resamples(groups, rng)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(U.auc(score, label)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="nom=chemin, repetable")
    ap.add_argument("--dir", default=str(ROOT / "data" / "processed_agent"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_pairs" / "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    folder = Path(args.dir)

    train = [load_session(p) for p in sorted(folder.glob("dc_train*.npz"))[:REF_TRAIN]]
    test = [load_session(p) for p in sorted(folder.glob("dc_test*.npz"))]
    if not test:
        sys.exit(f"aucune session dc_test dans {folder}")
    if not train:
        sys.exit("aucune session dc_train : le seuil doit venir du monde d'entrainement")
    thr = otsu(np.concatenate([s["eff_a"] for s in train]))

    # Une scene qui bouge toute seule (eau, gravier qui tombe) ne permet pas de dire
    # que le changement vient du bouton : ces paires sont ecartees.
    n_all = sum(len(s["starts"]) for s in test)
    for s in train + test:
        keep = s["eff_b"] <= thr
        s["starts"] = [t for t, k in zip(s["starts"], keep) if k]
        s["eff_a"], s["eff_b"] = s["eff_a"][keep], s["eff_b"][keep]
    print(f"paires ecartees (temoin qui bouge) : {n_all - sum(len(s['starts']) for s in test)}/{n_all}")

    eff_a = np.concatenate([s["eff_a"] for s in test])
    eff_b = np.concatenate([s["eff_b"] for s in test])
    label = eff_a > thr
    groups = np.concatenate([[i] * len(s["starts"]) for i, s in enumerate(test)])
    print(f"test : {len(test)} sessions, {len(label)} paires | seuil {thr:.3f} (appris sur dc_train)")
    print(f"coups utiles : {label.mean():.0%} | changement median au centre : "
          f"avec le bouton {np.median(eff_a):.3f}, temoin {np.median(eff_b):.3f}")
    if len(label) < MIN_PAIRS or min(label.sum(), (~label).sum()) < 10:
        print("ATTENTION : trop peu de paires ou de cas de chaque sorte pour conclure.")

    tex = np.concatenate([U.b0_texture(s["fr"], s["starts"]) for s in test])
    scores = {}
    # Une baseline peut predire a l'envers (un bloc proche parait lisse). Son sens
    # s'apprend sur dc_train, jamais sur le test : c'est elle, la vraie barre.
    train_label = np.concatenate([s["eff_a"] for s in train]) > thr
    for name, fn in [("contraste du bloc (sans modele)", U.b0_texture),
                     ("ratio sur le passe (sans modele)", U.b0_passe)]:
        on_train = np.concatenate([fn(s["fr"], s["starts"]) for s in train])
        sign = 1.0 if U.auc(on_train, train_label) >= 0.5 else -1.0
        print(f"{name} : AUC sur dc_train {U.auc(sign * on_train, train_label):.1%}, sens {sign:+.0f}")
        scores[name] = sign * np.concatenate([fn(s["fr"], s["starts"]) for s in test])
    real = np.concatenate([real_curves(s) for s in test])
    seps, raw = {}, {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m, nctx = U.load_model(path, device)
        scores[f"{name} : jumeau"] = np.concatenate(
            [U.twin_scores(m, nctx, s["fr"], s["acts"], s["starts"], device) for s in test])
        curves = [sep_curves(m, nctx, s, device) for s in test]
        tap_all = np.concatenate([c[0] for c in curves])
        none_all = np.concatenate([c[1] for c in curves])
        tap, none = tap_all[label], none_all[label]
        denom = real[label].mean(0) - none.mean(0)
        # sans effet reel a cet horizon (le bloc n'a pas encore casse), le rapport n'a pas de sens
        seps[name] = np.where(denom > 1e-3, (tap.mean(0) - none.mean(0)) / np.maximum(denom, 1e-3),
                              np.nan).tolist()
        raw[name] = {"coup utile, reel": real[label].mean(0).tolist(),
                     "coup utile, reve avec bouton": tap.mean(0).tolist(),
                     "coup utile, reve sans bouton": none.mean(0).tolist(),
                     "coup dans le vide, reel": real[~label].mean(0).tolist(),
                     "coup dans le vide, reve avec bouton": tap_all[~label].mean(0).tolist(),
                     "coup dans le vide, reve sans bouton": none_all[~label].mean(0).tolist()}

    out = {"seuil": thr, "n_paires": int(len(label)), "n_sessions": len(test),
           "coups_utiles": float(label.mean()), "auc": {}, "sep": seps, "effets": raw,
           "horizons_sep": HS_SEP}
    print(f"\n{'score':<36}{'AUC [IC 95 %]':<26}{'a contraste egal'}")
    for name, s in scores.items():
        a, lo, hi = auc_ic(s, label, groups, rng)
        strat = float("nan") if name.startswith("contraste") else U.auc_strat(s, label, tex)
        print(f"{name:<36}{f'{a:.1%} [{lo:.1%}, {hi:.1%}]':<26}{strat:.1%}")
        out["auc"][name] = {"auc": a, "lo": lo, "hi": hi, "strat": strat}
    if seps:
        print(f"\nSEP sur les coups utiles{'':<12}" + "".join(f"H={h:<6}" for h in HS_SEP))
        for name, row in seps.items():
            print(f"{name:<36}" + "".join(f"{v:<8.0%}" for v in row))
        print("\nChangement au centre de l'image, par horizon")
        for name, table in raw.items():
            print(f"  {name}")
            for what, row in table.items():
                print(f"    {what:<36}" + "".join(f"{v:<8.3f}" for v in row))

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
