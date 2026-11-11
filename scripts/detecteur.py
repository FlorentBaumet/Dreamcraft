"""
Detecteur « taper maintenant casse un bloc en 0,6 s » : une image en entree, la verite du jeu comme label.

  - entrainement : sessions dc_train annotees, une sur cinq gardee pour la validation (regle fixe) ;
  - mesure finale : sessions vt_test (monde jamais vu), lues une seule fois ;
  - barre sans modele : contraste du bloc vise, sens appris sur l'entrainement.

Regle fixee avant de voir vt_test : le detecteur reussit si l'ecart d'AUC avec le contraste
a un intervalle de confiance a 95 % entierement au-dessus de 0. Programme fixe, aucune selection
d'epoque ni de reglage sur la validation ou le test.

Lance : .venv\\Scripts\\python.exe scripts\\detecteur.py

En foret (un tronc casse-t-il en 12 ticks ?), memes regles, autres donnees :
  DC_LABEL=tronc_12 DC_TRAIN="dc_foret_probe*" DC_TEST="vt_foret*" DC_DET_OUT=outputs/detecteur_bois
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
_s = ilu.spec_from_file_location("av", Path(__file__).parent / "analyse_verite.py")
AV = ilu.module_from_spec(_s)
_s.loader.exec_module(AV)
E, U = AV.E, AV.E.U

DATA = ROOT / "data" / "processed_agent"
OUT = ROOT / os.environ.get("DC_DET_OUT", "outputs/detecteur")
LABEL = os.environ.get("DC_LABEL", "casse_12")
TRAIN, TEST = os.environ.get("DC_TRAIN", "dc_train*"), os.environ.get("DC_TEST", "vt_test*")
EPOCHS, BATCH, SEED, N_BOOT = 15, 128, 0, 2000


def masque(x):
    """Encart du tutoriel en haut a droite sur les sessions de la nuit du 06/11 : on le cache partout."""
    x[..., :10, 36:] = 0
    return x


def charge(paths):
    X, y, g, tex = [], [], [], []
    for k, p in enumerate(paths):
        cs = [c for c in AV.coups(p) if c["ticks_ok"]]
        if not cs:
            continue
        fr = E.C.load_episode_u8(p)[0]
        t = [c["t"] for c in cs]
        X.append(masque(fr[t].copy()))
        y += [c[LABEL] for c in cs]
        g += [k] * len(cs)
        tex.append(U.b0_texture(fr, t))
    if not X:                                    # pas encore de sessions de test : ensemble vide
        return np.zeros((0, 3, 64, 64), np.uint8), np.array([]), np.array([]), np.array([])
    return np.concatenate(X), np.array(y), np.array(g), np.concatenate(tex)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # pas de moyenne globale a la fin : la position du bloc au centre compte
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def predit(m, X, device):
    m.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(m(torch.from_numpy(X[i:i + 512]).float().div(255).to(device)))
                               .cpu().numpy() for i in range(0, len(X), 512)])


def ic_ecart(a, b, y, g, rng):
    """Ecart d'AUC a - b, bootstrap par session : les coups d'une session se ressemblent."""
    ids = np.unique(g)
    where = {i: np.flatnonzero(g == i) for i in ids}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([where[i] for i in rng.choice(ids, len(ids))])
        vals.append(U.auc(a[idx], y[idx]) - U.auc(b[idx], y[idx]))
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    train_files = [p for p in sorted(DATA.glob(f"{TRAIN}.npz")) if "truth" in np.load(p).files]
    val_files = train_files[4::5]
    fit_files = [p for p in train_files if p not in val_files]
    test_files = sorted(DATA.glob(f"{TEST}.npz"))
    t0 = time.time()
    fit, val, test = charge(fit_files), charge(val_files), charge(test_files)
    print(f"coups : entrainement {len(fit[1])} ({len(fit_files)} sessions), validation {len(val[1])} "
          f"({len(val_files)}), test {len(test[1])} ({len(test_files)}) | chargement {time.time() - t0:.0f} s")
    for nom, d in (("entrainement", fit), ("validation", val), ("test", test)):
        print(f"  {nom} : {d[1].mean():.0%} de coups positifs ({LABEL})")

    sign = 1.0 if U.auc(fit[3], fit[1]) >= 0.5 else -1.0           # sens du contraste, appris ici seulement
    m = Net().to(device)
    if os.environ.get("DC_NOTER_SEUL"):              # le modele deja sauvegarde, note sur un test arrive apres
        m.load_state_dict(torch.load(OUT / "model.pt", map_location=device, weights_only=True))
        print(f"modele recharge : {OUT / 'model.pt'} (pas d'entrainement)")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    X, y = torch.from_numpy(fit[0]), torch.from_numpy(fit[1].astype(np.float32))
    for ep in range(0 if os.environ.get("DC_NOTER_SEUL") else EPOCHS):
        m.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), BATCH):
            j = perm[i:i + BATCH]
            loss = loss_fn(m(X[j].float().div(255).to(device)), y[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"epoque {ep + 1:2d} : perte {tot / len(X):.3f} | AUC validation "
              f"{U.auc(predit(m, val[0], device), val[1]):.1%}", flush=True)
    if not os.environ.get("DC_NOTER_SEUL"):
        torch.save(m.state_dict(), OUT / "model.pt")

    res = {"epoques": EPOCHS, "sens_contraste": sign}
    for nom, d in (("validation", val), ("test", test)):
        if not len(d[1]):
            print(f"{nom} : aucune session")
            continue
        p, c = predit(m, d[0], device), sign * d[3]
        lo, hi = ic_ecart(p, c, d[1], d[2], rng)
        res[nom] = {"coups": int(len(d[1])), "sessions": int(len(np.unique(d[2]))),
                    "taux_casse": float(d[1].mean()), "auc_detecteur": float(U.auc(p, d[1])),
                    "auc_contraste": float(U.auc(c, d[1])), "ecart_ic95": [lo, hi],
                    "justesse_seuil_05": float(((p >= 0.5) == d[1]).mean()),
                    "justesse_toujours_oui": float(d[1].mean()), "reussi": bool(lo > 0)}
        r = res[nom]
        print(f"\n{nom} : {r['coups']} coups, {r['sessions']} sessions\n"
              f"  AUC detecteur {r['auc_detecteur']:.1%} | contraste {r['auc_contraste']:.1%} | "
              f"ecart IC 95 % [{lo:+.1%} ; {hi:+.1%}] -> {'REUSSI' if r['reussi'] else 'pas prouve'}\n"
              f"  justesse a 0,5 : {r['justesse_seuil_05']:.1%} (toujours oui : {r['justesse_toujours_oui']:.1%})")
    (OUT / "resultats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {OUT / 'resultats.json'}")


if __name__ == "__main__":
    main()
