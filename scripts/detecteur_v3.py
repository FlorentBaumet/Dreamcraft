"""
Detecteur v3 : le meme reseau, reentraine avec les coups que l'agent a lui-meme tapes en arene.

Les parties de l'agent contiennent ses erreurs : des coups dans le vide au fond des tunnels qu'il
creuse, que la sonde n'a jamais produits. Chaque engagement de frappe (12 ticks) devient un exemple,
etiquete par la verite du jeu : un bloc casse-t-il en 12 ticks ou moins ?

Programme fixe (memes epoques, meme graine que detecteur.py), aucun reglage sur les mesures.
Mesures : validation de la sonde, vt_test, et une session d'agent sur cinq gardee de cote.
Le vrai juge reste l'arene suivante, sur des manches neuves.

Lance : .venv\\Scripts\\python.exe scripts\\detecteur_v3.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_s = ilu.spec_from_file_location("detecteur", Path(__file__).parent / "detecteur.py")
D = ilu.module_from_spec(_s)
_s.loader.exec_module(D)

OUT = Path(os.environ.get("DC_OUT", D.ROOT / "outputs" / "detecteur_v3"))
ENGAGE = 12
AGENTS = ("verifie", "detecteur")


def coups_agent(path):
    """Debuts d'engagement complets dans les suites de "taper" (approcher = en marchant : ecarte)."""
    z = np.load(path)
    tr, ph = z["truth"], z["phase"]
    out, t, n = [], 0, len(ph)
    while t < n:
        if ph[t] != "taper":
            t += 1
            continue
        fin = t
        while fin < n and ph[fin] == "taper":
            fin += 1
        for s in range(t, fin - ENGAGE + 1, ENGAGE):
            w = tr[s:s + ENGAGE + 1]
            if not (w[:, 0] >= 0).all():
                continue
            c = np.flatnonzero(w[:, 3] > 0)
            out.append({"t": s, "casse_12": bool(len(c) and w[c[0], 0] - w[0, 0] <= ENGAGE),
                        "portee": bool(tr[s, 1] > 0 and tr[s, 2] <= 45)})
        t = fin
    return out


def charge_agent(paths):
    X, y, g, portee = [], [], [], []
    for k, p in enumerate(paths):
        cs = coups_agent(p)
        if not cs:
            continue
        fr = np.load(p)["frames"]
        X.append(D.masque(np.ascontiguousarray(fr[[c["t"] for c in cs]].transpose(0, 3, 1, 2))))
        y += [c["casse_12"] for c in cs]
        portee += [c["portee"] for c in cs]
        g += [k] * len(cs)
    return np.concatenate(X), np.array(y), np.array(g), np.array(portee)


def sessions_agent():
    return [p for p in sorted(D.DATA.glob("dc_arene_*.npz"))
            if "essai" not in p.name and p.name.split("_")[2] in AGENTS]


def main():
    torch.manual_seed(D.SEED)
    rng = np.random.default_rng(D.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    train_files = [p for p in sorted(D.DATA.glob("dc_train*.npz")) if "truth" in np.load(p).files]
    val_files = train_files[4::5]
    fit_files = [p for p in train_files if p not in val_files]
    ag = sessions_agent()
    ag_val, ag_fit = ag[4::5], [p for p in ag if p not in ag[4::5]]
    fit, val, test = D.charge(fit_files), D.charge(val_files), D.charge(sorted(D.DATA.glob("vt_test*.npz")))
    afit, aval = charge_agent(ag_fit), charge_agent(ag_val)
    print(f"sonde : {len(fit[1])} coups | agent : {len(afit[1])} coups ({len(ag_fit)} sessions), "
          f"{afit[1].mean():.0%} cassent, {afit[3].mean():.0%} a portee | agent garde : {len(aval[1])} coups "
          f"({len(ag_val)} sessions)", flush=True)

    X = torch.from_numpy(np.concatenate([fit[0], afit[0]]))
    y = torch.from_numpy(np.concatenate([fit[1], afit[1]]).astype(np.float32))
    m = D.Net().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(D.EPOCHS):
        m.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), D.BATCH):
            j = perm[i:i + D.BATCH]
            loss = loss_fn(m(X[j].float().div(255).to(device)), y[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoque {ep + 1:2d} : AUC agent garde {D.U.auc(D.predit(m, aval[0], device), aval[1]):.1%}", flush=True)
    torch.save(m.state_dict(), OUT / "model.pt")

    ancien = D.Net().to(device)
    ancien.load_state_dict(torch.load(D.OUT / "model.pt", weights_only=True))
    res = {}
    for nom, (Xs, ys, gs) in {"validation_sonde": val[:3], "vt_test": test[:3], "agent_garde": aval[:3]}.items():
        pn, pa = D.predit(m, Xs, device), D.predit(ancien, Xs, device)
        lo, hi = D.ic_ecart(pn, pa, ys, gs, rng)
        res[nom] = {"coups": int(len(ys)), "auc_v3": float(D.U.auc(pn, ys)), "auc_ancien": float(D.U.auc(pa, ys)),
                    "ecart_ic95": [lo, hi], "oui_v3": float((pn >= .5).mean()), "oui_ancien": float((pa >= .5).mean())}
        r = res[nom]
        print(f"{nom:16} {r['coups']:5d} coups | AUC v3 {r['auc_v3']:.1%} | ancien {r['auc_ancien']:.1%} | "
              f"ecart IC 95 % [{lo:+.1%} ; {hi:+.1%}] | dit oui : v3 {r['oui_v3']:.0%}, ancien {r['oui_ancien']:.0%}")
    hors = ~aval[3]
    if hors.any():
        res["agent_garde"]["oui_hors_portee"] = {
            "v3": float((D.predit(m, aval[0][hors], device) >= .5).mean()),
            "ancien": float((D.predit(ancien, aval[0][hors], device) >= .5).mean())}
        print(f"coups gardes hors de portee ({int(hors.sum())}) : dit oui v3 "
              f"{res['agent_garde']['oui_hors_portee']['v3']:.0%}, ancien {res['agent_garde']['oui_hors_portee']['ancien']:.0%}")
    (OUT / "resultats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {OUT / 'resultats.json'}")


if __name__ == "__main__":
    main()
