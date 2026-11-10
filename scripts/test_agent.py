"""
Tests de l'agent, sans le jeu : alignement des donnees, paires, detection de fenetre.

Lance : .venv\\Scripts\\python.exe scripts\\test_agent.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft import game_window as GW  # noqa: E402
from dreamcraft import verite as V  # noqa: E402
from dreamcraft.data import context as C  # noqa: E402

_s = importlib.util.spec_from_file_location("agent", Path(__file__).parent / "agent_loop.py")
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)


class EchoEnv:
    """L'image montre la derniere action recue : un decalage de labels se voit aussitot."""

    def __init__(self):
        self.value = 0

    def observe(self):
        return np.full((64, 64, 3), self.value, np.uint8)

    def act(self, a):
        self.value = code(a)

    def close(self):
        pass


def code(a):
    return int(100 * (a[A.IDX["taper"]] > 0.5) + 50 * (a[A.IDX["avancer"]] > 0.5)
               + 25 * (abs(a[A.CAM_DX]) > 0))


def test_labels_alignes():
    """actions[t] doit mener de frames[t] a frames[t+1], comme dans le corpus."""
    r = A.run_loop(EchoEnv(), A.ProbePolicy(), 400, 8, 1, fps=None)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ep.npz"
        A.save_episode(p, r["frames"], r["actions"], r["phases"])
        frames, actions, ok = C.load_episode_u8(p)
        phase = np.load(p)["phase"]
    assert len(frames) == len(actions) + 1 == len(ok) + 1 == len(phase) + 1
    for t in range(len(actions)):
        assert frames[t + 1][0, 0, 0] == code(actions[t]), f"decalage a t={t}"


def phases_de(n, **kw):
    pol = A.ProbePolicy(**kw)
    out = [pol.plan(None, 8) for _ in range(n)]
    return [o[1] for o in out], np.stack([o[0][0] for o in out])


def test_temoin_avant_le_coup():
    names, _ = phases_de(70, hold=30, pivot=10)
    assert names[:30] == ["B_temoin"] * 30
    assert names[30:60] == ["A_taper"] * 30
    assert names[60:70] == ["C_pivot"] * 10


def test_seul_le_bouton_change_dans_une_paire():
    names, acts = phases_de(2000)
    names = np.array(names)
    b, a = acts[names == "B_temoin"], acts[names == "A_taper"]
    assert not b.any(), "le temoin ne doit envoyer aucune action"
    assert (a[:, A.IDX["taper"]] == 1).all()
    a0 = a.copy()
    a0[:, A.IDX["taper"]] = 0
    assert not a0.any(), "en phase A, le seul signal est le bouton"


def test_cycles_varies():
    names, acts = phases_de(6000)
    names = np.array(names)
    assert "D_marche" in names, "il faut changer d'endroit"
    assert len({round(float(v), 3) for v in acts[names == "C_pivot", A.CAM_DX]}) > 5
    pol = A.ProbePolicy()
    for _ in range(6000):
        pol.plan(None, 8)
        assert -400 <= pol.pitch <= 200, f"inclinaison hors bornes : {pol.pitch}"


def test_nouvelle_session_repart_a_zero():
    pol = A.ProbePolicy()
    for _ in range(100):                 # s'arrete au milieu d'un cycle
        pol.plan(None, 8)
    pol.new_session()
    assert pol.pitch == 0.0
    assert [pol.plan(None, 8)[1] for _ in range(30)] == ["B_temoin"] * 30


def test_commandes_de_preparation():
    assert A.SETUP_COMMANDS[0] == "/gamerule sendCommandFeedback false", \
        "un message de chat ne doit jamais apparaitre sur les images"
    for c in A.SETUP_COMMANDS:
        assert c.startswith("/") and c.isascii(), c   # le presse-papier passe en ascii


def test_titre_de_fenetre():
    for bon in ("Minecraft 1.19.4", "Minecraft* 1.19.4 - Singleplayer", "Minecraft 1.21.5"):
        assert GW.is_game_title(bon), bon
    for mauvais in ("Minecraft Launcher", "Minecraft_World_Model", None, "",
                    "minecraft - Google Chrome", "agent_loop.py - Minecraft_World_Model",
                    "Minecraft World Model"):
        assert not GW.is_game_title(mauvais), mauvais


def fake_session(path, seed, n_cycles=60):
    """Sonde simulee : un coup sur deux casse le bloc vise, l'autre tombe dans le vide."""
    rng = np.random.default_rng(seed)
    pol = A.ProbePolicy(seed=seed)
    frames, actions, phases = [rng.integers(0, 255, (64, 64, 3), np.uint8)], [], []
    n_a, utile = 0, []
    for _ in range(n_cycles * 70):
        plan, name = pol.plan(None, 8)
        f = frames[-1].copy()
        if name == "A_taper" and (not phases or phases[-1] != "A_taper"):
            n_a += 1
            utile.append(n_a % 2 == 0)
        if name == "A_taper" and utile[-1] and sum(p == "A_taper" for p in phases[-3:]) == 3:
            f[26:38, 26:38] = rng.integers(0, 255, (12, 12, 3), np.uint8)
        if name in ("C_pivot", "D_marche"):
            f = rng.integers(0, 255, (64, 64, 3), np.uint8)
        frames.append(f)
        actions.append(plan[0])
        phases.append(name)
    A.save_episode(path, frames, actions, phases)
    return np.array(utile)


def test_mesure_des_paires():
    """La mesure retrouve les coups utiles, et le seuil separe les deux groupes."""
    _e = importlib.util.spec_from_file_location("ep", Path(__file__).parent / "eval_pairs.py")
    E = importlib.util.module_from_spec(_e)
    _e.loader.exec_module(E)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dc_test_x.npz"
        utile = fake_session(p, seed=1)
        s = E.load_session(p)
    assert len(s["starts"]) >= 50
    assert s["eff_b"].max() == 0, "le temoin ne change rien dans cette simulation"
    label = s["eff_a"] > E.otsu(s["eff_a"])
    assert (label == utile[:len(label)]).all(), "labels differents de la verite simulee"
    assert E.U.auc(s["eff_a"], label) == 1.0, "un score oracle doit atteindre 100 %"
    assert E.U.auc(np.zeros(len(label)), label) == 0.5, "un score constant doit valoir 50 %"


def ligne(tick, bloc=1, dist=20, casse=0, butin=0):
    return f"[18:00:00] [Render thread/INFO]: [System] [CHAT] DC {tick} {bloc} {dist} {casse} {butin}\n".encode()


def test_journal_du_jeu():
    """Le journal ne perd aucun bloc casse, meme si la ligne arrive coupee en deux."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "latest.log"
        p.write_bytes(ligne(1, casse=9))              # avant l'ouverture : ignoree
        j = V.JournalJeu(p)
        try:
            assert j.lire()[0] == -1
            with open(p, "ab") as f:
                f.write(ligne(2) + b"[Server thread/INFO]: autre chose\n" + ligne(3, casse=5) + ligne(4))
            r = j.lire()
            assert r[0] == 4 and r[3] == 5, "le bloc casse au tick 3 doit etre vu"
            assert j.lire()[3] == 0, "un bloc casse n'est annonce qu'une fois"
            coupe = ligne(5, casse=7)
            with open(p, "ab") as f:
                f.write(coupe[:20])
            assert j.lire()[0] == 4
            with open(p, "ab") as f:
                f.write(coupe[20:])
            r = j.lire()
            assert r[0] == 5 and r[3] == 7
            # minuit : le jeu recommence un journal vide, la verite doit continuer
            p.write_bytes(ligne(6, casse=3))
            r = j.lire()
            assert r[0] == 6 and r[3] == 3, "apres le changement de journal, la verite doit suivre"
        finally:
            j.close()


def test_regle_de_cassage():
    attendu = {"netherrack": 2, "stone": 6, "cobblestone": 8, "coal_ore": 12, "dirt": 15,
               "pumpkin": 30, "oak_planks": 60, "obsidian": 188}
    for nom, ticks in attendu.items():
        assert V.ticks_theorie(nom) == ticks, nom
    assert V.ticks_theorie("oak_log", "hache") == 8 and V.ticks_theorie("oak_log") == 60
    assert V.ticks_theorie("oak_leaves", "hache") == 6
    assert V.bloc_id("pumpkin") == 22, "les anciens numeros de bloc ne doivent pas bouger"
    assert all(V.bloc_nom(V.bloc_id(b[0])) == b[0] for b in V.BLOCS)
    assert V.bloc_nom(0) == "rien" and V.bloc_nom(V.AUTRE) == "autre"


def test_datapack():
    pack = V.fichiers_du_pack()
    assert json.loads(pack["pack.mcmeta"])["pack"]["pack_format"] == 12      # Minecraft 1.19.4
    for chemin, texte in pack.items():
        assert texte.isascii(), chemin
        if chemin.endswith(".mcfunction"):
            for cmd in texte.splitlines():
                assert not cmd.startswith("/"), f"pas de / dans une fonction : {cmd}"
                for appel in re.findall(r"function (dreamcraft:\w+)", cmd):
                    assert f"data/dreamcraft/functions/{appel.split(':')[1]}.mcfunction" in pack, appel
    charge = pack["data/dreamcraft/functions/charge.mcfunction"].splitlines()
    assert all(len(c.split()[3]) <= 16 for c in charge), "nom d'objectif trop long"
    assert len(charge) == len(V.BLOCS) + 1
    for salle in V.SALLES:
        assert salle in [b[0] for b in V.BLOCS], salle
        assert salle in pack["data/dreamcraft/functions/identifie.mcfunction"]
    tick = pack["data/dreamcraft/functions/joueur.mcfunction"]
    assert tick.index("function dreamcraft:rayon") < tick.index("tellraw"), "le rayon d'abord"


def test_alternance_des_salles():
    assert [V.salle_de(i, "naturel") for i in range(4)] == [None] * 4
    mixte = [V.salle_de(i, "mixte") for i in range(2 * len(V.SALLES))]
    assert mixte[::2] == [None] * len(V.SALLES), "une session sur deux en terrain naturel"
    assert mixte[1::2] == V.SALLES, "chaque type de salle une fois par tour"


class TruthEnv(EchoEnv):
    """L'etat du jeu porte le numero de l'image : un decalage avec les images se voit aussitot."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def read_truth(self):
        self.n += 1
        return np.array([self.n - 1, 1, 20, 3 if self.n % 10 == 0 else 0, 0])


def test_verite_alignee_sur_les_images():
    r = A.run_loop(TruthEnv(), A.ProbePolicy(), 100, 8, 1, fps=None)
    assert len(r["truth"]) == len(r["frames"]) == 101
    assert (r["truth"][:, 0] == np.arange(101)).all()
    assert V.blocs_casses(r["truth"]) == 10
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ep.npz"
        A.save_episode(p, r["frames"], r["actions"], r["phases"], r["truth"], "obsidian")
        with np.load(p) as z:
            assert str(z["room"]) == "obsidian" and len(z["truth"]) == 101
        frames, actions, _ = C.load_episode_u8(p)         # le format du corpus reste lisible
        assert len(frames) == 101 and len(actions) == 100
    assert A.run_loop(EchoEnv(), A.ProbePolicy(), 10, 8, 1, fps=None)["truth"] is None


def test_analyse_de_la_verite():
    """Un bloc casse 6 ticks apres le debut du coup, l'image change juste apres l'annonce."""
    _a = importlib.util.spec_from_file_location("av", Path(__file__).parent / "analyse_verite.py")
    AV = importlib.util.module_from_spec(_a)
    _a.loader.exec_module(AV)
    rng = np.random.default_rng(0)
    pol = A.ProbePolicy()
    frames, actions, phases = [rng.integers(0, 255, (64, 64, 3), np.uint8)], [], []
    truth = [[0, 1, 20, 0, 0]]
    debut = None
    for t in range(1, 400):
        plan, name = pol.plan(None, 8)
        if name == "A_taper" and (not phases or phases[-1] != "A_taper"):
            debut = t - 1
        f = frames[-1].copy()
        casse = name == "A_taper" and t - debut == 6       # truth[t] decrit l'image t
        if name == "A_taper" and t - debut == 7:
            f[26:38, 26:38] = rng.integers(0, 255, (12, 12, 3), np.uint8)
        if name in ("C_pivot", "D_marche"):
            f = rng.integers(0, 255, (64, 64, 3), np.uint8)
        frames.append(f)
        actions.append(plan[0])
        phases.append(name)
        truth.append([t, 1, 20, 1 if casse else 0, 0])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.npz"
        A.save_episode(p, frames, actions, phases, np.array(truth), None)
        cs = AV.coups(p)
        bruit = AV.bruit_chat(p)
    assert len(cs) >= 4 and bruit == 0
    for c in cs:
        assert c["casse"] == 1 and c["ticks"] == 6 and c["casse_12"], c
        assert c["effet"] > 0.01 and c["lag"] in (0, 1, 2), c
    assert "anchored feet" in V.fichiers_du_pack()["data/dreamcraft/functions/tick.mcfunction"]


def main():
    tests =[v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passes")


if __name__ == "__main__":
    main()
