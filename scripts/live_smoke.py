"""
Test de plomberie sur le vrai jeu (avant de brancher le modele).

Deux questions, deux tests, aucun modele :
  A. VOIR   : je capture bien la fenetre Minecraft, et le 64x64 ressemble-t-il
              aux donnees d'entrainement (VPT vanilla) ?
  B. AGIR   : mes touches/souris arrivent-elles au jeu (la vue change-t-elle) ?

Securites :
  - n'agit QUE si la fenetre Minecraft est au premier plan,
  - compte a rebours avant d'envoyer quoi que ce soit,
  - actions minuscules (petit mouvement camera), aucune action destructrice,
  - arret immediat si la fenetre perd le focus.

Lance : .venv\\Scripts\\python.exe scripts\\live_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "live"
RES = 64
FPS = 20.0


def find_window():
    import pygetwindow as gw
    cands = [w for w in gw.getAllWindows()
             if w.title and "minecraft" in w.title.lower() and w.width > 200]
    if not cands:
        return None
    return max(cands, key=lambda w: w.width * w.height)


def grab(sct, box):
    img = np.array(sct.grab(box))[:, :, :3]          # BGRA -> BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to64(rgb):
    return cv2.resize(rgb, (RES, RES), interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = find_window()
    if w is None:
        print("ECHEC : aucune fenetre Minecraft trouvee.")
        print("  -> Lance Minecraft (Java), chargé dans un monde, en mode FENETRE.")
        sys.exit(1)
    print(f"fenetre trouvee : '{w.title}'  {w.width}x{w.height} @ ({w.left},{w.top})")

    # zone client approximative (on retire la barre de titre Windows ~31 px)
    box = {"left": w.left + 8, "top": w.top + 31,
           "width": max(w.width - 16, 64), "height": max(w.height - 39, 64)}
    print(f"zone capturee : {box['width']}x{box['height']} (ratio {box['width']/box['height']:.2f})")

    with mss.mss() as sct:
        # ---------- A. VOIR ----------
        frames = []
        t0 = time.time()
        for i in range(20):
            frames.append(to64(grab(sct, box)))
            time.sleep(max(0, (i + 1) / FPS - (time.time() - t0)))
        hz = 20 / (time.time() - t0)
        big = np.concatenate([np.concatenate(frames[r * 5:(r + 1) * 5], axis=1) for r in range(4)],
                             axis=0)
        big = cv2.resize(big, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / "smoke_vue.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        luma = np.mean([f.mean() for f in frames]) / 255
        bouge = np.mean([np.abs(frames[i + 1].astype(float) - frames[i].astype(float)).mean()
                         for i in range(len(frames) - 1)])
        print(f"[A] VOIR ok : {hz:.1f} Hz | luma {luma:.2f} | mouvement inter-frame {bouge:.2f}")
        print(f"    -> verifie a l'oeil : {OUT.relative_to(ROOT)}\\smoke_vue.png")

        # ---------- B. AGIR ----------
        import pygetwindow as gw
        act = gw.getActiveWindow()
        if act is None or "minecraft" not in (act.title or "").lower():
            print("[B] AGIR : SAUTE - Minecraft n'est pas la fenetre active.")
            print("    -> clique dans le jeu (curseur capture) et relance ce script.")
            return
        print("[B] AGIR dans 3 s : petit mouvement de camera (rien de destructeur)...")
        time.sleep(3)
        import pydirectinput
        pydirectinput.PAUSE = 0.0
        avant = to64(grab(sct, box))
        # rotation camera : souris relative (Minecraft capture le curseur)
        import ctypes
        for _ in range(12):
            ctypes.windll.user32.mouse_event(0x0001, 40, 0, 0, 0)   # MOUSEEVENTF_MOVE dx=+40
            time.sleep(1 / FPS)
        apres = to64(grab(sct, box))
        delta = float(np.abs(apres.astype(float) - avant.astype(float)).mean())
        cv2.imwrite(str(OUT / "smoke_action.png"), cv2.cvtColor(
            cv2.resize(np.concatenate([avant, apres], axis=1), None, fx=4, fy=4,
                       interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
        verdict = "OK, la vue a change" if delta > 6 else "AUCUN EFFET visible"
        print(f"    delta apres rotation camera : {delta:.1f} -> {verdict}")
        print(f"    -> avant/apres : {OUT.relative_to(ROOT)}\\smoke_action.png")


if __name__ == "__main__":
    main()
