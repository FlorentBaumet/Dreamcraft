"""
Test de plomberie sur le vrai jeu (avant de brancher le modele).

  A. VOIR  : je capture bien le jeu, et le 64x64 ressemble-t-il aux donnees ?
  B. AGIR  : la souris arrive-t-elle au jeu, et en combien de temps l'image reagit ?
  C. CLIC  : avec --clic, le clic gauche arrive-t-il ? Casse le bloc vise : a lancer
             seulement dans un monde de test, face a un bloc proche.

Securites : n'agit QUE si le jeu est au premier plan, compte a rebours avant
d'envoyer quoi que ce soit, s'arrete si la fenetre perd le focus.

Lance : .venv\\Scripts\\python.exe scripts\\live_smoke.py [--clic]
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft import game_window as GW  # noqa: E402

OUT = ROOT / "outputs" / "live"
RES = 64
FPS = 20.0


def grab64(sct, box):
    img = np.array(sct.grab(box))[:, :, :3]          # BGRA -> BGR
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (RES, RES),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def see(sct, box):
    frames = []
    t0 = time.time()
    for i in range(20):
        frames.append(grab64(sct, box))
        time.sleep(max(0, (i + 1) / FPS - (time.time() - t0)))
    hz = 20 / (time.time() - t0)
    rows = [np.concatenate(frames[r * 5:(r + 1) * 5], axis=1) for r in range(4)]
    big = cv2.resize(np.concatenate(rows, axis=0).astype(np.uint8), None, fx=3, fy=3,
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "smoke_vue.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    noise = float(np.mean([np.abs(frames[i + 1] - frames[i]).mean() for i in range(19)]))
    print(f"[A] VOIR ok : {hz:.1f} Hz | luminosite {np.mean(frames) / 255:.2f} | "
          f"bruit entre images {noise:.2f}")
    print("    -> a verifier a l'oeil : outputs/live/smoke_vue.png")
    return noise


def camera_lag(sct, box, noise, hwnd):
    """Temps entre l'envoi d'un mouvement de souris et le changement a l'image."""
    delays = []
    for i in range(6):
        time.sleep(0.5)
        if not GW.is_foreground(hwnd):
            print("    arret : le jeu a perdu le focus")
            return
        before = grab64(sct, box)
        t0 = time.perf_counter()
        ctypes.windll.user32.mouse_event(0x0001, 60 if i % 2 == 0 else -60, 0, 0, 0)
        while time.perf_counter() - t0 < 0.5:
            if np.abs(grab64(sct, box) - before).mean() > max(3.0, 3 * noise):
                delays.append((time.perf_counter() - t0) * 1000)
                break
    if len(delays) < 4:
        print(f"[B] AGIR : ECHEC, la vue n'a reagi que {len(delays)} fois sur 6 "
              "(le jeu est-il en pause ? le curseur est-il capture ?)")
        return
    ms = float(np.median(delays))
    print(f"[B] AGIR ok : reaction en {ms:.0f} ms, soit {ms / 50:.1f} image a 20 Hz "
          f"({len(delays)}/6 essais)")


def click(sct, box, hwnd):
    import pydirectinput
    pydirectinput.PAUSE = 0.0
    if not GW.is_foreground(hwnd):
        print("[C] CLIC : saute, le jeu n'a plus le focus")
        return
    before = grab64(sct, box)
    pydirectinput.mouseDown(button="left")
    try:
        time.sleep(1.5)
        after = grab64(sct, box)
    finally:
        pydirectinput.mouseUp(button="left")
    centre = float(np.abs(after - before)[26:38, 26:38].mean())
    cv2.imwrite(str(OUT / "smoke_clic.png"), cv2.cvtColor(
        cv2.resize(np.concatenate([before, after], axis=1).astype(np.uint8), None, fx=4, fy=4,
                   interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    print(f"[C] CLIC : changement au centre {centre:.1f} -> outputs/live/smoke_clic.png "
          "(avant/apres : le bloc a-t-il casse ?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clic", action="store_true", help="teste aussi le clic gauche (casse un bloc)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    found = GW.find_game_window()
    if found is None:
        print("ECHEC : aucune fenetre du jeu trouvee (titre attendu : 'Minecraft 1.19.4...').")
        print("  -> Lance Minecraft Java, charge un monde, en mode FENETRE.")
        sys.exit(1)
    hwnd, title = found
    box = GW.client_box(hwnd)
    print(f"fenetre : '{title}' | zone {box['width']}x{box['height']} "
          f"(ratio {box['width'] / max(box['height'], 1):.2f}, attendu 1.78)")

    with mss.mss() as sct:
        noise = see(sct, box)
        if not GW.is_foreground(hwnd):
            print("Clique dans le jeu dans les 5 s (curseur capture, pas de menu ouvert)...")
            t0 = time.time()
            while time.time() - t0 < 5 and not GW.is_foreground(hwnd):
                time.sleep(0.1)
        if not GW.is_foreground(hwnd):
            print("Le jeu n'est pas au premier plan : tests B et C sautes.")
            return
        print("Depart dans 3 s...")
        time.sleep(3)
        camera_lag(sct, box, noise, hwnd)
        if args.clic:
            click(sct, box, hwnd)


if __name__ == "__main__":
    main()
