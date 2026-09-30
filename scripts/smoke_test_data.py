"""
Smoke test donnees -- DREAMCRAFT (Phase 0).

But : verifier qu'on peut recuperer de vraies donnees (frame, action) Minecraft
SANS installer MineRL (ni Java, ni compilation), via le dataset VPT d'OpenAI
(contractor demonstrations : mp4 + jsonl d'actions, 360p @ 20 Hz).

Ce que ca fait :
  1. telecharge l'index "early game" (6xx) -- riche en bucheronnage,
  2. choisit une demo de taille raisonnable,
  3. telecharge sa video (.mp4) + ses actions (.jsonl),
  4. echantillonne quelques frames, les sauve en PNG (taille d'origine + 64x64),
  5. decode l'action humaine de chaque frame (avancer / taper / camera...).

Lance : .venv\\Scripts\\python.exe scripts\\smoke_test_data.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "vpt_smoke"
OUT_DIR = ROOT / "outputs" / "smoke_test"

N_FRAMES = 6        # nb de frames a echantillonner
MAX_MP4_MB = 200    # on saute les demos trop lourdes
LOWRES = 64         # resolution "modele" (low-compute)

KEYMAP = {
    "key.keyboard.w": "avancer",
    "key.keyboard.s": "reculer",
    "key.keyboard.a": "gauche",
    "key.keyboard.d": "droite",
    "key.keyboard.space": "sauter",
    "key.keyboard.left.shift": "sneak",
}


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def head_size_mb(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        cl = r.headers.get("Content-Length")
        return (int(cl) / 1e6) if cl else None


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r  ...{total / 1e6:6.1f} MB", end="", flush=True)
    print()
    return dest


def decode_action(a):
    keys = a.get("keyboard", {}).get("keys", []) or []
    pressed = [KEYMAP[k] for k in keys if k in KEYMAP]
    mouse = a.get("mouse", {})
    buttons = mouse.get("buttons", []) or []
    attack = 0 in buttons   # bouton gauche = casser / attaquer
    use = 1 in buttons      # bouton droit = utiliser / placer
    dx = mouse.get("dx", 0.0)
    dy = mouse.get("dy", 0.0)
    label = list(pressed)
    if attack:
        label.append("TAPER")
    if use:
        label.append("utiliser")
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        label.append(f"camera(dx={dx:+.0f},dy={dy:+.0f})")
    return ", ".join(label) if label else "(rien)"


def main():
    print("=== SMOKE TEST DONNEES VPT ===")
    print(f"Index: {INDEX_URL}")
    idx = fetch_json(INDEX_URL)
    basedir = idx["basedir"].rstrip("/")
    relpaths = idx["relpaths"]
    print(f"basedir = {basedir}")
    print(f"{len(relpaths)} demonstrations dans l'index 'early game' (6xx).")

    chosen = None
    for rp in relpaths[:15]:
        # NB : dans cet index, le relpath inclut deja l'extension .mp4
        mp4_url = f"{basedir}/{rp}"
        try:
            size = head_size_mb(mp4_url)
        except Exception as e:
            print(f"  skip {rp} (HEAD KO: {e})")
            continue
        if size is None:
            continue
        print(f"  candidat {rp} -> {size:.1f} MB")
        if size <= MAX_MP4_MB:
            chosen = (rp, mp4_url, size)
            break

    if chosen is None:
        print(f"Aucune demo <= {MAX_MP4_MB} MB dans les 15 premieres.")
        sys.exit(1)

    rp, mp4_url, size = chosen
    jsonl_url = f"{basedir}/{rp[:-4]}.jsonl"   # remplace .mp4 -> .jsonl
    name = Path(rp).stem
    print(f"\nDemo choisie: {rp}  ({size:.1f} MB)")

    mp4_path = DATA_DIR / f"{name}.mp4"
    jsonl_path = DATA_DIR / f"{name}.jsonl"
    print("Telechargement video...")
    download(mp4_url, mp4_path)
    print("Telechargement actions...")
    download(jsonl_url, jsonl_path)

    actions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                actions.append(json.loads(line))
    print(f"\nActions: {len(actions)} lignes (1 par frame attendue).")

    cap = cv2.VideoCapture(str(mp4_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {n_frames} frames, {fps:.1f} fps, {w}x{h}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    last = max(1, min(n_frames, len(actions)) - 1)
    sample_idx = np.linspace(0, last, N_FRAMES).astype(int)

    print("\n--- (frame, action) echantillonnees ---")
    summary = []
    for k, fi in enumerate(sample_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            print(f"frame {fi}: lecture KO")
            continue
        act = decode_action(actions[fi]) if fi < len(actions) else "(pas d'action)"
        full_png = OUT_DIR / f"f{k:02d}_idx{fi}_full.png"
        low_png = OUT_DIR / f"f{k:02d}_idx{fi}_{LOWRES}px.png"
        cv2.imwrite(str(full_png), frame)
        low = cv2.resize(frame, (LOWRES, LOWRES), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(low_png), low)
        print(f"frame {fi:5d}  ->  {act}")
        summary.append(
            {
                "frame": int(fi),
                "action": act,
                "full_png": str(full_png.relative_to(ROOT)),
                "low_png": str(low_png.relative_to(ROOT)),
            }
        )
    cap.release()

    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "demo": rp,
                "video": {"frames": n_frames, "fps": fps, "w": w, "h": h},
                "samples": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nOK. Frames + {LOWRES}x{LOWRES} sauvees dans : {OUT_DIR.relative_to(ROOT)}")
    print("Resume : outputs/smoke_test/summary.json")


if __name__ == "__main__":
    main()
