"""Donnees VPT (OpenAI contractor demos) -> paires (frame, action).

Format source : par episode, un .mp4 (640x360 @ 20 Hz) + un .jsonl
(1 ligne d'action par frame). L'action de la ligne t decrit ce que le joueur
faisait entre la frame t et la frame t+1 :

    frame[t] + action[t]  ->  frame[t+1]

Format produit (npz, un par episode) :
    frames  : uint8  (N, RES, RES, 3)  RGB
    actions : float32 (N-1, ACTION_DIM) -- action[t] = transition t -> t+1
    gui     : uint8  (N-1,)             -- 1 si un menu/GUI est ouvert (a exclure)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

INDEX_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json"

# --- Espace d'action compact -------------------------------------------------
# 8 booleens + 2 continus (camera), dans cet ordre :
ACTION_NAMES = [
    "avancer", "reculer", "gauche", "droite",
    "sauter", "sneak", "taper", "utiliser",
    "cam_dx", "cam_dy",
]
ACTION_DIM = len(ACTION_NAMES)

# La camera (dx, dy en pixels souris) est ecretee puis normalisee dans [-1, 1].
# 60 px couvre ~99% des mouvements observes (les coups de souris plus violents
# sont rares et saturent a +-1).
CAM_CLIP = 60.0

_KEY_TO_IDX = {
    "key.keyboard.w": 0,
    "key.keyboard.s": 1,
    "key.keyboard.a": 2,
    "key.keyboard.d": 3,
    "key.keyboard.space": 4,
    "key.keyboard.left.shift": 5,
}


def encode_action(line: dict) -> np.ndarray:
    """Ligne jsonl VPT -> vecteur action float32 (ACTION_DIM,)."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    keys = (line.get("keyboard") or {}).get("keys") or []
    for k in keys:
        idx = _KEY_TO_IDX.get(k)
        if idx is not None:
            a[idx] = 1.0
    mouse = line.get("mouse") or {}
    buttons = mouse.get("buttons") or []
    if 0 in buttons:  # clic gauche = taper / casser
        a[6] = 1.0
    if 1 in buttons:  # clic droit = utiliser / placer
        a[7] = 1.0
    a[8] = float(np.clip(mouse.get("dx", 0.0), -CAM_CLIP, CAM_CLIP)) / CAM_CLIP
    a[9] = float(np.clip(mouse.get("dy", 0.0), -CAM_CLIP, CAM_CLIP)) / CAM_CLIP
    return a


def is_gui_open(line: dict) -> bool:
    """Menu/inventaire ouvert ? (dynamique differente -> on exclut ces frames)."""
    if "isGuiOpen" in line:
        return bool(line["isGuiOpen"])
    return bool((line.get("metadata") or {}).get("isGuiOpen", False))


# --- Telechargement ----------------------------------------------------------

def fetch_index(cache: Path | None = None) -> tuple[str, list[str]]:
    """Retourne (basedir, relpaths). Les relpaths incluent deja '.mp4'."""
    if cache is not None and cache.exists():
        idx = json.loads(cache.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(INDEX_URL, timeout=60) as r:
            idx = json.load(r)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(idx), encoding="utf-8")
    return idx["basedir"].rstrip("/"), idx["relpaths"]


def download(url: str, dest: Path, quiet: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if not quiet:
                print(f"\r  ...{total / 1e6:6.1f} MB", end="", flush=True)
    if not quiet:
        print()
    tmp.replace(dest)
    return dest


# --- Conversion episode -> npz ----------------------------------------------

def build_episode_npz(mp4: Path, jsonl: Path, out: Path, res: int = 64) -> dict:
    """Decode la video en sequence RES x RES + aligne les actions. Sauve en npz."""
    actions_raw = []
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                actions_raw.append(json.loads(line))

    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (res, res), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()

    n = min(len(frames), len(actions_raw) + 1)
    frames = np.stack(frames[:n])                                   # (N, res, res, 3)
    actions = np.stack([encode_action(a) for a in actions_raw[: n - 1]])  # (N-1, D)
    gui = np.array([is_gui_open(a) for a in actions_raw[: n - 1]], dtype=np.uint8)

    out.parent.mkdir(parents=True, exist_ok=True)
    # ecriture atomique : jamais de npz a moitie ecrit visible par un glob
    tmp = out.with_suffix(".npz.part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, frames=frames, actions=actions, gui=gui)
    tmp.replace(out)
    return {
        "frames": int(frames.shape[0]),
        "res": res,
        "gui_frames": int(gui.sum()),
        "npz": str(out),
    }


def load_episode(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (frames uint8 (N,R,R,3), actions float32 (N-1,D), gui uint8 (N-1,))."""
    d = np.load(npz_path)
    return d["frames"], d["actions"], d["gui"]
