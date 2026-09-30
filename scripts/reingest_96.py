"""
Re-ingestion 96px : re-decode les mp4 conserves en 96x96, en REUTILISANT les
actions/gui des npz 64px (les actions ne dependent pas de la resolution).

Sources :
  - VPT     : data/raw/<stem>.mp4 (20 Hz natif -> decodage 1:1),
  - YouTube : data/youtube_raw/*.mp4 (reechantillonnage 20 Hz, meme arithmetique
              que ingest_youtube -> memes comptes de frames).
Garde-fou : si le compte de frames 96 ne correspond pas au npz 64 -> episode
saute avec avertissement (pas de desalignement silencieux).

Priorite : les 6 episodes d'eval d'abord, puis le train.
Sortie : data/processed_96/<stem>.npz  (frames (N,96,96,3) + actions + gui copies)
Lance  : .venv\\Scripts\\python.exe scripts\\reingest_96.py [n_max]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
YT_RAW = ROOT / "data" / "youtube_raw"
P64 = ROOT / "data" / "processed"
P64_YT = ROOT / "data" / "processed_youtube"
OUT = ROOT / "data" / "processed_96"
RES = 96
TARGET_FPS = 20.0
EP_LEN = 6001

EVAL_FIRST = ["Player129-f153ac423f61-20210617-173110",
              "treechop-984393664dfd-20210924-174326",
              "Player871-2e9a64a90d31-20210627-154641",
              "Player309-dcc21a4f8784-20210721-163058",
              "nvcave_008", "nvcave_026"]


def decode_vpt(stem: str, n_expected: int):
    mp4 = RAW / f"{stem}.mp4"
    if not mp4.exists():
        return None
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while len(frames) < n_expected:
        okr, fr = cap.read()
        if not okr:
            break
        small = cv2.resize(fr, (RES, RES), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if len(frames) == n_expected else None


_YT_CACHE = {}


def decode_youtube_chunk(stem: str, n_expected: int):
    """nvcave_XXX : XXX-ieme chunk de EP_LEN frames a 20 Hz de la video source."""
    try:
        idx = int(stem.split("_")[-1])
    except ValueError:
        return None
    video = YT_RAW / "nightvision_caving_part13.mp4"
    if not video.exists():
        return None
    if "nv" not in _YT_CACHE:
        # decode toute la video reechantillonnee une fois, garde en RAM (96px : ~2 Go)
        cap = cv2.VideoCapture(str(video))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        step = src_fps / TARGET_FPS
        buf, next_pick, src_i = [], 0.0, 0
        while True:
            okr, fr = cap.read()
            if not okr:
                break
            if src_i >= next_pick:
                small = cv2.resize(fr, (RES, RES), interpolation=cv2.INTER_AREA)
                buf.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                next_pick += step
            src_i += 1
        cap.release()
        _YT_CACHE["nv"] = np.stack(buf)
        print(f"  [cache] video nightvision decodee : {len(buf)} frames @96px")
    allf = _YT_CACHE["nv"]
    start = idx * EP_LEN
    chunk = allf[start: start + n_expected]
    return chunk if chunk.shape[0] == n_expected else None


def main():
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    OUT.mkdir(parents=True, exist_ok=True)

    all64 = {p.stem: p for p in list(P64.glob("*.npz")) + list(P64_YT.glob("*.npz"))}
    order = EVAL_FIRST + sorted(s for s in all64 if s not in EVAL_FIRST)

    done = skip = 0
    for stem in order:
        if done >= n_max:
            break
        out = OUT / f"{stem}.npz"
        if out.exists():
            done += 1
            continue
        d64 = np.load(all64[stem])
        n = d64["frames"].shape[0]
        if stem.startswith("nvcave"):
            frames = decode_youtube_chunk(stem, n)
        else:
            frames = decode_vpt(stem, n)
        if frames is None:
            print(f"[skip] {stem} (source absente ou compte different)")
            skip += 1
            continue
        tmp = out.with_suffix(".npz.part")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, frames=frames, actions=d64["actions"], gui=d64["gui"])
        tmp.replace(out)
        done += 1
        if done % 20 == 0:
            print(f"{done} episodes @96px ({skip} sautes)", flush=True)

    print(f"\nTermine : {done} episodes @96px, {skip} sautes -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
