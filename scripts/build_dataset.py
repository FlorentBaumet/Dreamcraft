"""
Construit le mini-dataset Phase 0 : N episodes VPT -> npz (frames 64x64 + actions).

- Reutilise les fichiers deja telecharges (data/vpt_smoke, data/raw) si presents.
- Saute proprement les episodes dont les fichiers ont disparu du bucket (404).

Lance : .venv\\Scripts\\python.exe scripts\\build_dataset.py [n_episodes] [stride]

stride > 1 : echantillonne l'index tous les `stride` episodes -> joueurs et
periodes varies (au lieu des N premiers, tous du meme joueur).
"""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import vpt  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
SMOKE_DIR = ROOT / "data" / "vpt_smoke"
PROC_DIR = ROOT / "data" / "processed"
RES = 64


def find_local(name: str, ext: str) -> Path | None:
    for d in (RAW_DIR, SMOKE_DIR):
        p = d / f"{name}{ext}"
        if p.exists():
            return p
    return None


def main():
    n_wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    live_file = RAW_DIR / "live_relpaths.json"
    if live_file.exists():
        # liste des episodes encore heberges (scan_live_episodes.py),
        # sessions treechop en tete (les plus utiles pour le Monde 2)
        import json as _json
        d = _json.loads(live_file.read_text(encoding="utf-8"))
        basedir = d["basedir"].rstrip("/")
        relpaths = sorted(d["live"], key=lambda rp: (0 if "treechop" in rp else 1))
        print(f"Source : live_relpaths.json ({len(relpaths)} episodes vivants).")
    else:
        basedir, relpaths = vpt.fetch_index(cache=RAW_DIR / "index_6xx.json")
        if stride > 1:
            relpaths = relpaths[::stride]
        print(f"Source : index complet (stride {stride}).")
    print(f"Objectif : {n_wanted} episodes en npz.")

    done = 0
    for rp in relpaths:
        if done >= n_wanted:
            break
        name = Path(rp).stem
        out = PROC_DIR / f"{name}.npz"
        if out.exists():
            print(f"[ok deja] {name}")
            done += 1
            continue

        mp4 = find_local(name, ".mp4")
        jsonl = find_local(name, ".jsonl")
        try:
            if mp4 is None:
                print(f"[dl mp4 ] {name}")
                mp4 = vpt.download(f"{basedir}/{rp}", RAW_DIR / f"{name}.mp4")
            if jsonl is None:
                print(f"[dl json] {name}")
                jsonl = vpt.download(f"{basedir}/{rp[:-4]}.jsonl", RAW_DIR / f"{name}.jsonl")
        except urllib.error.HTTPError as e:
            print(f"[skip   ] {name} (HTTP {e.code})")
            continue
        except Exception as e:
            # timeout / reset reseau : on saute, un prochain run retentera
            print(f"[skip   ] {name} ({type(e).__name__})")
            continue

        print(f"[npz    ] {name} ...")
        info = vpt.build_episode_npz(mp4, jsonl, out, res=RES)
        print(f"          {info['frames']} frames @ {RES}px, GUI: {info['gui_frames']} frames")
        done += 1

    print(f"\nTermine : {done}/{n_wanted} episodes dans {PROC_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
