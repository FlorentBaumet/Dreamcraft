"""
Scanne l'index VPT (requetes HEAD uniquement) pour trouver les episodes
encore heberges. Ecrit la liste des vivants dans data/raw/live_relpaths.json.

Lance : .venv\\Scripts\\python.exe scripts\\scan_live_episodes.py [n_probes] [stride] [offset]

Fusionne avec la liste existante (on peut relancer avec d'autres strides/offsets
pour couvrir d'autres zones de l'index sans perdre les acquis).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dreamcraft.data import vpt  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT = RAW_DIR / "live_relpaths.json"


def is_live(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20):
            return True
    except Exception:
        return False


def main():
    n_probes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    basedir, relpaths = vpt.fetch_index(cache=RAW_DIR / "index_6xx.json")
    probes = relpaths[offset::stride][:n_probes]

    known = set()
    if OUT.exists():
        known = set(json.loads(OUT.read_text(encoding="utf-8"))["live"])
    probes = [rp for rp in probes if rp not in known]

    live = list(known)
    n_new = 0
    for i, rp in enumerate(probes):
        if is_live(f"{basedir}/{rp}"):
            live.append(rp)
            n_new += 1
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(probes)} sondes, +{n_new} nouveaux vivants")
    OUT.write_text(json.dumps({"basedir": basedir, "live": live}, indent=2),
                   encoding="utf-8")
    print(f"\n+{n_new} nouveaux ({len(live)} au total) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
