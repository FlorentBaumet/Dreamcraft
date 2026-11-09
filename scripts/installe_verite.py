"""
Prepare le jeu pour la verite : datapack dans les mondes, taille du chat a 0.
A lancer jeu ferme : le jeu reecrit options.txt en quittant, et charge le datapack a l'ouverture.

Lance : .venv\\Scripts\\python.exe scripts\\installe_verite.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dreamcraft import game_window as GW  # noqa: E402
from dreamcraft import verite as V  # noqa: E402


def main():
    if GW.find_game_window() is not None:
        sys.exit("Minecraft est ouvert : ferme-le, il ecraserait le reglage du chat en quittant.")
    jeu = V.dossier_jeu()
    for monde in ("dc_train", "dc_test"):
        print(f"datapack -> {V.installe_pack(monde, jeu)}")
    if V.chat_masque(jeu):
        print("chat deja masque")
        return
    opt = jeu / "options.txt"
    sauve = opt.with_name("options.avant_dreamcraft.txt")
    if not sauve.exists():
        shutil.copy2(opt, sauve)
    texte, n = re.subn(rb"^chatScale:[\d.]+", b"chatScale:0.0", opt.read_bytes(), flags=re.M)
    if n != 1:
        sys.exit("chatScale introuvable dans options.txt : a regler dans Options > Chat")
    opt.write_bytes(texte)
    print(f"taille du chat a 0 (ancien reglage garde dans {sauve.name})")


if __name__ == "__main__":
    main()
