"""
La verite du jeu, pour noter le modele. Le modele, lui, ne voit jamais que l'ecran.

Un datapack calcule a chaque tick : le bloc dans le viseur, sa distance, le bloc qui vient
d'etre casse et le butin ramasse. Un datapack ne peut pas ecrire de fichier : il envoie une
ligne dans le chat, le jeu la recopie dans logs/latest.log, et Python lit ce journal en direct.
La taille du chat est reglee a 0 pour que ces lignes n'apparaissent jamais sur les images.

Ligne : [CHAT] DC <tick> <bloc vise> <distance en 1/10 de bloc> <bloc casse> <butin>
Bloc  : 0 = rien a moins de 8 blocs, 99 = bloc hors de la table, sinon indice dans BLOCS + 1.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

# nom, durete, outil efficace. Ajouter en fin de liste seulement : l'indice est le numero du bloc
# dans les sessions deja enregistrees.
BLOCS = [
    ("stone", 1.5, "pioche"), ("granite", 1.5, "pioche"), ("diorite", 1.5, "pioche"),
    ("andesite", 1.5, "pioche"), ("tuff", 1.5, "pioche"), ("deepslate", 3.0, "pioche"),
    ("cobblestone", 2.0, "pioche"), ("netherrack", 0.4, "pioche"), ("obsidian", 50.0, "pioche"),
    ("coal_ore", 3.0, "pioche"), ("iron_ore", 3.0, "pioche"), ("copper_ore", 3.0, "pioche"),
    ("gold_ore", 3.0, "pioche"), ("redstone_ore", 3.0, "pioche"), ("lapis_ore", 3.0, "pioche"),
    ("diamond_ore", 3.0, "pioche"), ("emerald_ore", 3.0, "pioche"),
    ("dirt", 0.5, "pelle"), ("gravel", 0.6, "pelle"), ("sand", 0.5, "pelle"),
    ("oak_planks", 2.0, "hache"), ("pumpkin", 1.0, "hache"),
    # surface : les arbres
    ("oak_log", 2.0, "hache"), ("birch_log", 2.0, "hache"), ("spruce_log", 2.0, "hache"),
    ("dark_oak_log", 2.0, "hache"), ("jungle_log", 2.0, "hache"), ("acacia_log", 2.0, "hache"),
    ("mangrove_log", 2.0, "hache"), ("oak_leaves", 0.2, None), ("birch_leaves", 0.2, None),
    ("grass_block", 0.6, "pelle"),
]
TRONCS = [b[0] for b in BLOCS if b[0].endswith("_log")]
AUTRE, PORTEE_MAX, PORTEE = 99, 80, 45      # distances en 1/10 de bloc, portee du joueur : 4,5
BUTIN = ["cobblestone", "coal", "raw_iron", "raw_copper", "raw_gold", "redstone", "lapis_lazuli",
         "diamond", "emerald", "dirt", "gravel", "flint", "sand", "granite", "diorite", "andesite",
         "tuff", "netherrack", "obsidian", "oak_planks", "pumpkin", "cobbled_deepslate"] + TRONCS

# Salles d'examen : meme geste, duree de cassage differente (de 2 a 188 ticks a la pioche en diamant).
# Pas de sable ni de gravier : ils tomberaient du plafond.
SALLES = ["netherrack", "stone", "cobblestone", "coal_ore", "dirt", "diamond_ore", "pumpkin",
          "oak_planks", "obsidian"]
TRUTH_COLS = ("tick", "bloc", "distance", "casse", "butin")
LIGNE = re.compile(rb"\[CHAT\] DC (\d+) (\d+) (\d+) (\d+) (\d+)")


def bloc_id(nom: str) -> int:
    return [b[0] for b in BLOCS].index(nom) + 1


def bloc_nom(i: int) -> str:
    if i == 0:
        return "rien"
    return BLOCS[i - 1][0] if 1 <= i <= len(BLOCS) else "autre"


def ticks_theorie(nom: str, outil: str = "pioche") -> int:
    """Ticks pour casser le bloc avec un outil en diamant, sans enchantement, les pieds au sol.
    Regle du jeu : progres par tick = vitesse de l'outil / durete / 30 (vitesse 8 si l'outil
    est le bon, 1 sinon)."""
    _, durete, efficace = BLOCS[bloc_id(nom) - 1]
    return math.ceil(round(durete * 30 / (8 if efficace == outil else 1), 6))


def salle_de(i: int, mode: str):
    """Une session sur deux en terrain naturel, l'autre dans une salle d'un seul type de bloc."""
    if mode == "naturel" or i % 2 == 0:
        return None
    return SALLES[(i // 2) % len(SALLES)]


def fichiers_du_pack() -> dict[str, str]:
    score = lambda nom: {"score": {"name": f"#{nom}", "objective": "dc"}}  # noqa: E731
    message = ["DC "]
    for nom in ("gt", "bloc", "d", "casse"):
        message += [score(nom), " "]
    message.append(score("butin"))
    noms = list(enumerate((b[0] for b in BLOCS), start=1))
    ns = "data/dreamcraft/functions"
    return {
        "pack.mcmeta": json.dumps({"pack": {"pack_format": 12, "description": "DREAMCRAFT : verite du jeu"}}),
        "data/minecraft/tags/functions/load.json": json.dumps({"values": ["dreamcraft:charge"]}),
        "data/minecraft/tags/functions/tick.json": json.dumps({"values": ["dreamcraft:tick"]}),
        "data/dreamcraft/tags/blocks/traverse.json": json.dumps(
            {"values": [f"minecraft:{b}" for b in ("air", "cave_air", "void_air", "water", "lava")]}),
        "data/dreamcraft/tags/items/butin.json": json.dumps(
            {"values": [f"minecraft:{i}" for i in BUTIN]}),
        f"{ns}/charge.mcfunction": "\n".join(
            ["scoreboard objectives add dc dummy"]
            + [f"scoreboard objectives add m{i} minecraft.mined:minecraft.{n}" for i, n in noms]) + "\n",
        f"{ns}/tick.mcfunction": "\n".join(
            # anchored feet : sinon chaque pas du rayon rajouterait la hauteur des yeux
            ["execute as @a[limit=1] at @s anchored eyes positioned ^ ^ ^ anchored feet "
             "run function dreamcraft:joueur"]) + "\n",
        f"{ns}/joueur.mcfunction": "\n".join(
            ["execute store result score #gt dc run time query gametime",
             "scoreboard players set #bloc dc 0",
             "scoreboard players set #d dc 0",
             "scoreboard players set #casse dc 0",
             "function dreamcraft:rayon"]
            + [f"execute if score @s m{i} matches 1.. run scoreboard players set #casse dc {i}" for i, _ in noms]
            + [f"scoreboard players set @s m{i} 0" for i, _ in noms]
            + ["scoreboard players set #butin dc 0",      # /clear echoue si l'inventaire est vide
               "execute store result score #butin dc run clear @s #dreamcraft:butin 0",
               f"tellraw @s {json.dumps(message)}"]) + "\n",
        f"{ns}/rayon.mcfunction": "\n".join(
            ["execute unless block ~ ~ ~ #dreamcraft:traverse run function dreamcraft:identifie",
             "execute if score #bloc dc matches 0 run scoreboard players add #d dc 1",
             f"execute if score #bloc dc matches 0 if score #d dc matches ..{PORTEE_MAX - 1} "
             "positioned ^ ^ ^0.1 run function dreamcraft:rayon"]) + "\n",
        f"{ns}/identifie.mcfunction": "\n".join(
            [f"scoreboard players set #bloc dc {AUTRE}"]
            + [f"execute if block ~ ~ ~ minecraft:{n} run scoreboard players set #bloc dc {i}"
               for i, n in noms]) + "\n",
    }


def dossier_jeu() -> Path:
    return Path(os.environ.get("DC_MC_DIR", Path(os.environ.get("APPDATA", "")) / ".minecraft"))


def installe_pack(monde: str, jeu: Path | None = None) -> Path:
    """Ecrit le datapack dans un monde. Le jeu le charge a l'ouverture du monde."""
    racine = (jeu or dossier_jeu()) / "saves" / monde / "datapacks" / "dreamcraft_verite"
    for chemin, texte in fichiers_du_pack().items():
        p = racine / chemin
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(texte.encode("ascii"))
    return racine


def chat_masque(jeu: Path | None = None) -> bool:
    """Taille du chat a 0, sinon les lignes de verite salissent les images."""
    texte = ((jeu or dossier_jeu()) / "options.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^chatScale:([\d.]+)", texte, re.M)
    return bool(m) and float(m.group(1)) == 0.0


class JournalJeu:
    """Lit les lignes de verite au fur et a mesure que le jeu les ecrit."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(path, "rb")
        self.f.seek(0, os.SEEK_END)
        self.reste = b""
        self.etat = np.array([-1, 0, 0, 0, 0], dtype=np.int64)
        self.lignes = 0

    def _nouveau_fichier(self) -> bool:
        """A minuit, le jeu archive latest.log et en recommence un : il faut le rouvrir au debut
        (arene du 10/11 : la verite s'est figee a 00:00, cinq salles refusees a tort)."""
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        return st.st_size < self.f.tell() or st.st_ino != os.fstat(self.f.fileno()).st_ino

    def lire(self) -> np.ndarray:
        """Le dernier etat connu. `casse` couvre tous les ticks recus depuis l'appel precedent :
        un bloc casse n'est jamais perdu, meme si deux ticks arrivent en meme temps."""
        data = self.reste + self.f.read()
        if self._nouveau_fichier():
            self.f.close()
            self.f = open(self.path, "rb")
            data, self.reste = data + self.f.read(), b""
        complet, _, self.reste = data.rpartition(b"\n")
        casse = 0
        for m in LIGNE.finditer(complet):
            self.etat[:] = [int(x) for x in m.groups()]
            casse = int(self.etat[3]) or casse
            self.lignes += 1
        out = self.etat.copy()
        out[3] = casse
        return out

    def close(self):
        self.f.close()


def blocs_casses(truth: np.ndarray) -> int:
    return int((truth[:, 3] > 0).sum())


def troncs_casses(truth: np.ndarray) -> int:
    return int(np.isin(truth[:, 3], [bloc_id(n) for n in TRONCS]).sum())
