"""
Trouver la fenetre du jeu, sa zone utile, et savoir si elle a le focus (Windows).

Le test sur le titre est strict : chercher "minecraft" dans le titre attraperait
aussi le launcher, un explorateur ouvert sur ce projet ou un onglet de navigateur,
et les clics de l'agent partiraient dans la mauvaise fenetre.
"""
from __future__ import annotations

import ctypes
import re
from ctypes import wintypes

# "Minecraft 1.19.4", "Minecraft* 1.19.4 - Singleplayer"
GAME_TITLE = re.compile(r"^Minecraft\*?\s+\d")


def is_game_title(title: str | None) -> bool:
    return bool(GAME_TITLE.match(title or ""))


def _set_dpi_aware() -> None:
    """Sans ca, Windows renvoie des coordonnees mises a l'echelle, fausses pour la capture."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass


def find_game_window() -> tuple[int, str] | None:
    """(hwnd, titre) de la fenetre du jeu, ou None."""
    _set_dpi_aware()
    user32 = ctypes.windll.user32
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if is_game_title(buf.value):
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def client_box(hwnd: int) -> dict:
    """Zone de rendu du jeu en coordonnees ecran, sans bordure ni barre de titre."""
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    corner = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(corner))
    return {"left": corner.x, "top": corner.y, "width": rect.right, "height": rect.bottom}


def is_foreground(hwnd: int) -> bool:
    return ctypes.windll.user32.GetForegroundWindow() == hwnd
