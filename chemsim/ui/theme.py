"""Цвета, шрифты и простые виджеты интерфейса."""

from __future__ import annotations

import os

import pygame

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

BG = (16, 18, 24)
PANEL = (26, 29, 38)
PANEL_2 = (34, 38, 50)
BORDER = (54, 60, 78)
TEXT = (222, 226, 234)
DIM = (140, 147, 163)
ACCENT = (96, 165, 250)
ACCENT_2 = (250, 180, 80)
GOOD = (110, 210, 140)
BAD = (240, 100, 100)
REACTOR_BG = (10, 12, 17)

_FONTS: dict = {}


def font(size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
    key = (size, bold, mono)
    if key not in _FONTS:
        name = "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        path = os.path.join(FONT_DIR, name)
        try:
            _FONTS[key] = pygame.font.Font(path, size)
        except (OSError, FileNotFoundError):
            _FONTS[key] = pygame.font.SysFont("dejavusans,segoeui,arial", size, bold=bold)
    return _FONTS[key]


def text(surf, s, pos, size=14, color=TEXT, bold=False, mono=False, anchor="topleft"):
    img = font(size, bold, mono).render(str(s), True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def wrap(s: str, size: int, width: int, bold=False):
    """Разбить текст на строки по ширине (пикселей)."""
    f = font(size, bold)
    lines = []
    for para in s.split("\n"):
        words = para.split(" ")
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip()
            if f.size(cand)[0] <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def temperature_color(T: float):
    """Цвет «накала» для температуры (К): синий — холодно, белый — ~300 К,
    оранжевый — ~1500 К, красный/малиновый — очень горячо."""
    stops = [(50, (70, 110, 255)), (300, (200, 205, 215)), (800, (255, 200, 90)),
             (1500, (255, 140, 40)), (3000, (255, 60, 30)), (6000, (255, 40, 160))]
    if T <= stops[0][0]:
        return stops[0][1]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if T <= t1:
            f = (T - t0) / (t1 - t0)
            return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))
    return stops[-1][1]


def lighten(c, k):
    return tuple(min(255, int(v + (255 - v) * k)) for v in c)


def scale_color(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


class Button:
    def __init__(self, rect, label, on_click=None, *, color=None, toggled=False,
                 tooltip="", size=13, bold=False, group=None, value=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.color = color
        self.toggled = toggled
        self.tooltip = tooltip
        self.size = size
        self.bold = bold
        self.group = group
        self.value = value
        self.hover = False

    def draw(self, surf):
        base = self.color or PANEL_2
        if self.toggled:
            bg = lighten(base, 0.18) if self.color else (58, 90, 150)
            border = ACCENT
        else:
            bg = lighten(base, 0.08) if self.hover else base
            border = BORDER if not self.hover else lighten(BORDER, 0.3)
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, border, self.rect, width=2 if self.toggled else 1, border_radius=6)
        fg = TEXT
        if self.color:
            lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
            fg = (15, 15, 20) if lum > 150 else (240, 240, 245)
        text(surf, self.label, self.rect.center, self.size, fg, bold=self.bold, anchor="center")

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click(self)
                return True
        return False
