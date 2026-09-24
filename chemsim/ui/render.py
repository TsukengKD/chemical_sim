"""Отрисовка реактора: атомы, связи, стенки, эффекты."""

from __future__ import annotations

import math
import time

import numpy as np
import pygame

from ..elements import ELEMENT_LIST
from . import theme as th

_COLORS = np.array([e.color for e in ELEMENT_LIST], dtype=np.float64)
# экранный радиус атома (Å): «шарик» чуть меньше ковалентного радиуса + константа
_RADII = np.array([0.36 + 0.50 * e.r1 for e in ELEMENT_LIST], dtype=np.float64)
_SYMBOLS = [e.symbol for e in ELEMENT_LIST]


class ReactorView:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.scale = 10.0
        self.ox = 0.0
        self.oy = 0.0
        self.box = np.array([60.0, 40.0, 8.0])
        self.show_bonds = True
        self.color_by_charge = False
        self.show_labels = True
        self.sparks = []        # (x, y, радиус Å, время начала, цвет)

    # ------------------------------------------------------------ геометрия
    def fit(self, box):
        self.box = np.array(box, dtype=np.float64)
        pad = 14
        w = self.rect.width - 2 * pad
        h = self.rect.height - 2 * pad
        self.scale = min(w / box[0], h / box[1])
        self.ox = self.rect.x + pad + (w - box[0] * self.scale) / 2
        self.oy = self.rect.y + pad + (h - box[1] * self.scale) / 2

    def to_screen(self, x, y):
        return self.ox + x * self.scale, self.oy + (self.box[1] - y) * self.scale

    def to_world(self, sx, sy):
        return (sx - self.ox) / self.scale, self.box[1] - (sy - self.oy) / self.scale

    def contains(self, pos):
        return self.rect.collidepoint(pos)

    def pick(self, sim, pos, max_px=14):
        if sim.n == 0:
            return None
        x, y = self.to_world(*pos)
        d = (sim.pos[:, 0] - x) ** 2 + (sim.pos[:, 1] - y) ** 2
        k = int(np.argmin(d))
        if math.sqrt(d[k]) * self.scale <= max_px:
            return k
        return None

    def add_spark(self, x, y, radius, color=(255, 240, 160)):
        self.sparks.append((x, y, radius, time.time(), color))

    # ------------------------------------------------------------ рисование
    def draw(self, surf, sim, hover=None, brush=None):
        self.fit(sim.box)
        pygame.draw.rect(surf, th.REACTOR_BG, self.rect)
        self._draw_walls(surf, sim)
        clip = surf.get_clip()
        x0, y1 = self.to_screen(0, 0)
        x1, y0 = self.to_screen(sim.box[0], sim.box[1])
        surf.set_clip(pygame.Rect(int(x0), int(y0), int(x1 - x0) + 1, int(y1 - y0) + 1))
        if sim.n:
            sx = self.ox + sim.pos[:, 0] * self.scale
            sy = self.oy + (sim.box[1] - sim.pos[:, 1]) * self.scale
            depth = np.clip(sim.pos[:, 2] / max(sim.box[2], 1e-6), 0.0, 1.0)
            shade = 0.72 + 0.28 * depth
            if self.color_by_charge:
                q = np.clip(sim.charges / 0.8, -1.0, 1.0)
                base = np.empty((sim.n, 3))
                white = np.array([225.0, 225.0, 230.0])
                red = np.array([255.0, 70.0, 60.0])
                blue = np.array([70.0, 120.0, 255.0])
                pos_q = q > 0
                base[pos_q] = white + (red - white) * q[pos_q, None]
                base[~pos_q] = white + (blue - white) * (-q[~pos_q, None])
            else:
                base = _COLORS[sim.typ]
            cols = np.clip(base * shade[:, None], 0, 255).astype(int)
            rad = np.maximum(2.5, _RADII[sim.typ] * self.scale * (0.85 + 0.15 * depth))
            if self.show_bonds:
                self._draw_bonds(surf, sim, sx, sy, cols)
            order = np.argsort(sim.pos[:, 2])
            free = sim.free_valence
            show_lab = self.show_labels and self.scale >= 13
            for a in order:
                c = tuple(cols[a])
                r = float(rad[a])
                p = (float(sx[a]), float(sy[a]))
                pygame.draw.circle(surf, c, p, r)
                # блик — объём
                pygame.draw.circle(surf, th.lighten(c, 0.45), (p[0] - r * 0.3, p[1] - r * 0.3),
                                   max(1.0, r * 0.35))
                if free[a] > 0.5:
                    pygame.draw.circle(surf, (255, 235, 90), p, r + 2.5, width=1)
                if show_lab and r >= 9:
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    th.text(surf, _SYMBOLS[sim.typ[a]], p, max(9, int(r * 0.95)),
                            (20, 20, 25) if lum > 140 else (235, 235, 240), anchor="center")
            if hover is not None and hover < sim.n:
                pygame.draw.circle(surf, th.ACCENT, (float(sx[hover]), float(sy[hover])),
                                   float(rad[hover]) + 4, width=2)
        self._draw_sparks(surf)
        surf.set_clip(clip)
        if brush is not None:
            (bx, by), br, color = brush
            pygame.draw.circle(surf, color, (bx, by), br * self.scale, width=1)

    def _draw_walls(self, surf, sim):
        x0, y1 = self.to_screen(0, 0)
        x1, y0 = self.to_screen(sim.box[0], sim.box[1])
        T = sim.wall_T
        w = 5
        segs = [((x0, y0), (x0, y1), T[0]), ((x1, y0), (x1, y1), T[1]),
                ((x0, y1), (x1, y1), T[2]), ((x0, y0), (x1, y0), T[3])]
        for a, b, t in segs:
            col = th.temperature_color(t) if sim.mode == "walls" else (90, 96, 112)
            pygame.draw.line(surf, col, a, b, w)

    def _draw_bonds(self, surf, sim, sx, sy, cols):
        n = sim.npair
        if n == 0:
            return
        bo = sim.bond_order[:n]
        sel = np.nonzero(bo > 0.15)[0]
        if len(sel) == 0:
            return
        pi = sim.pi[:n][sel]
        pj = sim.pj[:n][sel]
        bo = bo[sel]
        width = max(2, int(self.scale * 0.13))
        for i, j, b in zip(pi, pj, bo):
            x1, y1, x2, y2 = sx[i], sy[i], sx[j], sy[j]
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy)
            if L < 1e-6 or L > 6 * self.scale:
                continue
            ci = cols[i]
            cj = cols[j]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if b < 0.5:
                # частичная связь (переходное состояние реакции)
                k = (b - 0.15) / 0.35
                col = th.scale_color(th.lighten(tuple((ci + cj) // 2), 0.3), 0.35 + 0.5 * k)
                pygame.draw.line(surf, col, (x1, y1), (x2, y2), 1)
                continue
            nl = int(min(3, max(1, round(b))))
            frac = b - math.floor(b)
            nx, ny = -dy / L, dx / L
            off = max(2.5, width * 1.3)
            lines = [0.0] if nl == 1 else ([-0.5, 0.5] if nl == 2 else [-1.0, 0.0, 1.0])
            if nl == 1 and 0.35 < frac < 0.65 and b > 1.0:
                lines = [-0.5, 0.5]    # ароматическая (полуторная) связь
            for s in lines:
                ox, oy = nx * off * s, ny * off * s
                ww = width if (nl == 1 or s == 0.0) else max(1, width - 1)
                pygame.draw.line(surf, th.lighten(tuple(ci), 0.15), (x1 + ox, y1 + oy),
                                 (mx + ox, my + oy), ww)
                pygame.draw.line(surf, th.lighten(tuple(cj), 0.15), (mx + ox, my + oy),
                                 (x2 + ox, y2 + oy), ww)

    def _draw_sparks(self, surf):
        now = time.time()
        keep = []
        for (x, y, r, t0, color) in self.sparks:
            age = now - t0
            if age > 0.6:
                continue
            keep.append((x, y, r, t0, color))
            k = 1.0 - age / 0.6
            px, py = self.to_screen(x, y)
            rr = r * self.scale * (0.6 + 0.6 * (1 - k))
            s = pygame.Surface((int(rr * 2 + 4), int(rr * 2 + 4)), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (int(140 * k),), (rr + 2, rr + 2), rr)
            pygame.draw.circle(s, (255, 255, 255, int(220 * k)), (rr + 2, rr + 2), rr * 0.35)
            surf.blit(s, (px - rr - 2, py - rr - 2))
        self.sparks = keep
