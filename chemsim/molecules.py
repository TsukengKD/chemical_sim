"""Реагенты: шаблоны молекул и размещение их в реакторе.

Шаблоны — это только стартовые геометрии веществ, которые игрок кладёт в
реактор (как склянки с реактивами).  Геометрия уточняется минимизацией энергии
тем же силовым полем.  Что с ними произойдёт дальше, решает физика.
"""

from __future__ import annotations

import math

import numpy as np

from .elements import ELEMENTS

_T = 1.0 / math.sqrt(3.0)

# формула -> (русское название, символы, приблизительные координаты Å)
_RAW = {
    "H2": ("водород", ["H", "H"], [[0, 0, 0], [0.74, 0, 0]]),
    "O2": ("кислород", ["O", "O"], [[0, 0, 0], [1.21, 0, 0]]),
    "N2": ("азот", ["N", "N"], [[0, 0, 0], [1.10, 0, 0]]),
    "F2": ("фтор", ["F", "F"], [[0, 0, 0], [1.41, 0, 0]]),
    "Cl2": ("хлор", ["Cl", "Cl"], [[0, 0, 0], [1.99, 0, 0]]),
    "Br2": ("бром", ["Br", "Br"], [[0, 0, 0], [2.28, 0, 0]]),
    "I2": ("иод", ["I", "I"], [[0, 0, 0], [2.67, 0, 0]]),
    "HCl": ("хлороводород", ["H", "Cl"], [[0, 0, 0], [1.27, 0, 0]]),
    "HF": ("фтороводород", ["H", "F"], [[0, 0, 0], [0.92, 0, 0]]),
    "NaCl": ("хлорид натрия (пара)", ["Na", "Cl"], [[0, 0, 0], [2.36, 0, 0]]),
    "CO": ("угарный газ", ["C", "O"], [[0, 0, 0], [1.16, 0, 0]]),
    "NO": ("оксид азота(II)", ["N", "O"], [[0, 0, 0], [1.21, 0, 0]]),
    "H2O": ("вода", ["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
    "H2S": ("сероводород", ["S", "H", "H"], [[0, 0, 0], [1.34, 0, 0], [-0.3, 1.3, 0]]),
    "CO2": ("углекислый газ", ["C", "O", "O"], [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
    "HCN": ("циановодород", ["H", "C", "N"], [[-1.09, 0, 0], [0, 0, 0], [1.16, 0, 0]]),
    "NH3": ("аммиак", ["N", "H", "H", "H"],
            [[0, 0, 0.38], [0.94, 0, 0], [-0.47, 0.81, 0], [-0.47, -0.81, 0]]),
    "CH4": ("метан", ["C", "H", "H", "H", "H"],
            [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
             [0.63, -0.63, -0.63]]),
    "H2O2": ("пероксид водорода", ["H", "O", "O", "H"],
             [[-0.3, 0.9, 0], [0, 0, 0], [1.47, 0, 0], [1.77, 0.9, 0.5]]),
    "CH2O": ("формальдегид", ["C", "O", "H", "H"],
             [[0, 0, 0], [1.2, 0, 0], [-0.55, 0.93, 0], [-0.55, -0.93, 0]]),
    "C2H2": ("ацетилен", ["C", "C", "H", "H"],
             [[0, 0, 0], [1.2, 0, 0], [-1.06, 0, 0], [2.26, 0, 0]]),
    "C2H4": ("этилен", ["C", "C", "H", "H", "H", "H"],
             [[0, 0, 0], [1.34, 0, 0], [-0.55, 0.93, 0], [-0.55, -0.93, 0],
              [1.89, 0.93, 0], [1.89, -0.93, 0]]),
    "C2H6": ("этан", ["C", "C", "H", "H", "H", "H", "H", "H"],
             [[0, 0, 0], [1.53, 0, 0], [-0.4, 1.0, 0], [-0.4, -0.5, 0.87],
              [-0.4, -0.5, -0.87], [1.93, -1.0, 0], [1.93, 0.5, 0.87], [1.93, 0.5, -0.87]]),
    "CH3OH": ("метанол", ["C", "O", "H", "H", "H", "H"],
              [[0, 0, 0], [1.43, 0, 0], [-0.4, 1.0, 0], [-0.4, -0.5, 0.87],
               [-0.4, -0.5, -0.87], [1.75, 0.9, 0]]),
    "C6H6": ("бензол", ["C"] * 6 + ["H"] * 6,
             [[1.39 * math.cos(a), 1.39 * math.sin(a), 0] for a in np.arange(6) * math.pi / 3]
             + [[2.48 * math.cos(a), 2.48 * math.sin(a), 0] for a in np.arange(6) * math.pi / 3]),
}

_CACHE: dict = {}


def template_names():
    return list(_RAW.keys())


def template(formula: str, relax: bool = True):
    """(символы, координаты с центром масс в нуле) для реагента.

    formula — ключ шаблона ("H2O") или символ элемента ("Na" — одиночный атом).
    """
    if formula in ELEMENTS:
        return [formula], np.zeros((1, 3))
    if formula not in _RAW:
        raise KeyError(f"Нет шаблона для {formula!r}")
    if formula in _CACHE:
        s, x = _CACHE[formula]
        return list(s), x.copy()
    _, syms, pos = _RAW[formula]
    x = np.array(pos, dtype=np.float64)
    if relax:
        from .minimize import Evaluator, fire_minimize
        ev = Evaluator(syms)
        x, _, _ = fire_minimize(ev, x, fmax=5e-3, max_steps=5000)
    m = np.array([ELEMENTS[s].mass for s in syms])
    x = x - (m[:, None] * x).sum(axis=0) / m.sum()
    _CACHE[formula] = (list(syms), x.copy())
    return list(syms), x


def template_title(formula: str) -> str:
    if formula in ELEMENTS:
        return ELEMENTS[formula].name
    return _RAW[formula][0]


def random_rotation(rng) -> np.ndarray:
    """Равномерно распределённая матрица поворота (через случайный кватернион)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
    ])


def place(sim, formula: str, count: int = 1, region=None, temperature=None,
          min_dist: float = 2.3, max_tries: int = 200, vdw_frac: float = 0.8):
    """Поместить count молекул в случайные свободные места области region.

    region = (x0, y0, x1, y1) в Å (по z — вся толщина ящика); None — весь ящик.
    Возвращает число размещённых молекул.
    """
    syms, x0 = template(formula)
    from .params import P_X6
    new_t = np.array([sim.params.index(sy) for sy in syms], dtype=np.int64)
    # минимальное расстояние до чужих атомов: не ближе 0.8 ван-дер-ваальсова
    # расстояния пары (иначе энергия перекрытия сразу разогреет газ)
    xv = sim.params.pairpar[:, :, P_X6] ** (1.0 / 6.0)
    margin = sim.dw + 0.6
    box = sim.box
    if region is None:
        region = (0.0, 0.0, box[0], box[1])
    rx0, ry0, rx1, ry1 = region
    placed = 0
    new_pos = []
    new_sym = []
    existing = sim.pos.copy()
    existing_t = sim.typ.copy()
    for _ in range(count):
        ok = False
        for _t in range(max_tries):
            R = random_rotation(sim.rng)
            xyz = x0 @ R.T
            lo = xyz.min(axis=0)
            hi = xyz.max(axis=0)
            c = np.empty(3)
            span_ok = True
            bounds = [(max(rx0, margin), min(rx1, box[0] - margin)),
                      (max(ry0, margin), min(ry1, box[1] - margin)),
                      (margin, box[2] - margin)]
            for k in range(3):
                a, b = bounds[k][0] - lo[k], bounds[k][1] - hi[k]
                if b < a:
                    span_ok = False
                    break
                c[k] = sim.rng.uniform(a, b)
            if not span_ok:
                continue
            cand = xyz + c
            others = existing if not new_pos else np.vstack([existing] + new_pos)
            if len(others):
                other_t = existing_t if not new_pos else np.concatenate(
                    [existing_t] + [new_t] * len(new_pos))
                d = cand[:, None, :] - others[None, :, :]
                lim = np.maximum(min_dist, vdw_frac * xv[new_t[:, None], other_t[None, :]])
                if ((d * d).sum(axis=2) < lim * lim).any():
                    continue
            new_pos.append(cand)
            new_sym.extend(syms)
            ok = True
            break
        if ok:
            placed += 1
    if new_pos:
        sim.add_atoms(new_sym, np.vstack(new_pos), temperature=temperature)
    return placed
