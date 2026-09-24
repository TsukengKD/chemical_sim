"""Минимизация энергии (алгоритм FIRE) и расчёт энергий молекул в вакууме.

Используется для проверки модели против эксперимента: равновесная геометрия,
энергии атомизации, теплоты реакций, барьеры.
"""

from __future__ import annotations

import numpy as np

from .kernel import compute_forces, all_pairs, N_EBREAK, N_ATOMOUT
from .params import default_params


class Evaluator:
    """Энергия и силы изолированной группы атомов (без стенок)."""

    def __init__(self, symbols, params=None):
        self.params = params or default_params()
        self.symbols = list(symbols)
        self.typ = np.array([self.params.index(s) for s in symbols], dtype=np.int64)
        n = len(self.typ)
        self.pi, self.pj = all_pairs(n)
        self.forces = np.zeros((n, 3))
        self.bo = np.zeros(len(self.pi))
        self.eb = np.zeros(N_EBREAK)
        self.ao = np.zeros((n, N_ATOMOUT))

    def __call__(self, pos):
        p = self.params
        e = compute_forces(np.ascontiguousarray(pos, dtype=np.float64), self.typ, self.pi,
                           self.pj, len(self.pi), p.elempar, p.pairpar, p.scalars,
                           self.forces, self.bo, self.eb, self.ao)
        return e, self.forces.copy()

    def bond_orders(self, pos):
        self(pos)
        out = {}
        for k in range(len(self.pi)):
            if self.bo[k] > 0.05:
                out[(int(self.pi[k]), int(self.pj[k]))] = float(self.bo[k])
        return out


def fire_minimize(evaluator, pos, fmax=1e-3, max_steps=20000, dt0=0.02, dtmax=0.2,
                  fixed=None, constraint=None, maxstep=0.1):
    """FIRE (Bitzek et al., PRL 97, 170201, 2006).  Возвращает (pos, E, число шагов).

    constraint(pos, forces) — необязательная функция, проецирующая силы
    (используется для поиска барьеров со связанными координатами).
    """
    x = np.array(pos, dtype=np.float64)
    v = np.zeros_like(x)
    dt = dt0
    alpha, n_pos = 0.1, 0
    e, f = evaluator(x)
    if constraint is not None:
        f = constraint(x, f)
    if fixed is not None:
        f[fixed] = 0.0
    for step in range(max_steps):
        if np.abs(f).max() < fmax:
            return x, e, step
        pw = np.vdot(f, v)
        if pw > 0:
            fn = np.linalg.norm(f)
            vn = np.linalg.norm(v)
            v = (1 - alpha) * v + alpha * f / (fn + 1e-30) * vn
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dtmax)
                alpha *= 0.99
        else:
            n_pos = 0
            dt *= 0.5
            alpha = 0.1
            v[:] = 0.0
        # единичная «масса»: шаг в Å, сила в кДж/моль/Å
        v += dt * f * 0.01
        dx = dt * v
        m = np.sqrt((dx * dx).sum(axis=1)).max()
        if m > maxstep:
            dx *= maxstep / m
        x += dx
        e, f = evaluator(x)
        if constraint is not None:
            f = constraint(x, f)
        if fixed is not None:
            f[fixed] = 0.0
    return x, e, max_steps


class _Restrained:
    """Энергия + гармоническое ограничение на координату реакции
    ξ = r(b,c) − r(a,b) (метод «протаскивания» по координате реакции)."""

    def __init__(self, ev, a, b, c, k=1.0e3):
        self.ev, self.a, self.b, self.c, self.k = ev, a, b, c, k
        self.xi0 = 0.0

    def xi(self, x):
        return np.linalg.norm(x[self.c] - x[self.b]) - np.linalg.norm(x[self.b] - x[self.a])

    def __call__(self, x):
        e, f = self.ev(x)
        a, b, c = self.a, self.b, self.c
        dab = x[b] - x[a]
        dbc = x[c] - x[b]
        rab = np.linalg.norm(dab)
        rbc = np.linalg.norm(dbc)
        dx = (rbc - rab) - self.xi0
        g = 2.0 * self.k * dx
        f = f.copy()
        # dξ/dx: +d(rbc) − d(rab)
        f[c] -= g * dbc / rbc
        f[b] += g * dbc / rbc
        f[b] += g * dab / rab
        f[a] -= g * dab / rab
        return e + self.k * dx * dx, f


def reaction_profile(symbols, pos, a, b, c, xi_values, fmax=5e-3, params=None):
    """Профиль минимальной энергии для A + B–C -> A–B + C.

    Возвращает список (ξ, E_без_ограничения, геометрия).  Максимум профиля
    относительно реагентов даёт оценку барьера реакции.
    """
    ev = Evaluator(symbols, params)
    rs = _Restrained(ev, a, b, c)
    x = np.array(pos, dtype=np.float64)
    out = []
    for xi0 in xi_values:
        rs.xi0 = xi0
        x, _, _ = fire_minimize(rs, x, fmax=fmax, max_steps=6000)
        e, _ = ev(x)
        out.append((rs.xi(x), e, x.copy()))
    return out
