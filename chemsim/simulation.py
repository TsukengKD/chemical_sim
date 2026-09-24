"""Состояние «реактора» и управление моделированием."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .elements import ELEMENTS, SYMBOLS
from .kernel import compute_forces, N_EBREAK, N_ATOMOUT, A_Q, A_SPARE
from .md import (build_nlist, run_md, wall_forces, kinetic_energy, seed_rng,
                 N_STATS, ST_EPOT, ST_EWALL, ST_WALLF, ST_HEAT, ST_STEPS, ST_ESCAPES,
                 ST_TIME, ST_RETRIES)
from .params import default_params, ForceFieldParams, E_MASS
from .units import KB, FORCE_TO_ACC, KJMOL_A3_TO_ATM

# режимы теплообмена
ISOLATED = "isolated"   # изолированная система: полная энергия сохраняется
WALLS = "walls"         # стенки при заданной температуре (теплообмен через удары)
BATH = "bath"           # термостат во всём объёме (быстрое выравнивание T)

WALL_NAMES = ["left", "right", "bottom", "top", "back", "front"]


@dataclass
class Stats:
    time: float = 0.0          # фс
    epot: float = 0.0          # кДж/моль
    ekin: float = 0.0
    ewall: float = 0.0
    temperature: float = 0.0   # К
    pressure_atm: float = 0.0
    heat_in: float = 0.0       # теплота, полученная от стенок/термостата, кДж/моль
    escapes: int = 0


class Simulation:
    """Реактор: атомы в ящике со стенками.

    Координаты в Å, скорости в Å/фс, энергии в кДж/моль, время в фс.
    Ось y экрана — «вертикаль»: нижняя стенка (bottom) — y = 0.
    """

    def __init__(self, box=(60.0, 40.0, 8.0), params: ForceFieldParams | None = None,
                 dt: float = 0.25, seed: int | None = None):
        self.params = params or default_params()
        self.box = np.array(box, dtype=np.float64)
        self.dt = dt            # максимальный шаг, фс
        self.dx_max = 0.025     # максимальное смещение атома за шаг, Å
        self.e_tol = 2.0        # допустимое изменение полной энергии за шаг, кДж/моль
        self.retries = 0        # сколько шагов пришлось повторить с меньшим dt
        self.skin = 0.8
        self.rlist = self.params.settings.r_cut + self.skin
        self.kw = 200.0      # жёсткость стенки, кДж/моль/Å³
        self.dw = 1.0        # толщина «мягкого» слоя стенки, Å
        self.mode = WALLS
        self.wall_T = np.full(6, 300.0)
        self.bath_T = 300.0
        self.gamma_wall = 0.02   # 1/фс
        self.gamma_bath = 0.005  # 1/фс
        self.rng = np.random.default_rng(seed)
        seed_rng(int(self.rng.integers(0, 2**31 - 1)))
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.typ = np.zeros(0, dtype=np.int64)
        self.invm = np.zeros(0)
        self.forces = np.zeros((0, 3))
        self.atomout = np.zeros((0, N_ATOMOUT))
        self.ebreak = np.zeros(N_EBREAK)
        self._cap = 1024
        self.pi = np.zeros(self._cap, dtype=np.int64)
        self.pj = np.zeros(self._cap, dtype=np.int64)
        self.bond_order = np.zeros(self._cap)
        self.npair = 0
        self.ref_pos = np.zeros((0, 3))
        self.time = 0.0
        self.epot = 0.0
        self.ewall = 0.0
        self.heat_in = 0.0
        self.escapes = 0
        self._press_acc = 0.0
        self._press_steps = 0.0
        self.pressure_atm = 0.0
        self._stats = np.zeros(N_STATS)
        self.version = 0     # растёт при изменении состава (для анализа)

    # ------------------------------------------------------------ состав
    @property
    def n(self) -> int:
        return len(self.typ)

    @property
    def symbols(self):
        return [SYMBOLS[t] for t in self.typ]

    def masses(self):
        return self.params.elempar[self.typ, E_MASS]

    def add_atoms(self, symbols, positions, velocities=None, temperature=None):
        symbols = list(symbols)
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        typ = np.array([self.params.index(s) for s in symbols], dtype=np.int64)
        m = self.params.elempar[typ, E_MASS]
        if velocities is None:
            T = self.bath_T if temperature is None else temperature
            velocities = self.maxwell(m, T)
        velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
        self.pos = np.vstack([self.pos, positions])
        self.vel = np.vstack([self.vel, velocities])
        self.typ = np.concatenate([self.typ, typ])
        self.invm = 1.0 / self.params.elempar[self.typ, E_MASS]
        self._structure_changed()
        return np.arange(self.n - len(typ), self.n)

    def remove_atoms(self, indices):
        keep = np.ones(self.n, dtype=bool)
        keep[np.asarray(indices, dtype=np.int64)] = False
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.typ = self.typ[keep]
        self.invm = self.invm[keep]
        self._structure_changed()

    def clear(self):
        self.remove_atoms(np.arange(self.n))
        self.time = 0.0
        self.heat_in = 0.0

    def maxwell(self, masses, T):
        """Скорости из распределения Максвелла при температуре T."""
        masses = np.asarray(masses, dtype=np.float64)
        sig = np.sqrt(KB * T / masses * FORCE_TO_ACC)
        return self.rng.normal(size=(len(masses), 3)) * sig[:, None]

    def _structure_changed(self):
        self.version += 1
        self.forces = np.zeros((self.n, 3))
        self.atomout = np.zeros((self.n, N_ATOMOUT))
        self._rebuild()
        self._compute_forces()

    # ------------------------------------------------------------ соседи/силы
    def _rebuild(self):
        while True:
            n = build_nlist(self.pos, self.rlist, self.pi, self.pj)
            if n <= self._cap:
                break
            self._cap = int(n * 1.5) + 64
            self.pi = np.zeros(self._cap, dtype=np.int64)
            self.pj = np.zeros(self._cap, dtype=np.int64)
            self.bond_order = np.zeros(self._cap)
        self.npair = n
        self.ref_pos = self.pos.copy()

    def _compute_forces(self):
        p = self.params
        if self.n == 0:
            self.epot = self.ewall = 0.0
            return
        self.epot = compute_forces(self.pos, self.typ, self.pi, self.pj, self.npair,
                                   p.elempar, p.pairpar, p.scalars, self.forces,
                                   self.bond_order, self.ebreak, self.atomout)
        self.ewall, _ = wall_forces(self.pos, self.box, self.kw, self.dw, self.forces)

    # ------------------------------------------------------------ динамика
    def step(self, nsteps: int = 1):
        """Выполнить nsteps шагов интегрирования."""
        if self.n == 0:
            self.time += nsteps * self.dt
            return
        p = self.params
        if self.mode == ISOLATED:
            gw, gb = 0.0, 0.0
        elif self.mode == WALLS:
            gw, gb = self.gamma_wall, 0.0
        else:
            gw, gb = 0.0, self.gamma_bath
        remaining = nsteps
        wallf = 0.0
        elapsed = 0.0
        while remaining > 0:
            done = run_md(self.pos, self.vel, self.typ, self.invm, self.pi, self.pj,
                          self.npair, p.elempar, p.pairpar, p.scalars, self.forces,
                          self.bond_order, self.ebreak, self.atomout, self.box, self.kw,
                          self.dw, self.wall_T, gw, self.bath_T, gb, self.dt, self.dx_max,
                          self.e_tol, remaining, self.ref_pos, 0.5 * self.skin, self._stats)
            if done > 0:
                self.epot = self._stats[ST_EPOT]
                self.ewall = self._stats[ST_EWALL]
                self.heat_in += self._stats[ST_HEAT]
                self.escapes += int(self._stats[ST_ESCAPES])
                self.retries += int(self._stats[ST_RETRIES])
                wallf += self._stats[ST_WALLF]
                elapsed += self._stats[ST_TIME]
            remaining -= done
            if remaining > 0:
                self._rebuild()
                # после перестройки списка силы те же (все пары внутри r_cut учтены)
        self.time += elapsed
        # давление = средняя по времени сила на стенки / площадь стенок
        area = 2.0 * (self.box[0] * self.box[1] + self.box[0] * self.box[2]
                      + self.box[1] * self.box[2])
        self._press_acc += wallf / area
        self._press_steps += elapsed
        if self._press_steps >= 100.0:
            self.pressure_atm = self._press_acc / self._press_steps * KJMOL_A3_TO_ATM
            self._press_acc = 0.0
            self._press_steps = 0.0

    # ------------------------------------------------------------ воздействия
    def set_mode(self, mode: str):
        assert mode in (ISOLATED, WALLS, BATH)
        self.mode = mode

    def set_temperature(self, T: float, walls=None):
        """Температура стенок (всех или выбранных) и термостата."""
        if walls is None:
            self.wall_T[:] = T
            self.bath_T = T
        else:
            for w in walls:
                self.wall_T[WALL_NAMES.index(w)] = T

    def spark(self, center, radius: float, T: float):
        """«Искра»: атомам в шаре задаются тепловые скорости при температуре T."""
        if self.n == 0:
            return 0
        c = np.asarray(center, dtype=np.float64)
        d = self.pos[:, :len(c)] - c[None, :]
        sel = np.nonzero((d * d).sum(axis=1) < radius * radius)[0]
        if len(sel):
            m = self.masses()[sel]
            e0 = kinetic_energy(self.vel[sel], self.invm[sel])
            self.vel[sel] = self.maxwell(m, T)
            self.heat_in += kinetic_energy(self.vel[sel], self.invm[sel]) - e0
        return len(sel)

    def scale_heat(self, factor: float):
        """Умножить все скорости на factor (быстрое нагревание/охлаждение)."""
        e0 = self.ekin
        self.vel *= factor
        self.heat_in += self.ekin - e0

    def set_box(self, box):
        """Поршень: изменить размеры ящика (работа над газом совершается стенками)."""
        self.box = np.array(box, dtype=np.float64)
        self._compute_forces()

    # ------------------------------------------------------------ измерения
    @property
    def ekin(self) -> float:
        return float(kinetic_energy(self.vel, self.invm)) if self.n else 0.0

    @property
    def temperature(self) -> float:
        if self.n == 0:
            return 0.0
        return 2.0 * self.ekin / (3.0 * self.n * KB)

    @property
    def etotal(self) -> float:
        return self.epot + self.ewall + self.ekin

    @property
    def charges(self):
        return self.atomout[:, A_Q]

    @property
    def free_valence(self):
        return self.atomout[:, A_SPARE]

    def bonds(self, threshold: float = 0.5):
        """Список связей (i, j, порядок) с порядком ≥ threshold."""
        bo = self.bond_order[:self.npair]
        sel = np.nonzero(bo >= threshold)[0]
        return self.pi[sel], self.pj[sel], bo[sel]

    def stats(self) -> Stats:
        return Stats(self.time, self.epot, self.ekin, self.ewall, self.temperature,
                     self.pressure_atm, self.heat_in, self.escapes)

    def volume(self) -> float:
        return float(np.prod(self.box))
