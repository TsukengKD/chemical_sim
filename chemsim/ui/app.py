"""Интерактивный химический симулятор (pygame).

Управление — см. справку в игре (клавиша H) или README.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from .. import molecules
from ..analysis import ChemistryMonitor
from ..elements import ELEMENTS, ELEMENT_LIST
from ..molecules import place, template_names, template_title
from ..names import pretty_formula
from ..scenarios import SCENARIOS
from ..simulation import Simulation, ISOLATED, WALLS, BATH
from ..units import KB, FORCE_TO_ACC
from . import theme as th
from .render import ReactorView

LEFT_W = 262
RIGHT_W = 350
BOTTOM_H = 92
SPEEDS = [1, 2, 5, 10, 20, 40, 80, 160, 320, 640]
FRAME_BUDGET_S = 0.028       # не тратить на физику больше ~28 мс за кадр
SPARK_T = 25000.0
AMOUNTS = [1, 2, 5, 10, 20, 50]
TOOLS = [("add", "Добавить", "ЛКМ — положить вещество в реактор"),
         ("spark", "Искра", "ЛКМ — искра: сильный локальный нагрев (как разряд)"),
         ("heat", "Нагрев", "Держите ЛКМ — нагревать область"),
         ("cool", "Холод", "Держите ЛКМ — охлаждать область"),
         ("erase", "Ластик", "Держите ЛКМ — убрать атомы")]
MODES = [(ISOLATED, "Изолир.", "Изолированная система: энергия сохраняется,\n"
                               "тепло реакций остаётся в газе"),
         (WALLS, "Стенки", "Теплообмен со стенками заданной температуры\n(как реальный сосуд)"),
         (BATH, "Термостат", "Термостат во всём объёме: температура\nподдерживается постоянной")]
PERIODIC_ROWS = [["H", "He"], ["Li", "B", "C", "N", "O", "F", "Ne"],
                 ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"], ["K", "Ca", "Br", "I"]]

HELP_TEXT = """КАК ЭТО РАБОТАЕТ
Каждый атом — частица, движущаяся по законам Ньютона. Силы между атомами берутся из потенциальной энергии, построенной только из табличных свойств элементов: электроотрицательности (Полинг), ковалентных радиусов (Пюккё), энергий связей, ван-дер-ваальсовых параметров. Химическая связь — это не «рецепт», а яма потенциальной энергии: атомы с неиспользованными валентностями притягиваются, насыщенные — отталкиваются. Реакции идут, когда молекулы сталкиваются с энергией, достаточной для перестройки связей; выделившаяся энергия химических связей становится теплом (скоростью атомов).

ИНСТРУМЕНТЫ (левая панель)
Добавить — выбрать атом или молекулу и щёлкнуть в реакторе. Искра — канал плазмы электрического разряда (~25 000 К): молекулы в нём распадаются на радикалы. Нагрев/Холод — «горелка» и «холодильник» под курсором. Ластик — удалить атомы. ПКМ — всегда искра. Колесо мыши — размер кисти.

КЛАВИШИ
Пробел — пауза · +/− — скорость · [ ] — температура стенок (Shift — шаг 500 К) · M — режим теплообмена · ← → — поршень (объём) · Q — раскраска по зарядам · B — связи · L — подписи атомов · O — опыты · C — очистить · S — снимок экрана · H/F1 — справка · Esc — закрыть окно.

ОБОЗНАЧЕНИЯ
Линии — химические связи (одна/две/три — порядок связи, тонкая — рвущаяся или образующаяся связь в момент реакции). Жёлтое кольцо — радикал (атом со свободной валентностью). Цвет стенок — их температура. «•» в формуле — радикал.

ЧЕСТНЫЕ ОГРАНИЧЕНИЯ
Это классическая (не квантовая) модель: нет возбуждённых состояний, света, спина (O₂ считается синглетной двойной связью), нулевых колебаний. Реакционные барьеры воспроизводятся полуколичественно. Время в реакторе — пикосекунды, плотность газа высокая (сотни атмосфер), поэтому всё происходит в миллиарды раз быстрее, чем в пробирке."""


class App:
    def __init__(self, width=1500, height=900, warmup_thread=True):
        pygame.init()
        pygame.display.set_caption("Химический симулятор — законы физики вместо рецептов")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.sim = Simulation(box=(60.0, 40.0, 8.0), seed=None)
        self.mon = ChemistryMonitor()
        self.view = ReactorView((LEFT_W, 0, 100, 100))
        self.state = "loading"
        self.loading_error = None
        self.tool = "add"
        self.selected = "H2"
        self.amount_idx = 2
        self.speed_idx = 6
        self.paused = False
        self.brush = 4.0
        self.overlay = None
        self.message = ""
        self.message_until = 0.0
        self.hover_atom = None
        self.mouse_down = False
        self.history = deque(maxlen=900)
        self.last_hist_t = -1e9
        self.target_lx = 60.0
        self.real_rate = 0.0
        self.frame_ms = 0.0
        self.step_cost = 2e-4          # оценка времени одного шага, с
        self.rate_mark = (time.perf_counter(), 0.0)
        self.steps_now = 0
        self.buttons: list[th.Button] = []
        self.scenario_rects = []
        self.current_scenario = None
        self.layout()
        if warmup_thread:
            threading.Thread(target=self._warmup, daemon=True).start()

    # ------------------------------------------------------------ подготовка
    def _warmup(self):
        """Первая компиляция ядра numba (~30 с при первом запуске, затем из кэша)."""
        try:
            s = Simulation(box=(20.0, 20.0, 8.0), seed=1)
            place(s, "H2O", 2)
            s.step(3)
            for name in template_names():
                molecules.template(name)
            if self.state == "loading":
                self.state = "ready"
        except Exception as exc:  # pragma: no cover
            self.loading_error = repr(exc)

    # ------------------------------------------------------------ раскладка
    def layout(self):
        W, H = self.screen.get_size()
        self.W, self.H = W, H
        self.view.rect = pygame.Rect(LEFT_W + 8, 8, W - LEFT_W - RIGHT_W - 16, H - BOTTOM_H - 16)
        self.buttons = []
        B = self.buttons
        # --- инструменты
        x0, y = 12, 44
        bw = (LEFT_W - 24 - 8) // 3
        for k, (key, label, tip) in enumerate(TOOLS):
            r = (x0 + (k % 3) * (bw + 4), y + (k // 3) * 32, bw, 28)
            B.append(th.Button(r, label, self._set_tool, toggled=(self.tool == key), tooltip=tip,
                               group="tool", value=key, size=12))
        # --- атомы (как в таблице Менделеева, укороченной)
        y = 138
        cw = (LEFT_W - 24) // 8
        for row in PERIODIC_ROWS:
            for k, sym in enumerate(row):
                el = ELEMENTS[sym]
                col = k if len(row) > 2 else (0 if k == 0 else 7)
                r = (x0 + col * cw, y, cw - 3, 30)
                B.append(th.Button(r, sym, self._select, color=el.color, bold=True, size=13,
                                   toggled=(self.selected == sym), group="species", value=sym,
                                   tooltip=f"{el.name}: валентность {el.valence}, χ = {el.chi}"))
            y += 34
        # --- молекулы
        y += 22
        names = template_names()
        mw = (LEFT_W - 24 - 8) // 3
        for k, f in enumerate(names):
            r = (x0 + (k % 3) * (mw + 4), y + (k // 3) * 29, mw, 26)
            B.append(th.Button(r, pretty_formula(f), self._select, size=12, group="species",
                               value=f, toggled=(self.selected == f), tooltip=template_title(f)))
        y += ((len(names) + 2) // 3) * 29 + 28
        # --- количество
        B.append(th.Button((x0, y, 32, 28), "−", lambda b: self._amount(-1), size=16))
        B.append(th.Button((LEFT_W - 12 - 32, y, 32, 28), "+", lambda b: self._amount(+1), size=16))
        self.amount_y = y
        # --- нижняя панель
        bx = self.view.rect.x
        by = H - BOTTOM_H + 8
        row2 = by + 42

        def add(w, label, cb, tip="", **kw):
            nonlocal bx
            b = th.Button((bx, by, w, 34), label, cb, tooltip=tip, **kw)
            B.append(b)
            bx += w + 6
            return b

        add(92, "Пауза" if not self.paused else "Пуск", self._toggle_pause, "Пробел",
            toggled=self.paused, group="pause")
        add(34, "−", lambda b: self._speed(-1), "Медленнее (−)", size=16)
        self.speed_x = bx
        bx += 118
        add(34, "+", lambda b: self._speed(+1), "Быстрее (+)", size=16)
        bx += 8
        for key, label, tip in MODES:
            add(88, label, self._set_mode, tip, toggled=(self.sim.mode == key), group="mode",
                value=key, size=12)
        bx += 8
        add(34, "−", lambda b: self._wall_temp(-100), "Холоднее ([)", size=16)
        self.temp_x = bx
        bx += 112
        add(34, "+", lambda b: self._wall_temp(+100), "Горячее (])", size=16)
        # второй ряд
        bx = self.view.rect.x
        by = row2
        add(150, "Горелка снизу", self._burner, "Нагреть только нижнюю стенку до 3000 К",
            group="burner", size=12)
        add(34, "◀", lambda b: self._piston(-4.0), "Поршень: сжать (←)", size=14)
        self.vol_x = bx
        bx += 118
        add(34, "▶", lambda b: self._piston(+4.0), "Поршень: расширить (→)", size=14)
        bx += 8
        add(96, "Опыты…", lambda b: self._open("scenarios"), "Готовые опыты (O)", size=13, bold=True)
        add(96, "Очистить", lambda b: self._clear(), "Убрать все атомы (C)", size=13)
        add(96, "Заряды", self._toggle_charge, "Раскраска по частичным зарядам (Q)",
            toggled=self.view.color_by_charge, group="charge", size=12)
        add(96, "Справка", lambda b: self._open("help"), "H / F1", size=13)

    # ------------------------------------------------------------ действия
    def notify(self, msg, seconds=4.0):
        self.message = msg
        self.message_until = time.time() + seconds

    def _set_tool(self, b):
        self.tool = b.value
        for x in self.buttons:
            if x.group == "tool":
                x.toggled = x.value == self.tool

    def _select(self, b):
        self.selected = b.value
        self._set_tool(type("B", (), {"value": "add"})())
        for x in self.buttons:
            if x.group == "species":
                x.toggled = x.value == self.selected

    def _amount(self, d):
        self.amount_idx = int(np.clip(self.amount_idx + d, 0, len(AMOUNTS) - 1))

    def _speed(self, d):
        self.speed_idx = int(np.clip(self.speed_idx + d, 0, len(SPEEDS) - 1))

    def _toggle_pause(self, b=None):
        self.paused = not self.paused
        for x in self.buttons:
            if x.group == "pause":
                x.toggled = self.paused
                x.label = "Пуск" if self.paused else "Пауза"

    def _set_mode(self, b):
        self.sim.set_mode(b.value)
        for x in self.buttons:
            if x.group == "mode":
                x.toggled = x.value == self.sim.mode
        self.notify(dict((k, t) for k, _, t in MODES)[b.value].replace("\n", " "))

    def _cycle_mode(self):
        keys = [m[0] for m in MODES]
        nxt = keys[(keys.index(self.sim.mode) + 1) % len(keys)]
        self._set_mode(type("B", (), {"value": nxt})())

    def _wall_temp(self, d):
        T = float(np.clip(np.max(self.sim.wall_T) + d, 20.0, 10000.0))
        self.sim.set_temperature(T)
        for x in self.buttons:
            if x.group == "burner":
                x.toggled = False

    def _burner(self, b):
        b.toggled = not b.toggled
        if b.toggled:
            self.sim.wall_T[2] = 3000.0
            if self.sim.mode != WALLS:
                self._set_mode(type("B", (), {"value": WALLS})())
            self.notify("Горелка: нижняя стенка нагрета до 3000 К")
        else:
            self.sim.wall_T[2] = self.sim.wall_T[3]

    def _piston(self, d):
        self.target_lx = float(np.clip(self.target_lx + d, 16.0, 140.0))

    def _clear(self):
        self.sim.clear()
        self.mon.reset()
        self.history.clear()
        self.last_hist_t = -1e9
        self.current_scenario = None
        self.notify("Реактор очищен")

    def _toggle_charge(self, b=None):
        self.view.color_by_charge = not self.view.color_by_charge
        for x in self.buttons:
            if x.group == "charge":
                x.toggled = self.view.color_by_charge

    def _open(self, what):
        self.overlay = None if self.overlay == what else what

    def load_scenario(self, sc):
        placed = sc.apply(self.sim)
        self.target_lx = sc.box[0]
        self.rate_mark = (time.perf_counter(), self.sim.time)
        self.mon.reset()
        self.history.clear()
        self.last_hist_t = -1e9
        self.current_scenario = sc
        for x in self.buttons:
            if x.group == "mode":
                x.toggled = x.value == self.sim.mode
            if x.group == "burner":
                x.toggled = False
        self.amount_idx = 2
        lost = [f"{f}" for (f, n), (_, want) in zip(placed, sc.reagents) if n < want]
        self.notify(sc.hint + (" (не всё поместилось: " + ", ".join(lost) + ")" if lost else ""), 8)

    # ------------------------------------------------------------ реактор: мышь
    def _apply_tool(self, pos, first):
        x, y = self.view.to_world(*pos)
        sim = self.sim
        if self.tool == "add" and first:
            n = AMOUNTS[self.amount_idx]
            syms, xyz = molecules.template(self.selected)
            size = float(np.ptp(xyz, axis=0).max()) if len(syms) > 1 else 1.0
            r = 2.0 + (1.4 * math.sqrt(n) + 0.8) * (size + 2.6) / 2
            T = float(np.max(sim.wall_T)) if sim.mode == WALLS else sim.bath_T
            got = place(sim, self.selected, n, region=(x - r, y - r, x + r, y + r), temperature=T)
            if got < n:
                self.notify(f"Поместилось {got} из {n}: мало места (увеличьте объём или кисть)")
        elif self.tool == "spark" and first:
            self._spark(x, y)
        elif self.tool in ("heat", "cool"):
            self._brush_temperature(x, y, 1.5 if self.tool == "heat" else 0.75)
        elif self.tool == "erase":
            if sim.n:
                d = (sim.pos[:, 0] - x) ** 2 + (sim.pos[:, 1] - y) ** 2
                sel = np.nonzero(d < self.brush ** 2)[0]
                if len(sel):
                    sim.remove_atoms(sel)

    def _spark(self, x, y):
        # искра = канал плазмы электрического разряда (10 000–30 000 К)
        n = self.sim.spark((x, y), max(3.0, self.brush), SPARK_T)
        self.view.add_spark(x, y, max(3.0, self.brush))
        if n == 0:
            self.notify("Искра в пустоте — рядом нет атомов")

    def _brush_temperature(self, x, y, factor):
        sim = self.sim
        if sim.n == 0:
            return
        d = (sim.pos[:, 0] - x) ** 2 + (sim.pos[:, 1] - y) ** 2
        sel = np.nonzero(d < self.brush ** 2)[0]
        if len(sel) == 0:
            return
        from ..md import kinetic_energy
        e0 = kinetic_energy(sim.vel[sel], sim.invm[sel])
        t_local = 2.0 * e0 / (3.0 * len(sel) * KB)
        # не остужать ниже 5 К и не греть выше 30 000 К
        if (factor > 1.0 and t_local > 30000.0) or (factor < 1.0 and t_local < 5.0):
            return
        sim.vel[sel] = sim.vel[sel] * math.sqrt(factor) ** 0.5
        e1 = kinetic_energy(sim.vel[sel], sim.invm[sel])
        sim.heat_in += e1 - e0
        self.view.add_spark(x, y, self.brush, (255, 120, 40) if factor > 1 else (80, 160, 255))

    # ------------------------------------------------------------ цикл
    def handle(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1100, event.w), max(700, event.h)),
                                                  pygame.RESIZABLE)
            self.layout()
            return True
        if self.state != "run":
            return True
        if event.type == pygame.KEYDOWN:
            return self._key(event)
        if self.overlay == "scenarios" and event.type == pygame.MOUSEBUTTONDOWN:
            for rect, sc in self.scenario_rects:
                if rect.collidepoint(event.pos):
                    self.load_scenario(sc)
                    self.overlay = None
                    return True
            self.overlay = None
            return True
        if self.overlay == "help" and event.type == pygame.MOUSEBUTTONDOWN:
            self.overlay = None
            return True
        for b in self.buttons:
            if b.handle(event):
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and self.view.contains(event.pos):
            if event.button == 1:
                self.mouse_down = True
                self._apply_tool(event.pos, True)
            elif event.button == 3:
                x, y = self.view.to_world(*event.pos)
                self._spark(x, y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.mouse_down = False
        elif event.type == pygame.MOUSEWHEEL:
            self.brush = float(np.clip(self.brush * (1.15 ** event.y), 1.5, 20.0))
        elif event.type == pygame.MOUSEMOTION:
            self.hover_atom = self.view.pick(self.sim, event.pos) if self.view.contains(event.pos) else None
        return True

    def _key(self, e):
        k = e.key
        shift = e.mod & pygame.KMOD_SHIFT
        if k == pygame.K_ESCAPE:
            if self.overlay:
                self.overlay = None
            else:
                return False
        elif k == pygame.K_SPACE:
            self._toggle_pause()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._speed(+1)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._speed(-1)
        elif k == pygame.K_RIGHTBRACKET:
            self._wall_temp(500 if shift else 100)
        elif k == pygame.K_LEFTBRACKET:
            self._wall_temp(-500 if shift else -100)
        elif k == pygame.K_m:
            self._cycle_mode()
        elif k == pygame.K_q:
            self._toggle_charge()
        elif k == pygame.K_b:
            self.view.show_bonds = not self.view.show_bonds
        elif k == pygame.K_l:
            self.view.show_labels = not self.view.show_labels
        elif k == pygame.K_o:
            self._open("scenarios")
        elif k in (pygame.K_h, pygame.K_F1):
            self._open("help")
        elif k == pygame.K_c:
            self._clear()
        elif k == pygame.K_LEFT:
            self._piston(-4.0)
        elif k == pygame.K_RIGHT:
            self._piston(+4.0)
        elif k == pygame.K_s:
            self.screenshot()
        return True

    def screenshot(self, path=None):
        os.makedirs("screenshots", exist_ok=True)
        path = path or os.path.join("screenshots", time.strftime("chemsim-%Y%m%d-%H%M%S.png"))
        pygame.image.save(self.screen, path)
        self.notify(f"Снимок сохранён: {path}")
        return path

    def update(self):
        if self.state == "ready":
            self.state = "run"
            self.load_scenario(SCENARIOS[0])
        if self.state != "run":
            return
        sim = self.sim
        # поршень — плавно
        if abs(sim.box[0] - self.target_lx) > 1e-6:
            step = float(np.clip(self.target_lx - sim.box[0], -0.25, 0.25))
            sim.set_box((sim.box[0] + step, sim.box[1], sim.box[2]))
        if self.mouse_down and self.tool in ("heat", "cool", "erase"):
            pos = pygame.mouse.get_pos()
            if self.view.contains(pos):
                self._apply_tool(pos, False)
        if not self.paused:
            # сколько шагов успеем за кадр (скорость не выше выбранной)
            want = SPEEDS[self.speed_idx]
            n = max(1, min(want, int(FRAME_BUDGET_S / max(self.step_cost, 1e-6))))
            t0 = time.perf_counter()
            sim.step(n)
            dt_real = time.perf_counter() - t0
            self.step_cost = 0.8 * self.step_cost + 0.2 * dt_real / n
            self.steps_now = n
        now = time.perf_counter()
        t_mark, s_mark = self.rate_mark
        if now - t_mark > 1.0:
            self.real_rate = (sim.time - s_mark) / (now - t_mark)
            self.rate_mark = (now, sim.time)
        self.mon.update(sim, record_history=False)
        if sim.time - self.last_hist_t >= 20.0 or not self.history:
            self.history.append((sim.time / 1000.0, sim.temperature, dict(self.mon.species)))
            self.last_hist_t = sim.time

    # ------------------------------------------------------------ рисование
    def draw(self):
        s = self.screen
        s.fill(th.BG)
        if self.state != "run":
            self._draw_loading()
            pygame.display.flip()
            return
        brush = None
        mp = pygame.mouse.get_pos()
        if self.view.contains(mp) and self.tool in ("heat", "cool", "erase", "spark"):
            col = {"heat": (255, 140, 60), "cool": (90, 170, 255), "erase": (200, 200, 200),
                   "spark": (255, 240, 150)}[self.tool]
            r = self.brush if self.tool != "spark" else max(3.0, self.brush)
            brush = (mp, r, col)
        self.view.draw(s, self.sim, self.hover_atom, brush)
        self._draw_left()
        self._draw_right()
        self._draw_bottom()
        for b in self.buttons:
            b.draw(s)
        self._draw_tooltip()
        if self.overlay == "help":
            self._draw_help()
        elif self.overlay == "scenarios":
            self._draw_scenarios()
        pygame.display.flip()

    def _panel(self, rect, title=None):
        pygame.draw.rect(self.screen, th.PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, th.BORDER, rect, width=1, border_radius=8)
        if title:
            th.text(self.screen, title, (rect.x + 12, rect.y + 8), 13, th.DIM, bold=True)

    def _draw_left(self):
        r = pygame.Rect(4, 4, LEFT_W - 8, self.H - 8)
        self._panel(r)
        th.text(self.screen, "ИНСТРУМЕНТ", (14, 22), 12, th.DIM, bold=True)
        th.text(self.screen, "АТОМЫ", (14, 116), 12, th.DIM, bold=True)
        mol_y = 138 + 34 * len(PERIODIC_ROWS) + 2
        th.text(self.screen, "МОЛЕКУЛЫ (реактивы)", (14, mol_y), 12, th.DIM, bold=True)
        y = self.amount_y
        th.text(self.screen, f"по {AMOUNTS[self.amount_idx]} шт. за щелчок", (LEFT_W // 2, y + 14),
                13, th.TEXT, anchor="center")
        # описание выбранного
        y += 44
        sel = self.selected
        if sel in ELEMENTS:
            el = ELEMENTS[sel]
            lines = [f"{el.name} ({el.symbol}), Z = {el.Z}",
                     f"масса {el.mass:.3f} а.е.м.",
                     f"электроотрицательность {el.chi or '—'}",
                     f"валентность {el.valence}" + (" (металл)" if el.metal else ""),
                     f"ковалентный радиус {el.r1:.2f} Å"]
        else:
            syms, _ = molecules.template(sel)
            lines = [f"{pretty_formula(sel)} — {template_title(sel)}", f"атомов: {len(syms)}"]
        th.text(self.screen, "ВЫБРАНО", (14, y), 12, th.DIM, bold=True)
        y += 20
        for ln in lines:
            for sub in th.wrap(ln, 13, LEFT_W - 28):
                th.text(self.screen, sub, (14, y), 13)
                y += 18
        y += 8
        th.text(self.screen, f"кисть: {self.brush:.1f} Å (колесо мыши)", (14, y), 12, th.DIM)

    def _draw_right(self):
        x = self.W - RIGHT_W + 4
        w = RIGHT_W - 8
        s = self.screen
        sim = self.sim
        # --- состояние
        r = pygame.Rect(x, 4, w, 206)
        self._panel(r, "СОСТОЯНИЕ РЕАКТОРА")
        T = sim.temperature
        rows = [
            ("Время", f"{sim.time / 1000.0:8.3f} пс"),
            ("Температура газа", f"{T:8.0f} К"),
            ("Давление", f"{sim.pressure_atm:8.0f} атм"),
            ("Объём", f"{sim.box[0]:.0f}×{sim.box[1]:.0f}×{sim.box[2]:.0f} Å"),
            ("Потенц. энергия", f"{sim.epot:9.0f} кДж/моль"),
            ("Кинет. энергия", f"{sim.ekin:9.0f} кДж/моль"),
            ("Полная энергия", f"{sim.etotal:9.0f} кДж/моль"),
            ("Тепло от стенок/рук", f"{sim.heat_in:+9.0f} кДж/моль"),
            ("Атомов / молекул", f"{sim.n} / {len(self.mon.molecules)}"),
        ]
        y = r.y + 30
        for k, v in rows:
            th.text(s, k, (x + 12, y), 13, th.DIM)
            col = th.temperature_color(T) if k.startswith("Темп") else th.TEXT
            th.text(s, v, (x + w - 12, y), 13, col, mono=True, anchor="topright")
            y += 19
        # --- вещества
        r2 = pygame.Rect(x, r.bottom + 6, w, 262)
        self._panel(r2, "ВЕЩЕСТВА В РЕАКТОРЕ")
        y = r2.y + 30
        for label, cnt, name in self.mon.summary(12):
            th.text(s, f"{cnt:4d}", (x + 12, y), 13, th.ACCENT, mono=True)
            th.text(s, label, (x + 58, y), 14, th.TEXT, bold=True)
            if name:
                nm = name if len(name) < 26 else name[:25] + "…"
                th.text(s, nm, (x + w - 12, y + 1), 12, th.DIM, anchor="topright")
            y += 19
        # --- реакции
        r3 = pygame.Rect(x, r2.bottom + 6, w, 196)
        self._panel(r3, "ЖУРНАЛ РЕАКЦИЙ")
        y = r3.y + 30
        evs = self.mon.events[-8:]
        if not evs:
            th.text(s, "пока реакций не было", (x + 12, y), 13, th.DIM)
        for ev in reversed(evs):
            eq = ev.equation()
            th.text(s, f"{ev.time / 1000:6.2f}", (x + 10, y + 1), 11, th.DIM, mono=True)
            f = th.font(13)
            while f.size(eq)[0] > w - 70 and len(eq) > 10:
                eq = eq[:-2]
            th.text(s, eq, (x + 60, y), 13)
            y += 20
        # --- график
        r4 = pygame.Rect(x, r3.bottom + 6, w, self.H - r3.bottom - 10)
        self._panel(r4, "ИСТОРИЯ")
        self._draw_chart(pygame.Rect(r4.x + 10, r4.y + 28, r4.width - 20, r4.height - 36))

    def _draw_chart(self, rect):
        s = self.screen
        if rect.height < 40 or len(self.history) < 2:
            return
        hist = list(self.history)
        t = np.array([h[0] for h in hist])
        T = np.array([h[1] for h in hist])
        t0, t1 = t[0], max(t[-1], t[0] + 1e-6)
        # температура (левая шкала)
        tmax = max(500.0, T.max() * 1.1)

        def px(tt, v, vmax):
            return (rect.x + (tt - t0) / (t1 - t0) * rect.width,
                    rect.bottom - v / vmax * rect.height)

        pygame.draw.line(s, th.BORDER, rect.bottomleft, rect.bottomright)
        pts = [px(a, b, tmax) for a, b in zip(t, T)]
        pygame.draw.lines(s, (255, 150, 70), False, pts, 2)
        th.text(s, f"T, К (макс {tmax:.0f})", (rect.x + 2, rect.y), 11, (255, 150, 70))
        # вещества (правая шкала)
        final = hist[-1][2]
        top = sorted(final, key=lambda k: -final[k])[:4]
        palette = [(96, 165, 250), (110, 210, 140), (240, 110, 200), (230, 220, 90)]
        cmax = max(1, max(max(h[2].get(sp, 0) for h in hist) for sp in top)) if top else 1
        for sp, col in zip(top, palette):
            pts = [px(h[0], h[2].get(sp, 0), cmax * 1.1) for h in hist]
            pygame.draw.lines(s, col, False, pts, 1)
        ly = rect.y + 16
        for sp, col in zip(top, palette):
            th.text(s, sp, (rect.right - 4, ly), 12, col, anchor="topright")
            ly += 15
        th.text(s, f"{t0:.1f} пс", (rect.x, rect.bottom + 1), 10, th.DIM)
        th.text(s, f"{t1:.1f} пс", (rect.right, rect.bottom + 1), 10, th.DIM, anchor="topright")

    def _draw_bottom(self):
        s = self.screen
        r = pygame.Rect(self.view.rect.x, self.H - BOTTOM_H + 2, self.view.rect.width, BOTTOM_H - 6)
        pygame.draw.rect(s, th.PANEL, r, border_radius=8)
        by = self.H - BOTTOM_H + 8
        sp = SPEEDS[self.speed_idx]
        lim = " (макс.)" if self.steps_now < sp and not self.paused else ""
        th.text(s, f"{sp} шаг/кадр{lim}", (self.speed_x + 57, by + 3), 12, th.TEXT, anchor="midtop")
        th.text(s, f"{self.real_rate / 1000:.2f} пс/с", (self.speed_x + 57, by + 18), 11, th.DIM,
                anchor="midtop")
        Tw = float(np.max(self.sim.wall_T))
        th.text(s, "стенки", (self.temp_x + 53, by + 2), 11, th.DIM, anchor="midtop")
        th.text(s, f"{Tw:.0f} К", (self.temp_x + 53, by + 15), 14, th.temperature_color(Tw),
                bold=True, anchor="midtop")
        by2 = by + 42
        th.text(s, "объём (поршень)", (self.vol_x + 57, by2 + 2), 11, th.DIM, anchor="midtop")
        th.text(s, f"{self.sim.box[0]:.0f} Å", (self.vol_x + 57, by2 + 16), 13, th.TEXT,
                anchor="midtop")
        if self.message and time.time() < self.message_until:
            msg = self.message
            f = th.font(13)
            while f.size(msg)[0] > self.view.rect.width - 20 and len(msg) > 10:
                msg = msg[:-2]
            rr = pygame.Rect(self.view.rect.x + 8, self.view.rect.bottom - 30, f.size(msg)[0] + 20, 24)
            pygame.draw.rect(s, (30, 34, 46), rr, border_radius=6)
            pygame.draw.rect(s, th.ACCENT, rr, width=1, border_radius=6)
            th.text(s, msg, (rr.x + 10, rr.y + 4), 13)
        if self.current_scenario:
            th.text(s, self.current_scenario.title, (self.view.rect.x + 14, self.view.rect.y + 8),
                    14, th.DIM, bold=True)
        if self.paused:
            th.text(s, "ПАУЗА", (self.view.rect.centerx, self.view.rect.y + 30), 22, th.ACCENT_2,
                    bold=True, anchor="center")

    def _draw_tooltip(self):
        mp = pygame.mouse.get_pos()
        tip = None
        for b in self.buttons:
            if b.hover and b.tooltip:
                tip = b.tooltip
        if tip is None and self.hover_atom is not None and self.hover_atom < self.sim.n:
            a = self.hover_atom
            el = ELEMENT_LIST[self.sim.typ[a]]
            mol = next((m for m in self.mon.molecules if a in m.atoms), None)
            parts = [f"{el.name} ({el.symbol})"]
            if mol is not None:
                parts.append(f"в составе: {mol.label}" + (f" — {mol.name}" if mol.name else ""))
            parts.append(f"заряд {self.sim.charges[a]:+.2f} e, свободная валентность "
                         f"{self.sim.free_valence[a]:.2f}")
            v = self.sim.vel[a]
            ke = 0.5 * el.mass * float(v @ v) / FORCE_TO_ACC
            parts.append(f"кинетическая энергия ≈ {ke:.1f} кДж/моль (≈ {2 * ke / (3 * KB):.0f} К)")
            tip = "\n".join(parts)
        if not tip:
            return
        lines = []
        for ln in tip.split("\n"):
            lines += th.wrap(ln, 12, 360)
        w = max(th.font(12).size(l)[0] for l in lines) + 16
        h = 18 * len(lines) + 8
        x = min(mp[0] + 16, self.W - w - 4)
        y = min(mp[1] + 18, self.H - h - 4)
        rr = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (36, 40, 54), rr, border_radius=6)
        pygame.draw.rect(self.screen, th.BORDER, rr, width=1, border_radius=6)
        for k, ln in enumerate(lines):
            th.text(self.screen, ln, (x + 8, y + 5 + 18 * k), 12)

    def _draw_help(self):
        s = self.screen
        r = pygame.Rect(0, 0, min(900, self.W - 80), min(760, self.H - 60))
        r.center = (self.W // 2, self.H // 2)
        pygame.draw.rect(s, (22, 25, 34), r, border_radius=10)
        pygame.draw.rect(s, th.ACCENT, r, width=2, border_radius=10)
        y = r.y + 16
        for para in HELP_TEXT.split("\n"):
            head = para.isupper()
            for ln in th.wrap(para, 14, r.width - 40, bold=head):
                th.text(s, ln, (r.x + 20, y), 14, th.ACCENT if head else th.TEXT, bold=head)
                y += 20
            y += 4
        th.text(s, "щелчок или Esc — закрыть", (r.centerx, r.bottom - 22), 12, th.DIM, anchor="center")

    def _draw_scenarios(self):
        s = self.screen
        r = pygame.Rect(0, 0, min(980, self.W - 80), min(800, self.H - 40))
        r.center = (self.W // 2, self.H // 2)
        pygame.draw.rect(s, (22, 25, 34), r, border_radius=10)
        pygame.draw.rect(s, th.ACCENT, r, width=2, border_radius=10)
        th.text(s, "ОПЫТЫ", (r.x + 20, r.y + 14), 18, th.ACCENT, bold=True)
        th.text(s, "Задаются только реактивы и условия — результат определяет физика",
                (r.x + 110, r.y + 18), 13, th.DIM)
        self.scenario_rects = []
        y = r.y + 50
        colw = (r.width - 60) // 2
        mp = pygame.mouse.get_pos()
        for k, sc in enumerate(SCENARIOS):
            cx = r.x + 20 + (k % 2) * (colw + 20)
            if k % 2 == 0 and k > 0:
                y += 136
            box = pygame.Rect(cx, y, colw, 128)
            hov = box.collidepoint(mp)
            pygame.draw.rect(s, th.PANEL_2 if not hov else (48, 56, 76), box, border_radius=8)
            pygame.draw.rect(s, th.BORDER if not hov else th.ACCENT, box, width=1, border_radius=8)
            th.text(s, f"{k + 1}. {sc.title}", (box.x + 10, box.y + 8), 14, th.TEXT, bold=True)
            yy = box.y + 30
            for ln in th.wrap(sc.description, 12, colw - 20)[:5]:
                th.text(s, ln, (box.x + 10, yy), 12, th.DIM)
                yy += 16
            self.scenario_rects.append((box, sc))

    def _draw_loading(self):
        s = self.screen
        cx, cy = self.W // 2, self.H // 2
        th.text(s, "Химический симулятор", (cx, cy - 80), 34, th.TEXT, bold=True, anchor="center")
        th.text(s, "подготовка физического ядра…", (cx, cy - 30), 18, th.DIM, anchor="center")
        th.text(s, "при первом запуске numba компилирует код (≈ 30 с), потом — мгновенно",
                (cx, cy), 14, th.DIM, anchor="center")
        t = time.time()
        for k in range(12):
            a = t * 2.0 + k * math.pi / 6
            col = th.lighten(th.ACCENT, (k / 12))
            pygame.draw.circle(s, col, (cx + 40 * math.cos(a), cy + 70 + 40 * math.sin(a)), 5)
        if self.loading_error:
            th.text(s, "Ошибка: " + self.loading_error, (cx, cy + 140), 14, th.BAD, anchor="center")

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if not self.handle(event):
                    running = False
            self.update()
            self.draw()
            self.clock.tick(60 if self.state == "run" else 20)
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
