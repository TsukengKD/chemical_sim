"""Вывод параметров потенциала из свойств элементов.

Для каждой пары элементов (A, B) строятся:

1. **Одинарная связь** — потенциал Морзе с параметрами
   * D1 — из таблицы экспериментальных энергий связи, иначе по уравнению
     Полинга:  D(A–B) = ½[D(A–A) + D(B–B)] + 96.5·(χA − χB)²  (кДж/моль);
   * r1 — экспериментальная длина, иначе сумма ковалентных радиусов Пюккё;
   * a1 = √(k / 2D1), где силовая постоянная k — по правилу Баджера
     k·(r − d_ij)³ = 1.86 мдин·Å² (d_ij зависит только от периодов A и B).

2. **π-связи** (двойная/тройная).  К одинарной связи добавляются экспоненциальные
   π-слагаемые  −P(r) = −Cπ·exp(−aπ(r − r2))  и  −Q(r) = −Cππ·exp(−aππ(r − r3)),
   коэффициенты которых находятся из условия: при порядке 2 минимум энергии лежит
   ровно в r2 и имеет глубину D2, при порядке 3 — в r3 с глубиной D3.

3. **Ван-дер-ваальсово взаимодействие** — параметры UFF (правила комбинирования
   UFF: геометрические средние).

4. **Металлы**.  Энергия «одинарной» связи M–M берётся не из двухатомной молекулы
   (которая для металлов нетипична), а из экспериментальной энергии когезии
   кристалла, пересчитанной по той же модели порядка связи, что работает в
   движке (приближение второго момента / Финниса–Синклера).

Ни на одном шаге не задаются продукты реакций.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bonddata
from .elements import ELEMENT_LIST, SYMBOLS, INDEX, Element
from .units import PAULING_IONIC, KCAL_TO_KJ

# ---------------------------------------------------------------- индексы
# параметры пары (pairpar[a, b, k])
P_BOND = 0     # 1.0, если пара может образовывать химическую связь
P_D1 = 1       # энергия одинарной связи, кДж/моль
P_R1 = 2       # равновесная длина одинарной связи, Å
P_A1 = 3       # параметр Морзе, 1/Å
P_RON = 4      # начало «выключения» связевого вклада, Å
P_ROFF = 5     # конец, Å
P_PIMAX = 6    # максимальный π-порядок (0, 1, 2)
P_CP = 7
P_AP = 8
P_R2 = 9
P_CQ = 10
P_AQ = 11
P_R3 = 12
P_EPS = 13     # глубина ван-дер-ваальсовой ямы, кДж/моль
P_X6 = 14      # x^6 (x — ван-дер-ваальсово расстояние)
P_GR6 = 15     # (γ_r x)^6 — «смягчение» отталкивания на малых r
P_GD6 = 16     # (γ_d x)^6 — демпфирование дисперсии
P_RONL = 17    # окно дальнего притяжения радикалов (хвост Морзе)
P_ROFFL = 18
P_QT = 19      # заряд, переходящий на атом первого типа по связи (ионность Полинга)
P_DI = 20      # показатель насыщения δ для атома первого типа в этой связи
NPP = 21

# параметры элемента (elempar[a, k])
E_VAL = 0      # валентность
E_EVAL = 1     # число валентных электронов
E_DELTA = 2    # показатель насыщения связи (Абелл–Терсофф)
E_KANG = 3     # константа угловой жёсткости (VSEPR)
E_MASS = 4
E_GACUTE = 5   # усиление конкуренции соседей под острыми углами (Терсофф)
E_CHI = 6      # электроотрицательность
E_METAL = 7    # 1 для металлов (есть электроны проводимости)
NEP = 8


@dataclass
class ModelSettings:
    """Универсальные (не зависящие от элементов) параметры модели."""

    # показатель δ в законе порядка связи b = (V/Z)^δ при перекоординации.
    # δ=0.5 — предел Абелла (энергия не зависит от числа соседей), δ=1 — жёсткая
    # валентность.  Для ковалентных неметаллов подобран по барьеру H + H2 -> H2 + H
    # (эксперимент ≈ 40 кДж/моль), для металлов — по отношению энергии когезии к
    # энергии связи димера.
    delta_nonmetal: float = 0.55
    delta_metal: float = 0.30
    # ионная связь ненаправленная и ненасыщаемая, как металлическая: для связи с
    # ионным характером I показатель δ = δ_ков·(1 − I) + δ_ион·I.  δ_ион = 0.3
    # даёт энергию когезии кристалла NaCl ≈ 650 кДж/моль (эксп. 642).
    delta_ionic: float = 0.30
    # угловая жёсткость VSEPR, кДж/моль (даёт частоту деформационного колебания
    # H2O ≈ 1600 см⁻¹)
    k_angle: float = 200.0
    # Угловая зависимость координации (как функция g(θ) в потенциале Терсоффа):
    # сосед k, находящийся под острым углом к связи i–j, «занимает» ту же
    # орбиталь и учитывается с весом 1 + c·cos²θ (для θ < 90°).  Без этого
    # треугольные кластеры (H3, H4…) были бы связаны.  Для металлов — 0
    # (металлическая связь ненаправленная, плотные упаковки выгодны).
    g_acute: float = 6.0
    # Электростатика: заряды из ионного характера связи по Полингу
    # I = 1 − exp(−Δχ²/4), нормированные на сумму валентностей связей атома.
    coulomb: bool = True
    coulomb_shield: float = 0.6   # радиус экранирования на малых r, Å
    # Окно «координации» (связь считается соседом при подсчёте валентности), в
    # единицах 1/a1 от r1.  Оно дополнительно ограничено сверху так, чтобы не
    # захватывать геминальные (1-3) пары: d13 ≈ 1.6·(r_A + r_B + 2·r_X)/2, где
    # r_X = 0.63 Å — наименьший ковалентный радиус «центрального» атома (O).
    # Так же устроены потенциалы Терсоффа и Бреннера (обрезание между первой и
    # второй координационными сферами).
    z_on: float = 0.5
    z_off: float = 2.5
    d13_factor: float = 1.60
    d13_center: float = 0.63
    # окно дальнего притяжения радикалов (хвост кривой Морзе), в единицах 1/a1
    tail_on: float = 1.5
    tail_off: float = 3.5
    # глобальный радиус обрезания
    r_sw: float = 4.5
    r_cut: float = 5.5
    # форма ван-дер-ваальсова потенциала
    gamma_r: float = 0.5
    gamma_d: float = 0.8
    # ширины сглаживания кусочно-гладких функций (в единицах порядка связи)
    k_min: float = 0.10     # сглаженный min
    k_over: float = 0.30    # переход к перекоординации
    k_spare: float = 0.10   # свободная валентность
    k_pi: float = 0.10      # переход π -> ππ
    k_lp: float = 0.20      # неподелённые пары
    eps_w: float = 1.0e-3   # регуляризатор деления в распределении π-связей


# ------------------------------------------------------------- правило Баджера
# d_ij (Å) для периодов i, j (Badger 1934–35; Herschbach & Laurie 1961)
_BADGER_D = {
    (1, 1): 0.025, (1, 2): 0.335, (1, 3): 0.585, (1, 4): 0.65, (1, 5): 0.77,
    (2, 2): 0.68, (2, 3): 0.90, (2, 4): 1.03, (2, 5): 1.15,
    (3, 3): 1.18, (3, 4): 1.28, (3, 5): 1.40,
    (4, 4): 1.37, (4, 5): 1.50,
    (5, 5): 1.64,
}
MDYN_A_TO_KJMOL_A2 = 602.214  # 1 мдин/Å = 100 Н/м = 602.214 кДж/(моль·Å²)


def badger_k(r: float, p1: int, p2: int) -> float:
    """Силовая постоянная связи (кДж/моль/Å²) по правилу Баджера."""
    d = _BADGER_D[(min(p1, p2), max(p1, p2))]
    dr = max(r - d, 0.35)
    return 1.86 / dr ** 3 * MDYN_A_TO_KJMOL_A2


def ionic_term(dchi: float) -> float:
    """Ионная добавка Полинга 96.5·Δχ² (кДж/моль).

    Формула Полинга надёжна при Δχ ≲ 1.5; при большей разности она сильно
    завышает энергию (ионный характер насыщается), поэтому дальше используется
    линейное продолжение, согласованное с галогенидами щелочных металлов.
    """
    dchi = abs(dchi)
    if dchi <= 1.5:
        return PAULING_IONIC * dchi ** 2
    return PAULING_IONIC * 2.25 + 120.0 * (dchi - 1.5)


def _metal_D1(el: Element, s: ModelSettings) -> float:
    """Эффективная энергия связи M–M из энергии когезии.

    В модели каждый из Z соседей получает порядок b = (V/Z)^δ, энергия связи при
    оптимальной длине равна −b²·D1, связей на атом Z/2:
        E_coh = (Z/2)·(V/Z)^(2δ)·D1.
    """
    z = el.lattice_z
    v = el.valence
    b = (v / z) ** s.delta_metal if z > v else 1.0
    return el.cohesive / (0.5 * z * b * b)


def homonuclear(el: Element, order: int, s: ModelSettings):
    """(D, r) для связи A–A порядка order либо None."""
    if el.valence == 0:
        return None
    if order == 1:
        if el.metal:
            D = _metal_D1(el, s)
        else:
            D = el.D1
        r = el.rb1 if el.rb1 is not None else 2.0 * el.r1
        return (D, r) if D is not None else None
    if order == 2:
        if el.D2 is None or el.max_order < 2:
            return None
        r = el.rb2 if el.rb2 is not None else 2.0 * el.r2
        return (el.D2, r)
    if order == 3:
        if el.D3 is None or el.max_order < 3:
            return None
        r = el.rb3 if el.rb3 is not None else 2.0 * el.r3
        return (el.D3, r)
    return None


def _radius(el: Element, order: int) -> float:
    return {1: el.r1, 2: el.r2 or el.r1, 3: el.r3 or el.r2 or el.r1}[order]


def pair_bond(a: Element, b: Element, order: int, s: ModelSettings):
    """(D, r, источник) для связи A–B порядка order либо None."""
    if a.valence == 0 or b.valence == 0:
        return None
    if order > min(a.max_order, b.max_order):
        return None
    if a.symbol == b.symbol:
        h = homonuclear(a, order, s)
        return (h[0], h[1], "эксперимент (A–A)") if h else None
    tab = bonddata.lookup(a.symbol, b.symbol, order)
    if tab is not None:
        return (tab[0], tab[1], "эксперимент")
    ha = homonuclear(a, order, s)
    hb = homonuclear(b, order, s)
    if ha is None or hb is None:
        return None
    D = 0.5 * (ha[0] + hb[0]) + ionic_term(a.chi - b.chi)
    r = _radius(a, order) + _radius(b, order)
    return (D, r, "оценка (Полинг + радиусы Пюккё)")


def _morse(D, a, r0, r):
    e = math.exp(-a * (r - r0))
    return D * ((1.0 - e) ** 2 - 1.0), 2.0 * a * D * (1.0 - e) * e, 2.0 * a * a * D * (2.0 * e * e - e)


@dataclass
class PairInfo:
    a: str
    b: str
    bondable: bool
    orders: dict = field(default_factory=dict)  # порядок -> (D, r, источник)
    a1: float = 0.0
    k1: float = 0.0
    pimax: int = 0
    cp: float = 0.0
    ap: float = 0.0
    cq: float = 0.0
    aq: float = 0.0
    eps: float = 0.0
    xv: float = 0.0
    notes: list = field(default_factory=list)


def derive_pair(a: Element, b: Element, s: ModelSettings) -> PairInfo:
    info = PairInfo(a.symbol, b.symbol, False)
    info.eps = math.sqrt(a.uff_D * b.uff_D) * KCAL_TO_KJ
    info.xv = math.sqrt(a.uff_x * b.uff_x)
    one = pair_bond(a, b, 1, s)
    if one is None:
        return info
    info.bondable = True
    D1, r1, _ = one
    info.orders[1] = one
    k = badger_k(r1, a.period, b.period)
    a1 = math.sqrt(k / (2.0 * D1))
    a1 = min(max(a1, 0.6), 3.5)
    info.a1, info.k1 = a1, 2.0 * D1 * a1 * a1

    # --- π-связи: подгонка Cπ, aπ (и Cππ, aππ) к (D2, r2), (D3, r3)
    two = pair_bond(a, b, 2, s)
    if two is not None:
        D2, r2, _ = two
        m, dm, d2m = _morse(D1, a1, r1, r2)
        cp = D2 + m
        ap = -dm / cp if cp > 0 else -1.0
        curv = d2m - ap * ap * cp
        if cp > 0 and 0 < ap < 1.9 * a1 and curv > 0:
            info.orders[2] = two
            info.pimax, info.cp, info.ap = 1, cp, ap
            three = pair_bond(a, b, 3, s)
            if three is not None:
                D3, r3, _ = three
                m3, dm3, d2m3 = _morse(D1, a1, r1, r3)
                p3 = cp * math.exp(-ap * (r3 - r2))
                e2 = m3 - p3
                de2 = dm3 + ap * p3
                d2e2 = d2m3 - ap * ap * p3
                cq = D3 + e2
                aq = -de2 / cq if cq > 0 else -1.0
                curv3 = d2e2 - aq * aq * cq
                if cq > 0 and 0 < aq < 1.9 * a1 and curv3 > 0:
                    info.orders[3] = three
                    info.pimax, info.cq, info.aq = 2, cq, aq
                else:
                    info.notes.append("тройная связь отброшена: неустойчивая подгонка")
        else:
            info.notes.append("двойная связь отброшена: неустойчивая подгонка")
    return info


class ForceFieldParams:
    """Таблицы параметров для всех пар элементов (используются ядром numba)."""

    def __init__(self, settings: ModelSettings | None = None):
        self.settings = s = settings or ModelSettings()
        n = len(ELEMENT_LIST)
        self.symbols = list(SYMBOLS)
        self.elempar = np.zeros((n, NEP))
        self.pairpar = np.zeros((n, n, NPP))
        self.pairs: dict[tuple[str, str], PairInfo] = {}
        for i, el in enumerate(ELEMENT_LIST):
            self.elempar[i, E_VAL] = el.valence
            self.elempar[i, E_EVAL] = el.val_electrons if el.valence > 0 else 0
            self.elempar[i, E_DELTA] = s.delta_metal if el.metal else s.delta_nonmetal
            directional = (not el.metal) and el.valence >= 2
            self.elempar[i, E_KANG] = s.k_angle if directional else 0.0
            self.elempar[i, E_MASS] = el.mass
            self.elempar[i, E_GACUTE] = 0.0 if el.metal else s.g_acute
            self.elempar[i, E_CHI] = el.chi
            self.elempar[i, E_METAL] = 1.0 if el.metal else 0.0
        # коэффициенты формы ван-дер-ваальсова потенциала: минимум в x, глубина ε
        self.c_rep, self.c_disp = _vdw_shape_coefficients(s.gamma_r, s.gamma_d)
        for i, ea in enumerate(ELEMENT_LIST):
            for j, eb in enumerate(ELEMENT_LIST):
                if j < i:
                    self.pairpar[i, j] = self.pairpar[j, i]
                    self.pairpar[i, j, P_QT] = -self.pairpar[j, i, P_QT]
                    continue
                info = derive_pair(ea, eb, s)
                self.pairs[(ea.symbol, eb.symbol)] = info
                self.pairs[(eb.symbol, ea.symbol)] = info
                pp = self.pairpar[i, j]
                pp[P_EPS] = info.eps
                pp[P_X6] = info.xv ** 6
                pp[P_GR6] = (s.gamma_r * info.xv) ** 6
                pp[P_GD6] = (s.gamma_d * info.xv) ** 6
                if info.bondable:
                    D1, r1, _ = info.orders[1]
                    pp[P_BOND] = 1.0
                    pp[P_D1] = D1
                    pp[P_R1] = r1
                    pp[P_A1] = info.a1
                    roff = r1 + s.z_off / info.a1
                    if not (ea.metal and eb.metal):
                        d13 = s.d13_factor * 0.5 * (ea.r1 + eb.r1 + 2.0 * s.d13_center)
                        roff = min(roff, d13 - 0.1)
                    roff = min(roff, s.r_cut - 0.2)
                    ron = min(r1 + s.z_on / info.a1, roff - 0.3)
                    pp[P_RON] = ron
                    pp[P_ROFF] = roff
                    pp[P_ROFFL] = min(r1 + s.tail_off / info.a1, s.r_cut - 0.2)
                    pp[P_RONL] = min(r1 + s.tail_on / info.a1, pp[P_ROFFL] - 0.3)
                    if s.coulomb:
                        dchi = eb.chi - ea.chi
                        ion = 1.0 - math.exp(-0.25 * dchi * dchi)
                        pp[P_QT] = math.copysign(ion, dchi) if dchi != 0 else 0.0
                    pp[P_PIMAX] = info.pimax
                    if info.pimax >= 1:
                        pp[P_CP] = info.cp
                        pp[P_AP] = info.ap
                        pp[P_R2] = info.orders[2][1]
                    if info.pimax >= 2:
                        pp[P_CQ] = info.cq
                        pp[P_AQ] = info.aq
                        pp[P_R3] = info.orders[3][1]
        # δ каждой стороны связи с учётом ионности (несимметричен)
        for i, ea in enumerate(ELEMENT_LIST):
            for j, eb in enumerate(ELEMENT_LIST):
                d_i = self.elempar[i, E_DELTA]
                if ea.valence > 0 and eb.valence > 0 and ea.chi > 0 and eb.chi > 0:
                    ion = 1.0 - math.exp(-0.25 * (ea.chi - eb.chi) ** 2)
                else:
                    ion = 0.0
                self.pairpar[i, j, P_DI] = d_i * (1.0 - ion) + s.delta_ionic * ion
        self.masses = self.elempar[:, E_MASS].copy()
        # скаляры для ядра
        self.scalars = np.array([
            s.r_sw, s.r_cut, self.c_rep, self.c_disp,
            s.k_min, s.k_over, s.k_spare, s.k_pi, s.k_lp, s.eps_w,
            COULOMB_K if s.coulomb else 0.0, s.coulomb_shield ** 3,
        ])

    def index(self, symbol: str) -> int:
        return INDEX[symbol]

    def pair(self, a: str, b: str) -> PairInfo:
        return self.pairs[(a, b)]


# индексы скаляров
S_RSW, S_RCUT, S_CREP, S_CDISP, S_KMIN, S_KOVER, S_KSPARE, S_KPI, S_KLP, S_EPSW, S_KE, S_AS3 = range(12)

# постоянная Кулона: e²/(4πε0) в кДж·Å/моль
COULOMB_K = 1389.3546


def _vdw_shape_coefficients(gr: float, gd: float):
    """Найти c_r, c_d так, чтобы E(r) = c_r ε (x/rs)^12 − c_d ε x^6/(r^6+(γd x)^6)
    имела минимум при r = x глубиной −ε (как у потенциала UFF)."""
    x = 1.0
    rs6 = x ** 6 + (gr * x) ** 6
    A = (x ** 6 / rs6) ** 2
    dA = -12.0 * x ** 12 * x ** 5 / rs6 ** 3
    den = x ** 6 + (gd * x) ** 6
    B = x ** 6 / den
    dB = -6.0 * x ** 6 * x ** 5 / den ** 2
    # c_r A − c_d B = −1 ;  c_r dA − c_d dB = 0
    M = np.array([[A, -B], [dA, -dB]])
    c_r, c_d = np.linalg.solve(M, np.array([-1.0, 0.0]))
    return float(c_r), float(c_d)


_DEFAULT = None


def default_params() -> ForceFieldParams:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ForceFieldParams()
    return _DEFAULT
