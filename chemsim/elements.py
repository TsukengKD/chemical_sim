"""Периодическая таблица: табличные (экспериментальные) свойства элементов.

Все параметры модели выводятся ТОЛЬКО из этих свойств элементов (плюс таблица
экспериментальных энергий связей в ``bonddata.py``).  В движке нет ни одной
«заготовленной» реакции: какие вещества получатся, решает потенциальная энергия.

Источники:
* массы — IUPAC 2021 (стандартные атомные веса);
* электроотрицательность — шкала Полинга (Allred, 1961);
* ковалентные радиусы одинарной/двойной/тройной связи — P. Pyykkö,
  M. Atsumi, Chem. Eur. J. 15 (2009) 186; 15 (2009) 12770; Pyykkö et al. 2005;
* ван-дер-ваальсовы параметры x_i (Å) и D_i (ккал/моль) — универсальное силовое
  поле UFF: A. K. Rappé et al., J. Am. Chem. Soc. 114 (1992) 10024;
* энергии гомоядерных связей — средние энергии связей (CRC Handbook,
  J. E. Huheey «Inorganic Chemistry», табл. E.1);
* энергии когезии металлов — C. Kittel, «Introduction to Solid State Physics».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str                  # русское название
    Z: int                     # атомный номер
    mass: float                # а.е.м.
    chi: float                 # электроотрицательность по Полингу (0 для благородных газов)
    valence: int               # число ковалентных связей (обычная валентность)
    val_electrons: int         # число валентных электронов (для подсчёта неподелённых пар)
    period: int                # период (для правила Баджера)
    r1: float                  # ковалентный радиус одинарной связи, Å
    r2: Optional[float]        # ... двойной связи
    r3: Optional[float]        # ... тройной связи
    uff_x: float               # ван-дер-ваальсово расстояние UFF, Å
    uff_D: float               # глубина ван-дер-ваальсовой ямы UFF, ккал/моль
    D1: Optional[float] = None  # энергия одинарной связи A–A, кДж/моль
    D2: Optional[float] = None  # энергия двойной связи A=A
    D3: Optional[float] = None  # энергия тройной связи A≡A
    rb1: Optional[float] = None  # экспериментальная длина связи A–A, Å
    rb2: Optional[float] = None
    rb3: Optional[float] = None
    max_order: int = 1         # максимальный порядок связи, который допускает атом
    metal: bool = False
    # для металлов: энергия когезии (кДж/моль на атом) и координационное число решётки
    cohesive: Optional[float] = None
    lattice_z: Optional[int] = None
    color: tuple = (200, 200, 200)
    group_label: str = ""

    @property
    def noble(self) -> bool:
        return self.valence == 0


ELEMENT_LIST = [
    Element("H", "Водород", 1, 1.008, 2.20, 1, 1, 1, 0.32, None, None, 2.886, 0.044,
            D1=436.0, rb1=0.741, color=(235, 235, 235), group_label="неметалл"),
    Element("He", "Гелий", 2, 4.0026, 0.0, 0, 2, 1, 0.46, None, None, 2.362, 0.056,
            color=(217, 255, 255), group_label="благородный газ"),
    Element("Li", "Литий", 3, 6.94, 0.98, 1, 1, 2, 1.33, 1.24, None, 2.451, 0.025,
            rb1=2.673, metal=True, cohesive=158.0, lattice_z=8, color=(204, 128, 255),
            group_label="щелочной металл"),
    Element("B", "Бор", 5, 10.81, 2.04, 3, 3, 2, 0.85, 0.78, 0.73, 4.083, 0.180,
            D1=293.0, color=(255, 181, 181), group_label="полуметалл"),
    Element("C", "Углерод", 6, 12.011, 2.55, 4, 4, 2, 0.75, 0.67, 0.60, 3.851, 0.105,
            D1=346.0, D2=614.0, D3=839.0, rb1=1.535, rb2=1.339, rb3=1.203, max_order=3,
            color=(120, 120, 120), group_label="неметалл"),
    Element("N", "Азот", 7, 14.007, 3.04, 3, 5, 2, 0.71, 0.60, 0.54, 3.660, 0.069,
            D1=167.0, D2=418.0, D3=945.0, rb1=1.45, rb2=1.25, rb3=1.098, max_order=3,
            color=(60, 90, 250), group_label="неметалл"),
    Element("O", "Кислород", 8, 15.999, 3.44, 2, 6, 2, 0.63, 0.57, 0.53, 3.500, 0.060,
            D1=142.0, D2=498.0, rb1=1.475, rb2=1.208, max_order=2,
            color=(255, 40, 40), group_label="неметалл"),
    Element("F", "Фтор", 9, 18.998, 3.98, 1, 7, 2, 0.64, 0.59, 0.53, 3.364, 0.050,
            D1=159.0, rb1=1.412, color=(144, 224, 80), group_label="галоген"),
    Element("Ne", "Неон", 10, 20.180, 0.0, 0, 8, 2, 0.67, 0.96, None, 3.243, 0.042,
            color=(179, 227, 245), group_label="благородный газ"),
    Element("Na", "Натрий", 11, 22.990, 0.93, 1, 1, 3, 1.55, 1.60, None, 2.983, 0.030,
            rb1=3.079, metal=True, cohesive=107.0, lattice_z=8, color=(171, 92, 242),
            group_label="щелочной металл"),
    Element("Mg", "Магний", 12, 24.305, 1.31, 2, 2, 3, 1.39, 1.32, 1.27, 3.021, 0.111,
            metal=True, cohesive=145.0, lattice_z=12, color=(138, 255, 0),
            group_label="щёлочноземельный металл"),
    Element("Al", "Алюминий", 13, 26.982, 1.61, 3, 3, 3, 1.26, 1.13, 1.11, 4.499, 0.505,
            metal=True, cohesive=327.0, lattice_z=12, color=(191, 166, 166),
            group_label="металл"),
    Element("Si", "Кремний", 14, 28.085, 1.90, 4, 4, 3, 1.16, 1.07, 1.02, 4.295, 0.402,
            D1=222.0, rb1=2.35, color=(240, 200, 160), group_label="полуметалл"),
    Element("P", "Фосфор", 15, 30.974, 2.19, 3, 5, 3, 1.11, 1.02, 0.94, 4.147, 0.305,
            D1=201.0, D2=310.0, D3=489.0, rb1=2.21, rb2=2.03, rb3=1.893, max_order=3,
            color=(255, 128, 0), group_label="неметалл"),
    Element("S", "Сера", 16, 32.06, 2.58, 2, 6, 3, 1.03, 0.94, 0.95, 4.035, 0.274,
            D1=226.0, D2=425.0, rb1=2.05, rb2=1.889, max_order=2,
            color=(255, 230, 40), group_label="неметалл"),
    Element("Cl", "Хлор", 17, 35.45, 3.16, 1, 7, 3, 0.99, 0.95, 0.93, 3.947, 0.227,
            D1=242.0, rb1=1.988, color=(31, 240, 31), group_label="галоген"),
    Element("Ar", "Аргон", 18, 39.948, 0.0, 0, 8, 3, 0.96, 1.07, 0.96, 3.868, 0.185,
            color=(128, 209, 227), group_label="благородный газ"),
    Element("K", "Калий", 19, 39.098, 0.82, 1, 1, 4, 1.96, 1.93, None, 3.812, 0.035,
            rb1=3.905, metal=True, cohesive=90.1, lattice_z=8, color=(143, 64, 212),
            group_label="щелочной металл"),
    Element("Ca", "Кальций", 20, 40.078, 1.00, 2, 2, 4, 1.71, 1.47, 1.33, 3.399, 0.238,
            metal=True, cohesive=178.0, lattice_z=12, color=(61, 255, 0),
            group_label="щёлочноземельный металл"),
    Element("Br", "Бром", 35, 79.904, 2.96, 1, 7, 4, 1.14, 1.09, 1.10, 4.189, 0.251,
            D1=193.0, rb1=2.281, color=(190, 50, 50), group_label="галоген"),
    Element("I", "Иод", 53, 126.904, 2.66, 1, 7, 5, 1.33, 1.29, 1.25, 4.500, 0.339,
            D1=151.0, rb1=2.666, color=(160, 40, 200), group_label="галоген"),
]

ELEMENTS = {e.symbol: e for e in ELEMENT_LIST}
SYMBOLS = [e.symbol for e in ELEMENT_LIST]
INDEX = {s: i for i, s in enumerate(SYMBOLS)}


def element(symbol: str) -> Element:
    try:
        return ELEMENTS[symbol]
    except KeyError:
        raise KeyError(f"Неизвестный элемент: {symbol!r}. Доступны: {', '.join(SYMBOLS)}")
