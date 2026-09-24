"""Проверка модели против эксперимента.

Запуск:  python -m chemsim.validate  [--md report.md]

Сравниваются:
* энергии атомизации и геометрии молекул (минимизация энергии модели);
* теплоты реакций (из тех же энергий атомизации — модель про реакции «не знает»);
* барьеры элементарных реакций (метод протаскивания по координате реакции);
* соблюдение валентности (гипотетические «неправильные» молекулы должны быть
  нестабильны).

Экспериментальные данные: стандартные энтальпии образования в газовой фазе при
298 К (NIST Chemistry WebBook / CODATA), структуры — NIST CCCBDB, барьеры —
обзорные кинетические данные (Baulch et al., J. Phys. Chem. Ref. Data 2005 и др.).
Модель классическая (нет нулевых колебаний), поэтому сравнение с энтальпиями —
приближённое (ошибка порядка нескольких процентов ожидаема).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from .minimize import Evaluator, fire_minimize, reaction_profile
from .molecules import template

# ΔfH°(газ, 298 К), кДж/моль
DHF_ATOM = {"H": 218.0, "C": 716.7, "N": 472.7, "O": 249.2, "F": 79.4, "Cl": 121.3,
            "S": 277.2, "Na": 107.5}
DHF_MOL = {
    "H2": 0.0, "O2": 0.0, "N2": 0.0, "F2": 0.0, "Cl2": 0.0,
    "H2O": -241.8, "CH4": -74.6, "NH3": -45.9, "CO2": -393.5, "CO": -110.5,
    "C2H6": -84.0, "C2H4": 52.4, "C2H2": 227.4, "HCN": 135.1, "H2O2": -136.1,
    "CH2O": -108.6, "CH3OH": -201.0, "C6H6": 82.9, "HCl": -92.3, "HF": -273.3,
    "NaCl": -181.4, "NO": 91.3, "H2S": -20.6,
}

# экспериментальные длины связей (Å) и углы (°) для проверки геометрии
GEOMETRY = {
    "H2O": [("O", "H", 0.958), ("angle", "H-O-H", 104.5)],
    "NH3": [("N", "H", 1.012), ("angle", "H-N-H", 106.7)],
    "CH4": [("C", "H", 1.087), ("angle", "H-C-H", 109.47)],
    "CO2": [("C", "O", 1.160), ("angle", "O-C-O", 180.0)],
    "C2H6": [("C", "C", 1.535), ("C", "H", 1.094)],
    "C2H4": [("C", "C", 1.339), ("C", "H", 1.086)],
    "C2H2": [("C", "C", 1.203), ("C", "H", 1.063)],
    "HCN": [("C", "N", 1.153), ("C", "H", 1.066)],
    "H2O2": [("O", "O", 1.475), ("O", "H", 0.967)],
    "CH2O": [("C", "O", 1.205), ("C", "H", 1.111)],
    "C6H6": [("C", "C", 1.397), ("C", "H", 1.084)],
    "N2": [("N", "N", 1.098)], "O2": [("O", "O", 1.208)], "H2": [("H", "H", 0.741)],
}

REACTIONS = [
    ("2 H2 + O2 → 2 H2O", {"H2": -2, "O2": -1, "H2O": 2}),
    ("CH4 + 2 O2 → CO2 + 2 H2O", {"CH4": -1, "O2": -2, "CO2": 1, "H2O": 2}),
    ("N2 + 3 H2 → 2 NH3", {"N2": -1, "H2": -3, "NH3": 2}),
    ("H2 + Cl2 → 2 HCl", {"H2": -1, "Cl2": -1, "HCl": 2}),
    ("H2 + F2 → 2 HF", {"H2": -1, "F2": -1, "HF": 2}),
    ("C2H4 + H2 → C2H6", {"C2H4": -1, "H2": -1, "C2H6": 1}),
    ("C2H2 + 2 H2 → C2H6", {"C2H2": -1, "H2": -2, "C2H6": 1}),
    ("2 H2O2 → 2 H2O + O2", {"H2O2": -2, "H2O": 2, "O2": 1}),
    ("CH3OH + 1.5 O2 → CO2 + 2 H2O", {"CH3OH": -1, "O2": -1.5, "CO2": 1, "H2O": 2}),
    ("C6H6 + 7.5 O2 → 6 CO2 + 3 H2O", {"C6H6": -1, "O2": -7.5, "CO2": 6, "H2O": 3}),
    ("HCN + 3 H2 → CH4 + NH3", {"HCN": -1, "H2": -3, "CH4": 1, "NH3": 1}),
]

# (название, символы, координаты, (a, b, c), фрагменты реагентов, эксперимент кДж/моль)
_CH4 = [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63]]
BARRIERS = [
    ("H + H2 → H2 + H", ["H", "H", "H"], [[-3.0, 0, 0], [0, 0, 0], [0.741, 0, 0]], (0, 1, 2),
     [("H2",), ("H",)], 40.0),
    ("OH + H2 → H2O + H", ["O", "H", "H", "H"],
     [[-3, 0, 0], [-3.3, 0.9, 0], [0, 0, 0], [0.741, 0, 0]], (0, 2, 3), [("OH",), ("H2",)], 22.0),
    ("CH4 + H → CH3 + H2", ["C", "H", "H", "H", "H", "H"], _CH4 + [[2.9, 2.9, 2.9]], (5, 1, 0),
     [("CH4",), ("H",)], 55.0),
    ("Cl + H2 → HCl + H", ["Cl", "H", "H"], [[-3.5, 0, 0], [0, 0, 0], [0.741, 0, 0]], (0, 1, 2),
     [("H2",), ("Cl",)], 20.0),
    ("F + H2 → HF + H", ["F", "H", "H"], [[-3.0, 0, 0], [0, 0, 0], [0.741, 0, 0]], (0, 1, 2),
     [("H2",), ("F",)], 6.0),
    ("H + Cl2 → HCl + Cl", ["H", "Cl", "Cl"], [[-3.0, 0, 0], [0, 0, 0], [1.99, 0, 0]], (0, 1, 2),
     [("Cl2",), ("H",)], 8.0),
]

# гипотетические частицы, нарушающие валентность: (название, символы, координаты,
# распад на известные частицы) — должны быть выше распада по энергии
VALENCE = [
    ("CH5 → CH4 + H", ["C", "H", "H", "H", "H", "H"],
     [[0, 0, 0], [1.1, 0, 0], [-1.1, 0, 0], [0, 1.1, 0], [0, -1.1, 0], [0, 0, 1.1]], ["CH4", "H"]),
    ("H3O → H2O + H", ["O", "H", "H", "H"],
     [[0, 0, 0.2], [1, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0]], ["H2O", "H"]),
    ("NH4 → NH3 + H", ["N", "H", "H", "H", "H"],
     [[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]],
     ["NH3", "H"]),
    ("H3 (линейный) → H2 + H", ["H", "H", "H"], [[-0.95, 0, 0], [0, 0, 0], [0.95, 0.05, 0]],
     ["H2", "H"]),
    ("H3 (треугольник) → H2 + H", ["H", "H", "H"], [[0, 0, 0], [0.9, 0, 0], [0.45, 0.8, 0]],
     ["H2", "H"]),
    ("H4 (квадрат) → 2 H2", ["H"] * 4, [[0, 0, 0], [1.0, 0, 0], [1, 1, 0], [0, 1, 0.05]],
     ["H2", "H2"]),
    ("(H2O)2: ковалентной связи между молекулами нет", ["O", "H", "H", "O", "H", "H"],
     [[0, 0, 0], [.96, 0, 0], [-.3, .9, 0], [2.9, 0, 0], [3.2, .9, 0], [3.2, -0.4, 0.8]],
     ["H2O", "H2O"]),
]

_FRAGMENTS = {"OH": (["O", "H"], [[0, 0, 0], [0.97, 0, 0]]), "H": (["H"], [[0, 0, 0]]),
              "Cl": (["Cl"], [[0, 0, 0]]), "F": (["F"], [[0, 0, 0]])}


def molecule_energy(name: str, params=None):
    """Минимальная энергия (кДж/моль, относительно свободных атомов) и геометрия."""
    if name in _FRAGMENTS:
        syms, pos = _FRAGMENTS[name]
        pos = np.array(pos, dtype=float)
    else:
        syms, pos = template(name, relax=False)
    ev = Evaluator(syms, params)
    if len(syms) == 1:
        return 0.0, syms, pos
    x, e, _ = fire_minimize(ev, pos, fmax=1e-4, max_steps=40000)
    return e, syms, x


def atomization_exp(name: str) -> float:
    from .names import parse_formula
    counts = parse_formula(name)
    return sum(DHF_ATOM[a] * n for a, n in counts.items()) - DHF_MOL[name]


def _measure(syms, x, spec):
    if spec[0] == "angle":
        a, c, b = spec[1].split("-")
        best = None
        for i, si in enumerate(syms):
            if si != c:
                continue
            nb = [j for j, s in enumerate(syms) if j != i and np.linalg.norm(x[j] - x[i]) < 1.3]
            for u in range(len(nb)):
                for v in range(u + 1, len(nb)):
                    j, k = nb[u], nb[v]
                    if {syms[j], syms[k]} != {a, b} and not (syms[j] == a and syms[k] == b):
                        continue
                    d1 = x[j] - x[i]
                    d2 = x[k] - x[i]
                    ang = math.degrees(math.acos(np.clip(d1 @ d2 / np.linalg.norm(d1) / np.linalg.norm(d2), -1, 1)))
                    best = ang if best is None else best
        return best
    a, b, _ = spec
    best = None
    for i, si in enumerate(syms):
        for j, sj in enumerate(syms):
            if j <= i or {si, sj} != {a, b}:
                continue
            r = float(np.linalg.norm(x[i] - x[j]))
            if best is None or r < best:
                best = r
    return best


@dataclass
class Row:
    name: str
    model: float
    exp: float

    @property
    def err(self):
        return self.model - self.exp

    @property
    def rel(self):
        return (self.model - self.exp) / abs(self.exp) * 100.0 if self.exp else float("nan")


def atomization_table(params=None):
    rows = []
    for name in DHF_MOL:
        if name in ("F2", "Cl2", "NO", "H2S"):
            # у F2, Cl2 энергия берётся из таблицы напрямую; проверяем остальное
            pass
        e, _, _ = molecule_energy(name, params)
        rows.append(Row(name, -e, atomization_exp(name)))
    return rows


def geometry_table(params=None):
    rows = []
    for name, specs in GEOMETRY.items():
        _, syms, x = molecule_energy(name, params)
        for spec in specs:
            val = _measure(syms, x, spec)
            label = f"{name}: " + (spec[1] + " (°)" if spec[0] == "angle" else f"{spec[0]}–{spec[1]} (Å)")
            if val is not None:
                rows.append(Row(label, val, spec[-1]))
    return rows


def reaction_table(params=None):
    cache = {}
    rows = []
    for title, stoich in REACTIONS:
        dh_model = 0.0
        dh_exp = 0.0
        for sp, nu in stoich.items():
            if sp not in cache:
                cache[sp] = molecule_energy(sp, params)[0]
            dh_model += nu * cache[sp]
            dh_exp += nu * DHF_MOL[sp]
        rows.append(Row(title, dh_model, dh_exp))
    return rows


def barrier_table(params=None):
    rows = []
    for title, syms, pos, (a, b, c), frags, exp in BARRIERS:
        e_react = sum(molecule_energy(f[0], params)[0] for f in frags)
        prof = reaction_profile(syms, pos, a, b, c, np.linspace(-2.2, 2.2, 45), params=params)
        emax = max(p[1] for p in prof)
        rows.append(Row(title, emax - e_react, exp))
    return rows


def valence_table(params=None):
    """(процесс, E(частицы) − E(продуктов распада)).  Для димера воды вместо энергии
    проверяется отсутствие межмолекулярных ковалентных связей: возвращается
    +1, если их нет, и −1, если есть."""
    rows = []
    for title, syms, pos, parts in VALENCE:
        ev = Evaluator(syms, params)
        x, e, _ = fire_minimize(ev, np.array(pos, float), fmax=1e-3, max_steps=20000)
        if title.startswith("(H2O)2"):
            inter = [b for (i, j), b in ev.bond_orders(x).items() if (i < 3) != (j < 3)]
            rows.append((title, 1.0 if max(inter, default=0.0) < 0.2 else -1.0))
            continue
        e_parts = sum(molecule_energy(p, params)[0] for p in parts)
        rows.append((title, e - e_parts))
    return rows


def report(params=None) -> str:
    lines = ["# Проверка модели против эксперимента", ""]

    def table(title, rows, unit, fmt="{:.1f}"):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| | модель | эксперимент | ошибка |")
        lines.append("|---|---:|---:|---:|")
        for r in rows:
            rel = f" ({r.rel:+.0f}%)" if r.exp else ""
            lines.append(f"| {r.name} | {fmt.format(r.model)} | {fmt.format(r.exp)} | "
                         f"{fmt.format(r.err)}{rel} |")
        lines.append("")
        return rows

    at = table("Энергии атомизации, кДж/моль", atomization_table(params), "кДж/моль")
    mae = np.mean([abs(r.rel) for r in at])
    lines.append(f"Средняя относительная ошибка: {mae:.1f}%  (CO — известное ограничение: "
                 "модель не описывает донорно-акцепторную третью связь C≡O)")
    lines.append("")
    table("Геометрия", geometry_table(params), "", fmt="{:.3f}")
    table("Теплоты реакций ΔH, кДж/моль (модель ничего не знает о реакциях — только "
          "энергии молекул)", reaction_table(params), "кДж/моль")
    table("Барьеры элементарных реакций, кДж/моль (эксперимент — энергии активации)",
          barrier_table(params), "кДж/моль")
    lines.append("## Валентность: гипотетические частицы должны быть нестабильны")
    lines.append("")
    lines.append("| процесс | E(частицы) − E(продуктов распада), кДж/моль | вывод |")
    lines.append("|---|---:|---|")
    for title, de in valence_table(params):
        lines.append(f"| {title} | {de:+.1f} | {'нестабильна ✓' if de > 0 else 'связана ✗'} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Сравнение модели с экспериментом")
    ap.add_argument("--md", help="сохранить отчёт в Markdown-файл")
    args = ap.parse_args()
    text = report()
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
