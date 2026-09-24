"""Химический анализ того, что происходит в реакторе.

* какие связи существуют (с гистерезисом, чтобы колебания около порога не
  выглядели как реакции);
* какие молекулы есть (связные компоненты графа связей), их формулы и названия;
* какие реакции произошли (перегруппировки атомов между молекулами).

Анализ только НАБЛЮДАЕТ за динамикой — на физику он не влияет.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .elements import SYMBOLS
from .names import hill_formula, display_formula, species_name, pretty_formula

BOND_FORM = 0.6     # связь «образовалась», если порядок ≥ 0.6
BOND_BREAK = 0.35   # и «разорвалась», если порядок < 0.35
RADICAL_SPARE = 0.5  # свободная валентность, начиная с которой атом — радикальный центр


class BondTracker:
    """Множество связей с гистерезисом по порядку связи."""

    def __init__(self):
        self.bonds: dict[tuple[int, int], float] = {}

    def reset(self):
        self.bonds.clear()

    def update(self, sim):
        n = sim.npair
        pi = sim.pi[:n]
        pj = sim.pj[:n]
        bo = sim.bond_order[:n]
        sel = np.nonzero(bo >= BOND_BREAK)[0]
        new = {}
        for k in sel:
            key = (int(pi[k]), int(pj[k]))
            b = float(bo[k])
            if b >= BOND_FORM or key in self.bonds:
                new[key] = b
        self.bonds = new
        return self.bonds


@dataclass
class Molecule:
    atoms: list
    counts: dict
    hill: str
    formula: str          # привычная запись
    name: str | None
    radical: bool

    @property
    def label(self) -> str:
        f = pretty_formula(self.formula)
        dot = "•" if self.radical else ""
        return f + dot


def _components(n, bonds):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, list] = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return list(groups.values())


def find_molecules(sim, bonds) -> list[Molecule]:
    syms = [SYMBOLS[t] for t in sim.typ]
    spare = sim.free_valence
    mols = []
    for atoms in _components(sim.n, bonds):
        counts = Counter(syms[a] for a in atoms)
        hill = hill_formula(counts)
        radical = bool(np.any(spare[atoms] > RADICAL_SPARE))
        mols.append(Molecule(sorted(atoms), dict(counts), hill, display_formula(counts),
                             species_name(hill), radical))
    return mols


@dataclass
class ReactionEvent:
    time: float                       # фс
    reactants: list                   # список формул
    products: list

    def equation(self) -> str:
        def side(lst):
            c = Counter(lst)
            parts = []
            for f, k in sorted(c.items(), key=lambda x: (-x[1], x[0])):
                parts.append((f"{k} " if k > 1 else "") + f)
            return " + ".join(parts)
        return f"{side(self.reactants)} → {side(self.products)}"


class ChemistryMonitor:
    """Периодически анализирует систему: состав, реакции, история."""

    def __init__(self, max_events: int = 200):
        self.tracker = BondTracker()
        self.molecules: list[Molecule] = []
        self.species: Counter = Counter()
        self.events: list[ReactionEvent] = []
        self.max_events = max_events
        self.reaction_counts: Counter = Counter()
        self.history: list[tuple[float, Counter]] = []
        self._mol_of_atom: np.ndarray | None = None
        self._labels: list[str] = []
        self._version = -1

    def reset(self):
        self.tracker.reset()
        self.molecules = []
        self.species = Counter()
        self.events = []
        self.reaction_counts = Counter()
        self.history = []
        self._mol_of_atom = None
        self._version = -1

    def update(self, sim, record_history: bool = True):
        bonds = self.tracker.update(sim)
        mols = find_molecules(sim, bonds)
        mol_of = np.empty(sim.n, dtype=np.int64)
        for k, m in enumerate(mols):
            mol_of[m.atoms] = k
        labels = [m.label for m in mols]
        # реакции: изменилось разбиение атомов на молекулы
        if self._mol_of_atom is not None and self._version == sim.version:
            self._detect_reactions(sim.time, self._mol_of_atom, self._labels, mol_of, labels,
                                   self._prev_mols, mols)
        self._mol_of_atom = mol_of
        self._labels = labels
        self._prev_mols = mols
        self._version = sim.version
        self.molecules = mols
        self.species = Counter(labels)
        if record_history:
            self.history.append((sim.time, self.species.copy()))
            if len(self.history) > 4000:
                self.history = self.history[::2]
        return mols

    def _detect_reactions(self, t, old_of, old_labels, new_of, new_labels, old_mols, new_mols):
        # молекулы, чей набор атомов не изменился, пропускаем
        changed_old = set()
        changed_new = set()
        for k, m in enumerate(new_mols):
            olds = set(old_of[m.atoms].tolist())
            if len(olds) != 1 or len(old_mols[next(iter(olds))].atoms) != len(m.atoms):
                changed_new.add(k)
                changed_old.update(olds)
        if not changed_new:
            return
        # связные группы «старые молекулы ↔ новые молекулы»
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for k in changed_new:
            for o in set(old_of[new_mols[k].atoms].tolist()):
                union(("n", k), ("o", o))
        for o in changed_old:
            for a in old_mols[o].atoms:
                union(("o", o), ("n", int(new_of[a])))
        groups: dict = {}
        for node in parent:
            groups.setdefault(find(node), []).append(node)
        for nodes in groups.values():
            reac = [old_labels[i] for kind, i in nodes if kind == "o"]
            prod = [new_labels[i] for kind, i in nodes if kind == "n"]
            if sorted(reac) == sorted(prod):
                continue
            ev = ReactionEvent(t, reac, prod)
            self.events.append(ev)
            self.reaction_counts[ev.equation()] += 1
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def summary(self, top: int = 12):
        """Список (подпись, количество, название), по убыванию количества."""
        byname = {}
        for m in self.molecules:
            byname.setdefault(m.label, m)
        out = []
        for label, cnt in self.species.most_common(top):
            m = byname[label]
            out.append((label, cnt, m.name or ""))
        return out
