"""Аналитические силы должны совпадать с численной производной энергии."""

import numpy as np
import pytest

from chemsim.minimize import Evaluator


def numeric_forces(ev, pos, h=1e-6):
    num = np.zeros_like(pos)
    for a in range(len(pos)):
        for k in range(3):
            pp = pos.copy()
            pp[a, k] += h
            pm = pos.copy()
            pm[a, k] -= h
            num[a, k] = -(ev(pp)[0] - ev(pm)[0]) / (2 * h)
    return num


CASES = {
    "water_dimer": (["O", "H", "H", "O", "H", "H"],
                    [[0, 0, 0], [.96, 0, 0], [-.3, .9, 0], [2.9, 0.1, 0], [3.2, .9, 0.2],
                     [3.2, -0.4, 0.8]]),
    "H3_triangle": (["H", "H", "H"], [[0, 0, 0], [0.9, 0.05, 0], [0.45, 0.8, 0.1]]),
    "H_plus_H2": (["H", "H", "H"], [[-1.3, 0.1, 0], [0, 0, 0], [0.78, 0, 0.05]]),
    "NaCl_cluster": (["Na", "Cl", "Na", "Cl"],
                     [[0, 0, 0], [2.8, 0.1, 0], [2.7, 2.9, 0.2], [0.1, 2.8, -0.1]]),
    "methanol": (["C", "O", "H", "H", "H", "H"],
                 [[0, 0, 0], [1.43, 0, 0], [-.4, 1.0, 0], [-.4, -.5, .87], [-.4, -.5, -.87],
                  [1.75, .9, 0.1]]),
    "ethylene_plus_H": (["C", "C", "H", "H", "H", "H", "H"],
                        [[0, 0, 0], [1.34, 0, 0], [-.55, .93, 0], [-.55, -.93, 0],
                         [1.89, .93, 0], [1.89, -.93, 0], [0.6, 0.3, 1.6]]),
    "N2_O2": (["N", "N", "O", "O"], [[0, 0, 0], [1.1, 0, 0], [0.5, 1.6, 0.2], [1.6, 1.9, 0]]),
    "radicals": (["O", "H", "C", "H", "H", "H"],
                 [[0, 0, 0], [0.97, 0, 0], [2.2, 1.0, 0.3], [2.6, 2.0, 0.3], [2.6, 0.5, 1.2],
                  [2.6, 0.5, -0.7]]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_forces_match_finite_differences(name):
    syms, pos = CASES[name]
    ev = Evaluator(syms)
    pos = np.array(pos, dtype=float)
    e, f = ev(pos)
    num = numeric_forces(ev, pos)
    scale = np.abs(f).max() + 1.0
    assert np.abs(num - f).max() / scale < 1e-5


def test_forces_random_clusters():
    rng = np.random.default_rng(7)
    elements = ["H", "C", "N", "O", "F", "Cl", "Na", "S", "He", "P"]
    for _ in range(15):
        n = int(rng.integers(3, 9))
        syms = list(rng.choice(elements, size=n))
        pos = rng.normal(size=(n, 3)) * 1.3
        ev = Evaluator(syms)
        e, f = ev(pos)
        num = numeric_forces(ev, pos)
        scale = np.abs(f).max() + 1.0
        assert np.abs(num - f).max() / scale < 1e-5, syms


def test_newton_third_law():
    syms, pos = CASES["ethylene_plus_H"]
    ev = Evaluator(syms)
    _, f = ev(np.array(pos, dtype=float))
    assert np.abs(f.sum(axis=0)).max() < 1e-8
