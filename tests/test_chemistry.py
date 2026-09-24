"""Модель против эксперимента: молекулы, реакции, барьеры, валентность."""

import numpy as np
import pytest

from chemsim import validate
from chemsim.minimize import Evaluator, fire_minimize


@pytest.fixture(scope="module")
def atomization():
    return {r.name: r for r in validate.atomization_table()}


@pytest.mark.parametrize("name", ["H2", "O2", "N2", "HCl", "HF", "NaCl"])
def test_diatomics_reproduce_input_data(atomization, name):
    assert abs(atomization[name].rel) < 1.5


@pytest.mark.parametrize("name", ["H2O", "CH4", "NH3", "CO2", "C2H6", "C2H4", "C2H2",
                                  "HCN", "H2O2", "CH3OH", "C6H6", "H2S"])
def test_polyatomic_atomization_energies(atomization, name):
    # энергии многоатомных молекул — предсказание модели (не подгонка)
    assert abs(atomization[name].rel) < 5.0


def test_geometries():
    for row in validate.geometry_table():
        if "(°)" in row.name:
            assert abs(row.err) < 3.0, row
        else:
            assert abs(row.err) < 0.05, row


@pytest.mark.parametrize("title", ["2 H2 + O2 → 2 H2O", "CH4 + 2 O2 → CO2 + 2 H2O",
                                   "H2 + Cl2 → 2 HCl", "H2 + F2 → 2 HF",
                                   "C2H4 + H2 → C2H6"])
def test_reaction_enthalpies(title):
    rows = {r.name: r for r in validate.reaction_table()}
    assert abs(rows[title].rel) < 8.0


def test_valence_rules():
    for title, de in validate.valence_table():
        assert de > 0.0, title


def test_barriers_are_physical():
    rows = {r.name: r for r in validate.barrier_table()}
    for r in rows.values():
        assert 0.0 < r.model < 120.0, r
    # барьер H + H2 — эталонная задача химической кинетики (эксп. ≈ 40 кДж/моль)
    assert rows["H + H2 → H2 + H"].model < 80.0
    assert 30.0 < rows["CH4 + H → CH3 + H2"].model < 80.0


def test_radical_recombination_has_no_barrier():
    ev = Evaluator(["H", "H"])
    rs = np.linspace(0.8, 6.0, 105)
    es = np.array([ev(np.array([[0, 0, 0], [r, 0, 0.0]]))[0] for r in rs])
    # при сближении двух радикалов энергия нигде не поднимается выше, чем у
    # разделённых атомов (с точностью до долей кДж/моль ≪ kT)
    assert es.max() - es[-1] < 0.5
    assert es[0] < -400.0


def test_noble_gases_do_not_bond():
    ev = Evaluator(["He", "He", "Ne", "Ar"])
    pos = np.array([[0, 0, 0], [3.0, 0, 0], [0, 3.2, 0], [3.0, 3.5, 0.5]])
    x, e, _ = fire_minimize(ev, pos, fmax=1e-3)
    assert -10.0 < e < 0.0            # только слабое ван-дер-ваальсово притяжение
    assert max(ev.bond_orders(x).values(), default=0.0) < 1e-6


def test_hydrogen_bond_in_water_dimer():
    ev = Evaluator(["O", "H", "H", "O", "H", "H"])
    pos = np.array([[0, 0, 0], [.96, 0, 0], [-.3, .9, 0], [2.9, 0, 0], [3.2, .9, 0],
                    [3.2, -0.4, 0.8]])
    x, e, _ = fire_minimize(ev, pos, fmax=1e-3)
    e1, _, _ = validate.molecule_energy("H2O")
    assert e - 2 * e1 < -2.0          # связанный димер (эксп. −21 кДж/моль)
