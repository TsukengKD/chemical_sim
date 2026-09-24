"""Анализ: формулы, названия, молекулы, регистрация реакций."""

import numpy as np

from chemsim.analysis import ChemistryMonitor
from chemsim.molecules import place, template
from chemsim.names import hill_formula, display_formula, parse_formula, species_name
from chemsim.simulation import Simulation


def test_formulas():
    assert hill_formula({"C": 2, "H": 6, "O": 1}) == "C2H6O"
    assert hill_formula({"Na": 1, "Cl": 1}) == "ClNa"
    assert display_formula({"Na": 1, "Cl": 1}) == "NaCl"
    assert display_formula({"H": 3, "N": 1}) == "NH3"
    assert display_formula({"H": 2, "O": 1}) == "H2O"
    assert display_formula({"C": 1, "O": 2}) == "CO2"
    assert species_name(hill_formula(parse_formula("NaOH"))) == "гидроксид натрия"
    assert species_name("H2O") == "вода"


def test_templates_are_relaxed_molecules():
    for name in ["H2O", "CH4", "C6H6", "NH3"]:
        syms, x = template(name)
        assert len(syms) == len(x)


def test_monitor_identifies_species():
    sim = Simulation(box=(30, 30, 8), seed=1)
    place(sim, "H2O", 5, temperature=300)
    place(sim, "CH4", 3, temperature=300)
    place(sim, "N2", 2, temperature=300)
    mon = ChemistryMonitor()
    mon.update(sim)
    counts = {label: n for label, n, _ in mon.summary()}
    assert counts == {"H₂O": 5, "CH₄": 3, "N₂": 2}


def test_reaction_detection():
    sim = Simulation(box=(20, 20, 8), seed=1)
    # два атома H далеко друг от друга
    sim.add_atoms(["H", "H"], [[5, 10, 4], [15, 10, 4]], velocities=np.zeros((2, 3)))
    mon = ChemistryMonitor()
    mon.update(sim)
    # переносим их на расстояние связи (как если бы они столкнулись)
    sim.pos[1] = [5.74, 10, 4]
    sim._rebuild()
    sim._compute_forces()
    mon.update(sim)
    assert mon.events, "реакция H + H -> H2 должна быть зарегистрирована"
    assert mon.events[-1].equation() == "2 H• → H₂"
