"""Интеграционные тесты: химия должна ВОЗНИКАТЬ из физики.

Никаких «рецептов» в коде нет — проверяем, что поведение качественно
совпадает с реальным.
"""

from chemsim.analysis import ChemistryMonitor
from chemsim.molecules import place
from chemsim.simulation import Simulation, ISOLATED, BATH


def run(mix, T, steps, box=(34, 26, 8), mode=BATH, seed=11, sample=500):
    sim = Simulation(box=box, seed=seed)
    for f, n in mix:
        place(sim, f, n, temperature=T)
    sim.set_mode(mode)
    sim.set_temperature(T)
    mon = ChemistryMonitor()
    mon.update(sim)
    for _ in range(steps // sample):
        sim.step(sample)
        mon.update(sim)
    return sim, mon


def test_radicals_start_hydrogen_oxygen_chain():
    # «искра»: несколько атомов H в горячей смеси H2 + O2 (изолированная система)
    sim, mon = run([("H2", 30), ("O2", 15), ("H", 6)], 3500, 12000, box=(30, 24, 8),
                   mode=ISOLATED)
    formed = set()
    for _, counts in mon.history:
        formed.update(counts)
    # O2 атакуется радикалами: появляются частицы цепного механизма горения
    assert {"HO₂•", "OH•", "H₂O", "O•"} & formed, formed
    assert any("O₂" in ev.equation().split("→")[0] for ev in mon.events)


def test_noble_gas_stays_inert():
    sim, mon = run([("He", 20), ("Ar", 20)], 5000, 4000)
    assert all(not ev for ev in mon.events)
    assert set(mon.species) == {"He", "Ar"}


def test_nitrogen_is_inert_at_moderate_temperature():
    sim, mon = run([("N2", 30)], 1500, 6000)
    assert set(mon.species) == {"N₂"}


def test_exothermic_reaction_heats_isolated_system():
    # H + H -> H2 в изолированной системе: температура должна расти
    sim = Simulation(box=(24, 20, 8), seed=5)
    place(sim, "H", 40, temperature=500)
    sim.set_mode(ISOLATED)
    t0 = sim.temperature
    mon = ChemistryMonitor()
    for _ in range(20):
        sim.step(500)
        mon.update(sim)
    assert mon.species.get("H₂", 0) >= 5
    assert sim.temperature > t0 + 300
