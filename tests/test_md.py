"""Молекулярная динамика: сохранение энергии, стенки, термостат, список соседей."""

import numpy as np

from chemsim.kernel import compute_forces, all_pairs, N_EBREAK, N_ATOMOUT
from chemsim.molecules import place
from chemsim.simulation import Simulation, ISOLATED, WALLS, BATH


def make_sim(mix, T, box=(30, 24, 8), seed=3):
    sim = Simulation(box=box, seed=seed)
    for f, n in mix:
        placed = place(sim, f, n, temperature=T)
        assert placed == n
    return sim


def test_energy_conservation_isolated():
    sim = make_sim([("H2O", 12), ("CH4", 4), ("O2", 6)], 1200)
    sim.set_mode(ISOLATED)
    sim.step(20)
    e0 = sim.etotal
    es = []
    for _ in range(10):
        sim.step(200)
        es.append(sim.etotal)
    drift = max(abs(e - e0) for e in es)
    assert drift < 1e-3 * abs(e0) + 5.0, drift
    assert sim.escapes == 0


def test_neighbor_list_matches_all_pairs():
    sim = make_sim([("H2O", 10), ("N2", 6)], 800)
    sim.step(300)
    p = sim.params
    pi, pj = all_pairs(sim.n)
    f = np.zeros((sim.n, 3))
    e = compute_forces(sim.pos, sim.typ, pi, pj, len(pi), p.elempar, p.pairpar, p.scalars,
                       f, np.zeros(len(pi)), np.zeros(N_EBREAK), np.zeros((sim.n, N_ATOMOUT)))
    sim._rebuild()
    sim._compute_forces()
    assert abs(e - sim.epot) < 1e-6
    fw = sim.forces.copy()
    from chemsim.md import wall_forces
    wall_forces(sim.pos, sim.box, sim.kw, sim.dw, f)
    assert np.abs(f - fw).max() < 1e-6


def test_atoms_stay_in_box():
    sim = make_sim([("H2", 20)], 3000, box=(20, 20, 6))
    sim.set_mode(ISOLATED)
    sim.step(2000)
    assert (sim.pos > -0.5).all() and (sim.pos < sim.box + 0.5).all()


def test_wall_thermostat_heats_gas():
    sim = make_sim([("Ar", 25)], 100, box=(24, 24, 8))
    sim.set_mode(WALLS)
    sim.set_temperature(900)
    for _ in range(12):
        sim.step(1000)
    assert 600 < sim.temperature < 1200
    assert sim.heat_in > 0


def test_bath_thermostat():
    sim = make_sim([("N2", 20)], 300, box=(24, 20, 8))
    sim.set_mode(BATH)
    sim.set_temperature(1500)
    temps = []
    for _ in range(20):
        sim.step(500)
        temps.append(sim.temperature)
    assert abs(np.mean(temps[-8:]) - 1500) < 250


def test_pressure_ideal_gas_order_of_magnitude():
    # разреженный аргон: P ≈ n k T (идеальный газ)
    sim = make_sim([("Ar", 30)], 600, box=(40, 40, 10))
    sim.set_mode(BATH)
    sim.set_temperature(600)
    ps = []
    for _ in range(10):
        sim.step(1000)
        ps.append(sim.pressure_atm)
    from chemsim.units import KB, KJMOL_A3_TO_ATM
    # эффективный объём меньше из-за мягких стенок
    v_eff = np.prod(sim.box - 2 * 0.7)
    p_ideal = sim.n * KB * 600 / v_eff * KJMOL_A3_TO_ATM
    assert 0.6 * p_ideal < np.mean(ps[3:]) < 1.5 * p_ideal


def test_add_remove_and_spark():
    sim = make_sim([("H2", 5)], 300)
    n0 = sim.n
    sim.remove_atoms([0, 1])
    assert sim.n == n0 - 2
    sim.add_atoms(["O"], [[15, 12, 4]])
    hit = sim.spark((15, 12), 5.0, 5000)
    assert hit >= 1
    sim.step(50)
