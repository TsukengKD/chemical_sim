"""Командная строка: опыты без графики, проверка модели, справка.

    python -m chemsim                    — игра (графический интерфейс)
    python -m chemsim run --scenario knallgas --sparks 4 --time 10
    python -m chemsim run --mix "H2:40,O2:20" --temp 3500 --mode isolated --time 5
    python -m chemsim validate [--md report.md]
    python -m chemsim scenarios          — список готовых опытов
    python -m chemsim elements           — параметры элементов и связей
"""

from __future__ import annotations

import argparse
import csv
import sys
import time


def _parse_mix(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, n = part.partition(":")
        out.append((name.strip(), int(n) if n else 1))
    return out


def _parse_box(text):
    parts = [float(x) for x in text.lower().replace("×", "x").split("x")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("размер ящика: ДЛИНАxВЫСОТАxТОЛЩИНА, например 44x30x8")
    return tuple(parts)


def cmd_run(args):
    from .analysis import ChemistryMonitor
    from .molecules import place
    from .scenarios import BY_KEY
    from .simulation import Simulation

    sim = Simulation(box=args.box or (44.0, 30.0, 8.0), seed=args.seed)
    if args.scenario:
        sc = BY_KEY[args.scenario]
        if args.box:
            sc.box = args.box
        sc.apply(sim)
        print(f"Опыт: {sc.title}\n{sc.description}\n")
    if args.mix:
        T0 = args.temp if args.temp is not None else 300.0
        for formula, n in _parse_mix(args.mix):
            got = place(sim, formula, n, temperature=T0)
            if got < n:
                print(f"! поместилось только {got} из {n} {formula}", file=sys.stderr)
    if args.mode:
        sim.set_mode({"isolated": "isolated", "walls": "walls", "bath": "bath"}[args.mode])
    if args.temp is not None:
        sim.set_temperature(args.temp)
    if args.precise:
        sim.dx_max = 0.012
        sim.e_tol = 0.5
    mon = ChemistryMonitor()
    mon.update(sim)
    for _ in range(args.sparks):
        k = int(sim.rng.integers(sim.n))
        sim.spark(sim.pos[k, :2], 4.0, 25000.0)
        sim.step(200)
    print(f"атомов: {sim.n}, режим: {sim.mode}, ящик: {sim.box[0]:.0f}×{sim.box[1]:.0f}×"
          f"{sim.box[2]:.0f} Å, T = {sim.temperature:.0f} К\n")
    writer = None
    fh = None
    if args.csv:
        fh = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(["t_ps", "T_K", "P_atm", "Epot", "Ekin", "species"])
    t_end = sim.time + args.time * 1000.0
    next_report = sim.time
    wall0 = time.time()
    while sim.time < t_end:
        sim.step(200)
        mon.update(sim)
        if sim.time >= next_report:
            top = ", ".join(f"{c} {l}" for l, c, _ in mon.summary(8))
            print(f"{sim.time / 1000:8.3f} пс  T={sim.temperature:6.0f} К  P={sim.pressure_atm:7.0f} атм"
                  f"  | {top}")
            if writer:
                writer.writerow([f"{sim.time / 1000:.4f}", f"{sim.temperature:.1f}",
                                 f"{sim.pressure_atm:.1f}", f"{sim.epot:.2f}", f"{sim.ekin:.2f}",
                                 "; ".join(f"{l}:{c}" for l, c, _ in mon.summary(50))])
            next_report += args.every * 1000.0
    if fh:
        fh.close()
    print(f"\nпрошло {time.time() - wall0:.1f} с реального времени")
    print("\nИтоговый состав:")
    for label, cnt, name in mon.summary(30):
        print(f"  {cnt:5d}  {label:12s} {name}")
    print("\nРеакции (сколько раз наблюдались):")
    if not mon.reaction_counts:
        print("  — реакций не было")
    for eq, c in mon.reaction_counts.most_common(args.top):
        print(f"  {c:5d}  {eq}")


def cmd_scenarios(args):
    from .scenarios import SCENARIOS
    for sc in SCENARIOS:
        reag = ", ".join(f"{n} {f}" for f, n in sc.reagents)
        print(f"{sc.key:13s} {sc.title}\n{'':13s} реактивы: {reag}; T = {sc.temperature:.0f} К; "
              f"режим: {sc.mode}\n")


def cmd_elements(args):
    from .elements import ELEMENT_LIST
    from .params import default_params
    p = default_params()
    print(f"{'эл':3s} {'назв':10s} {'масса':>8s} {'χ':>5s} {'вал':>3s} {'r1,Å':>5s}  {'D(A–A)':>7s}")
    for e in ELEMENT_LIST:
        info = p.pair(e.symbol, e.symbol)
        d = f"{info.orders[1][0]:.0f}" if info.bondable else "—"
        print(f"{e.symbol:3s} {e.name:10s} {e.mass:8.3f} {e.chi:5.2f} {e.valence:3d} {e.r1:5.2f}  {d:>7s}")
    if args.pair:
        a, b = args.pair.split("-")
        info = p.pair(a, b)
        print(f"\nСвязь {a}–{b}:")
        for n, (D, r, src) in sorted(info.orders.items()):
            print(f"  порядок {n}: D = {D:.0f} кДж/моль, r = {r:.3f} Å  ({src})")
        if info.bondable:
            print(f"  параметр Морзе a1 = {info.a1:.3f} 1/Å, силовая постоянная k = {info.k1:.0f} "
                  f"кДж/(моль·Å²)")
        print(f"  ван-дер-Ваальс: ε = {info.eps:.3f} кДж/моль, x = {info.xv:.2f} Å")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chemsim", description="Химический симулятор")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="опыт без графики")
    r.add_argument("--scenario", help="ключ готового опыта (см. `scenarios`)")
    r.add_argument("--mix", help='реактивы, например "H2:40,O2:20" (формулы или символы атомов)')
    r.add_argument("--temp", type=float, help="температура газа и стенок, К")
    r.add_argument("--mode", choices=["isolated", "walls", "bath"], help="теплообмен")
    r.add_argument("--box", type=_parse_box, help="размер ящика, например 44x30x8 (Å)")
    r.add_argument("--time", type=float, default=5.0, help="длительность, пс")
    r.add_argument("--every", type=float, default=0.5, help="интервал вывода, пс")
    r.add_argument("--sparks", type=int, default=0, help="сколько искр сделать в начале")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--csv", help="сохранить историю в CSV")
    r.add_argument("--top", type=int, default=20, help="сколько реакций показать")
    r.add_argument("--precise", action="store_true", help="мельче шаг (точнее, медленнее)")
    v = sub.add_parser("validate", help="сравнение модели с экспериментом")
    v.add_argument("--md", help="сохранить отчёт в Markdown")
    sub.add_parser("scenarios", help="список опытов")
    e = sub.add_parser("elements", help="параметры элементов")
    e.add_argument("--pair", help="показать параметры связи, например C-O")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        if not args.scenario and not args.mix:
            ap.error("нужно указать --scenario или --mix")
        cmd_run(args)
    elif args.cmd == "validate":
        from . import validate
        text = validate.report()
        print(text)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as fh:
                fh.write(text)
    elif args.cmd == "scenarios":
        cmd_scenarios(args)
    elif args.cmd == "elements":
        cmd_elements(args)
    else:
        from .ui.app import main as gui
        gui()


if __name__ == "__main__":
    main()
