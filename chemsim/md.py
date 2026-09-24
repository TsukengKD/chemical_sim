"""Молекулярная динамика: список соседей, стенки, интегратор, термостат.

Интегрирование — скоростной алгоритм Верле (симплектический, сохраняет энергию),
термостат — ланжевеновский шаг (Ornstein–Uhlenbeck) либо только у стенок
(«тёплые стенки»: газ нагревается/остывает, ударяясь о них, как в реальном
сосуде), либо во всём объёме.
"""

import math

import numpy as np

from .jit import njit
from .kernel import compute_forces
from .units import FORCE_TO_ACC, KB

# статистика, возвращаемая run_md
ST_EPOT, ST_EWALL, ST_WALLF, ST_HEAT, ST_STEPS, ST_ESCAPES, ST_TIME, ST_RETRIES = range(8)
N_STATS = 8


@njit
def build_nlist(pos, rlist, pi, pj):
    """Список пар i<j с r < rlist (ячеечный метод, непериодический ящик).

    Возвращает число пар; если оно больше ёмкости pi/pj, пары не дописываются —
    вызывающий код должен увеличить массивы и повторить.
    """
    N = pos.shape[0]
    cap = pi.shape[0]
    if N < 2:
        return 0
    mn = np.empty(3)
    mx = np.empty(3)
    for k in range(3):
        mn[k] = pos[0, k]
        mx[k] = pos[0, k]
    for a in range(N):
        for k in range(3):
            if pos[a, k] < mn[k]:
                mn[k] = pos[a, k]
            if pos[a, k] > mx[k]:
                mx[k] = pos[a, k]
    nc = np.empty(3, dtype=np.int64)
    for k in range(3):
        nc[k] = max(1, min(64, int((mx[k] - mn[k]) / rlist)))
    ncell = nc[0] * nc[1] * nc[2]
    head = -np.ones(ncell, dtype=np.int64)
    nxt = -np.ones(N, dtype=np.int64)
    cidx = np.empty((N, 3), dtype=np.int64)
    for a in range(N):
        for k in range(3):
            c = int((pos[a, k] - mn[k]) / (mx[k] - mn[k] + 1e-9) * nc[k])
            if c >= nc[k]:
                c = nc[k] - 1
            if c < 0:
                c = 0
            cidx[a, k] = c
        cc = (cidx[a, 0] * nc[1] + cidx[a, 1]) * nc[2] + cidx[a, 2]
        nxt[a] = head[cc]
        head[cc] = a
    r2 = rlist * rlist
    n = 0
    for a in range(N):
        for dx in range(-1, 2):
            cx = cidx[a, 0] + dx
            if cx < 0 or cx >= nc[0]:
                continue
            for dy in range(-1, 2):
                cy = cidx[a, 1] + dy
                if cy < 0 or cy >= nc[1]:
                    continue
                for dz in range(-1, 2):
                    cz = cidx[a, 2] + dz
                    if cz < 0 or cz >= nc[2]:
                        continue
                    b = head[(cx * nc[1] + cy) * nc[2] + cz]
                    while b >= 0:
                        if b > a:
                            ddx = pos[b, 0] - pos[a, 0]
                            ddy = pos[b, 1] - pos[a, 1]
                            ddz = pos[b, 2] - pos[a, 2]
                            if ddx * ddx + ddy * ddy + ddz * ddz < r2:
                                if n < cap:
                                    pi[n] = a
                                    pj[n] = b
                                n += 1
                        b = nxt[b]
    return n


@njit
def wall_forces(pos, box, kw, dw, forces):
    """Мягкие стенки E = kw·(dw − d)³ при d < dw (d — расстояние до стенки).

    Добавляет силы; возвращает (энергия, сумма модулей сил на стенки)."""
    N = pos.shape[0]
    e = 0.0
    fsum = 0.0
    for a in range(N):
        for k in range(3):
            d = pos[a, k]
            if d < dw:
                pen = dw - d
                e += kw * pen * pen * pen
                f = 3.0 * kw * pen * pen
                forces[a, k] += f
                fsum += f
            d = box[k] - pos[a, k]
            if d < dw:
                pen = dw - d
                e += kw * pen * pen * pen
                f = 3.0 * kw * pen * pen
                forces[a, k] -= f
                fsum += f
    return e, fsum


@njit
def kinetic_energy(vel, invm):
    ke = 0.0
    for a in range(vel.shape[0]):
        ke += 0.5 / invm[a] * (vel[a, 0] ** 2 + vel[a, 1] ** 2 + vel[a, 2] ** 2)
    return ke / FORCE_TO_ACC


@njit
def seed_rng(seed):
    np.random.seed(seed)


@njit
def run_md(pos, vel, typ, invm, pi, pj, npair, elempar, pairpar, scal,
           forces, bond_order, ebreak, atomout,
           box, kw, dw, wall_T, gamma_wall, T_glob, gamma_glob,
           dt_max, dx_max, e_tol, nsteps, ref_pos, max_disp, stats):
    """До nsteps шагов Верле.  Останавливается раньше, если атомы сместились
    настолько, что список соседей нужно перестроить.  forces на входе должны
    соответствовать pos.  Возвращает число выполненных шагов.

    Шаг адаптивный: dt = min(dt_max, dx_max / v_max, √(2·dx_max / a_max)), т.е.
    ни один атом не смещается за шаг больше чем на dx_max (важно для горячих
    атомов водорода при взрывах).  Кроме того, шаг «под контролем энергии»:
    если за один шаг полная энергия изменилась больше чем на e_tol (признак
    очень крутого участка поверхности — например, при сильном изгибе молекулы),
    шаг отменяется и повторяется с вдвое меньшим dt.  Прошедшее время —
    stats[ST_TIME]."""
    N = pos.shape[0]
    pos0 = np.empty_like(pos)
    vel0 = np.empty_like(vel)
    frc0 = np.empty_like(forces)
    epot = compute_forces(pos, typ, pi, pj, npair, elempar, pairpar, scal,
                          forces, bond_order, ebreak, atomout)
    ewall, _ = wall_forces(pos, box, kw, dw, forces)
    retries = 0
    md2 = max_disp * max_disp
    elapsed = 0.0
    heat = 0.0
    wallf = 0.0
    escapes = 0
    done = 0
    for step in range(nsteps):
        # нужно ли перестроить список соседей?
        worst = 0.0
        for a in range(N):
            dx = pos[a, 0] - ref_pos[a, 0]
            dy = pos[a, 1] - ref_pos[a, 1]
            dz = pos[a, 2] - ref_pos[a, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > worst:
                worst = d2
        if worst > md2:
            break
        # адаптивный шаг
        v2max = 0.0
        a2max = 0.0
        for a in range(N):
            v2 = vel[a, 0] ** 2 + vel[a, 1] ** 2 + vel[a, 2] ** 2
            if v2 > v2max:
                v2max = v2
            f2 = (forces[a, 0] ** 2 + forces[a, 1] ** 2 + forces[a, 2] ** 2) * invm[a] * invm[a]
            if f2 > a2max:
                a2max = f2
        dt = dt_max
        if v2max > 0.0:
            dt = min(dt, dx_max / math.sqrt(v2max))
        if a2max > 0.0:
            dt = min(dt, math.sqrt(2.0 * dx_max / (math.sqrt(a2max) * FORCE_TO_ACC)))
        # шаг Верле с контролем энергии
        e_before = epot + ewall + kinetic_energy(vel, invm)
        pos0[:, :] = pos
        vel0[:, :] = vel
        frc0[:, :] = forces
        ep0 = epot
        ew0 = ewall
        for attempt in range(8):
            hdt = 0.5 * dt * FORCE_TO_ACC
            for a in range(N):
                s = hdt * invm[a]
                for k in range(3):
                    vel[a, k] += s * forces[a, k]
                    pos[a, k] += dt * vel[a, k]
            epot = compute_forces(pos, typ, pi, pj, npair, elempar, pairpar, scal,
                                  forces, bond_order, ebreak, atomout)
            ewall, fs = wall_forces(pos, box, kw, dw, forces)
            for a in range(N):
                s = hdt * invm[a]
                for k in range(3):
                    vel[a, k] += s * forces[a, k]
            if e_tol <= 0.0 or attempt == 7:
                break
            e_after = epot + ewall + kinetic_energy(vel, invm)
            if abs(e_after - e_before) <= e_tol:
                break
            # слишком крутой участок: откат и половинный шаг
            pos[:, :] = pos0
            vel[:, :] = vel0
            forces[:, :] = frc0
            epot = ep0
            ewall = ew0
            dt *= 0.5
            retries += 1
        wallf += fs * dt
        # аварийная защита: атом «пробил» стенку (не должно случаться)
        for a in range(N):
            for k in range(3):
                if pos[a, k] < -1.0:
                    pos[a, k] = 0.0
                    vel[a, k] = abs(vel[a, k])
                    escapes += 1
                elif pos[a, k] > box[k] + 1.0:
                    pos[a, k] = box[k]
                    vel[a, k] = -abs(vel[a, k])
                    escapes += 1
        c_wall = math.exp(-gamma_wall * dt)
        c_glob = math.exp(-gamma_glob * dt)
        # ланжевеновский термостат: у стенок и/или во всём объёме
        if gamma_wall > 0.0 or gamma_glob > 0.0:
            for a in range(N):
                Tt = -1.0
                c1 = 1.0
                if gamma_wall > 0.0:
                    for k in range(3):
                        if pos[a, k] < dw:
                            Tt = wall_T[2 * k]
                            break
                        if box[k] - pos[a, k] < dw:
                            Tt = wall_T[2 * k + 1]
                            break
                    if Tt >= 0.0:
                        c1 = c_wall
                if Tt < 0.0 and gamma_glob > 0.0:
                    Tt = T_glob
                    c1 = c_glob
                if Tt >= 0.0:
                    m = 1.0 / invm[a]
                    sig = math.sqrt((1.0 - c1 * c1) * KB * Tt / m * FORCE_TO_ACC)
                    ke0 = 0.0
                    ke1 = 0.0
                    for k in range(3):
                        v0 = vel[a, k]
                        v1 = c1 * v0 + sig * np.random.standard_normal()
                        vel[a, k] = v1
                        ke0 += v0 * v0
                        ke1 += v1 * v1
                    heat += 0.5 * m * (ke1 - ke0) / FORCE_TO_ACC
        done += 1
        elapsed += dt
    stats[ST_TIME] = elapsed
    stats[ST_EPOT] = epot
    stats[ST_EWALL] = ewall
    stats[ST_WALLF] = wallf
    stats[ST_HEAT] = heat
    stats[ST_STEPS] = done
    stats[ST_ESCAPES] = escapes
    stats[ST_RETRIES] = retries
    return done


