"""Ядро реакционного силового поля: энергия и аналитические силы.

Физическая модель (подробно — в README):

    E = Σ_пар [ R1(r) − (bσ + τ)·A1(r) − h1(bπ)·P(r) − h2(bπ)·Q(r)
                + (1−zL)(1−α∨)·E_отт(r) + E_дисп(r) + (1−z)·k_e·q_i·q_j·u_DSF(r) ]
        + Σ_углов K·bσ_ij·bσ_ik·(cos θ − cos θ0)²
        − Σ_пар 1-3  X_jk·(отталкивание + кулон)_jk

* R1 − A1 — потенциал Морзе одинарной связи в форме Абелла–Терсоффа:
  отталкивание R1 не зависит от окружения, притяжение A1 умножается на порядок
  σ-связи bσ.
* bσ_ij = z(r)·min(f_ij, f_ji),  f_ij = (Z_ij/V_i)^(−δ) при Z_ij > V_i — валентность:
  если у атома больше партнёров, чем валентность V, связи ослабевают и удлиняются
  (закон Полинга/Абелла).  Z_ij = Σ_k z_ik·w·g(θ_jik) — число соседей атома i,
  «видимое» из связи i–j; соседи под острым углом (θ < 90°) конкурируют сильнее
  (g = 1 + c·cos²θ, как угловая функция потенциала Терсоффа), а более слабая связь
  не отнимает валентность у более прочной (w = min(1, D_ik/D_ij)).  Для ионных
  связей показатель δ уменьшается с ионностью (ионная связь ненаправленная).
* Свободная валентность s_i = V_i − Σ bσ распределяется между соседями, у которых
  тоже есть свободная валентность; так возникают π-связи (двойные и тройные),
  ароматические полуторные связи и радикальные центры.
* τ — «зарождающаяся связь» радикалов на больших расстояниях (хвост кривой
  Морзе): τ = (1−z)·zL·c_i·c_j·n, где c — свободная валентность атомов, а n
  нормирует сумму хвостов радикала при нескольких партнёрах; партнёры под острыми
  углами друг к другу конкурируют сильнее (как g(θ) у Терсоффа).  Рекомбинация
  радикалов идёт без барьера, но один радикал не может притягивать сразу многих.
* c_i = clamp(V_i − Σ_{k≠j} b_ik) — свободная валентность атома i для партнёра j;
  α∨ = c_i + c_j − c_i·c_j (хотя бы один радикал) выключает замкнуто-оболочечное
  (паулиевское) отталкивание, которое действует только между насыщенными молекулами.
* Заряды: по каждой связи на менее электроотрицательный атом переходит
  I = 1 − exp(−Δχ²/4) (ионный характер по Полингу) на единицу валентности связи.
  Кулон действует между несвязанными атомами (водородная связь, ионные кристаллы).
* Углы: теория отталкивания электронных пар (VSEPR).  Стерическое число
  SN = (число σ-связей) + (число неподелённых пар), cos θ0 = −1/(SN − 1):
  SN=2 → 180°, 3 → 120°, 4 → 109.47°.  Как в классических силовых полях,
  взаимодействия геминальных (1-3) атомов исключаются (плавно, с весом связей).

Все функции гладкие (C¹), поэтому полная энергия сохраняется при интегрировании,
а реакции идут непрерывно — через переходные состояния, без «переключателей».

Производные вычисляются обратным проходом (ручное обратное дифференцирование)
по всей цепочке.  Корректность проверяется сравнением с конечными разностями.
"""

import math

import numpy as np

from .jit import njit
from .params import (P_BOND, P_D1, P_R1, P_A1, P_RON, P_ROFF, P_PIMAX, P_CP, P_AP, P_R2,
                     P_CQ, P_AQ, P_R3, P_EPS, P_X6, P_GR6, P_GD6, P_RONL, P_ROFFL, P_QT,
                     P_DI, E_VAL, E_EVAL, E_DELTA, E_KANG, E_GACUTE, E_METAL,
                     S_RSW, S_RCUT, S_CREP, S_CDISP, S_KMIN, S_KOVER, S_KSPARE, S_KPI,
                     S_KLP, S_EPSW, S_KE, S_AS3)

# сдвиг cos θ0 на одну неподелённую пару: из углов NH3 (107°) и H2O (104.5°)
# при идеальном тетраэдре 109.47° (правило VSEPR: пары занимают больший объём)
LP_SQUEEZE = 0.0415
# доля кулоновского взаимодействия, остающаяся у пар 1-4 (как в AMBER/OPLS)
SCALE14 = 0.5

# индексы разложения энергии
EB_SIGMA, EB_PI, EB_REP, EB_DISP, EB_ANGLE, EB_COUL = 0, 1, 2, 3, 4, 5
N_EBREAK = 6

# per-atom выходные величины (A_SPARE — свободная валентность V − Σb, радикальность)
A_Z, A_SPARE, A_BTOT, A_SN, A_Q = 0, 1, 2, 3, 4
N_ATOMOUT = 5


# ------------------------------------------------------------- гладкие функции
@njit(inline="always")
def _phi(d, k):
    """Гладкая (C¹) замена |d| около нуля: φ(0)=0, φ'(0)=0, φ(d)=d при d≥k."""
    if d >= k:
        return d, 1.0
    kk = k * k
    return d * d * (2.0 * k - d) / kk, (4.0 * k * d - 3.0 * d * d) / kk


@njit(inline="always")
def _ramp(x, k):
    """Гладкий max(x, 0), точный вне интервала (0, k)."""
    if x <= 0.0:
        return 0.0, 0.0
    return _phi(x, k)


@njit(inline="always")
def _smin(a, b, k):
    """Гладкий min(a, b): точен при a = b и при |a − b| ≥ k."""
    d = a - b
    if d >= 0.0:
        ph, dph = _phi(d, k)
        return 0.5 * (a + b - ph), 0.5 * (1.0 - dph), 0.5 * (1.0 + dph)
    ph, dph = _phi(-d, k)
    return 0.5 * (a + b - ph), 0.5 * (1.0 + dph), 0.5 * (1.0 - dph)


@njit(inline="always")
def _clamp01(x, k):
    v1, d1 = _ramp(x, k)
    v2, d2 = _ramp(x - 1.0, k)
    return v1 - v2, d1 - d2


@njit(inline="always")
def _taper(r, ron, roff):
    """1 при r ≤ ron, 0 при r ≥ roff, между ними — полином 5-й степени."""
    if r <= ron:
        return 1.0, 0.0
    if r >= roff:
        return 0.0, 0.0
    w = roff - ron
    t = (r - ron) / w
    t2 = t * t
    s = t2 * t * (10.0 - 15.0 * t + 6.0 * t2)
    ds = 30.0 * t2 * (1.0 - t) * (1.0 - t) / w
    return 1.0 - s, -ds


DSF_ALPHA = 0.2   # параметр затухания DSF, 1/Å


@njit(inline="always")
def _dsf_h(r, as3):
    """h(r) = erfc(αr)/r_s,  r_s = (r³ + a³)^(1/3) — экранирование на малых r."""
    rs3 = r * r * r + as3
    rs = rs3 ** (1.0 / 3.0)
    er = math.erfc(DSF_ALPHA * r)
    h = er / rs
    dh = -1.1283791670955126 * DSF_ALPHA * math.exp(-DSF_ALPHA * DSF_ALPHA * r * r) / rs \
        - er * (r * r / rs3) / rs
    return h, dh


@njit(inline="always")
def _satur(zb, V, dl, kover):
    """f = (1 + ramp(Z/V − 1))^(−δ) и df/dZ."""
    e, de = _ramp(zb / V - 1.0, kover)
    f = (1.0 + e) ** (-dl)
    return f, -dl * f / (1.0 + e) * de / V


@njit(inline="always")
def _side_vec(p, a, pi, pj, dvec):
    """Вектор от атома a (одного из концов пары p) к другому концу и номер соседа."""
    if pi[p] == a:
        return dvec[p, 0], dvec[p, 1], dvec[p, 2], pj[p]
    return -dvec[p, 0], -dvec[p, 1], -dvec[p, 2], pi[p]


@njit(inline="always")
def _add_cos_force(forces, c, o1, o2, ux, uy, uz, ru, vx, vy, vz, rv, cs, gcs):
    """Силы от энергии E(cos θ) с dE/dcos = gcs; θ — угол o1–c–o2."""
    inv = 1.0 / (ru * rv)
    cu = cs / (ru * ru)
    cv = cs / (rv * rv)
    dux = vx * inv - cu * ux
    duy = vy * inv - cu * uy
    duz = vz * inv - cu * uz
    dvx = ux * inv - cv * vx
    dvy = uy * inv - cv * vy
    dvz = uz * inv - cv * vz
    forces[o1, 0] -= gcs * dux
    forces[o1, 1] -= gcs * duy
    forces[o1, 2] -= gcs * duz
    forces[o2, 0] -= gcs * dvx
    forces[o2, 1] -= gcs * dvy
    forces[o2, 2] -= gcs * dvz
    forces[c, 0] += gcs * (dux + dvx)
    forces[c, 1] += gcs * (duy + dvy)
    forces[c, 2] += gcs * (duz + dvz)


# ------------------------------------------------------------------ ядро
@njit
def compute_forces(pos, typ, pi, pj, npair, elempar, pairpar, scal,
                   forces, bond_order, ebreak, atomout):
    """Энергия (кДж/моль) и силы (кДж/моль/Å) реакционного поля.

    pos (N,3), typ (N,) — индексы элементов; pi, pj — пары соседей (i<j),
    npair — число пар.  Результаты: forces (N,3) (перезаписывается),
    bond_order (npair,) = bσ+bπ+τ, ebreak (N_EBREAK,), atomout (N, N_ATOMOUT).
    Возвращает полную потенциальную энергию.
    """
    N = pos.shape[0]
    P = npair
    rsw = scal[S_RSW]
    rcut = scal[S_RCUT]
    c_rep = scal[S_CREP]
    c_disp = scal[S_CDISP]
    kmin = scal[S_KMIN]
    kover = scal[S_KOVER]
    kspare = scal[S_KSPARE]
    kpi = scal[S_KPI]
    klp = scal[S_KLP]
    epsw = scal[S_EPSW]
    ke = scal[S_KE]
    as3 = scal[S_AS3]
    hc, dhc = _dsf_h(rcut, as3)

    for a in range(N):
        forces[a, 0] = 0.0
        forces[a, 1] = 0.0
        forces[a, 2] = 0.0
    for k in range(ebreak.shape[0]):
        ebreak[k] = 0.0

    # ---- рабочие массивы по парам
    dvec = np.empty((P, 3))
    rr = np.empty(P)
    zz = np.zeros(P)
    dzz = np.zeros(P)
    zl = np.zeros(P)
    dzl = np.zeros(P)
    zbi = np.zeros(P)
    zbj = np.zeros(P)
    fi_ = np.zeros(P)
    fj_ = np.zeros(P)
    dfi = np.zeros(P)
    dfj = np.zeros(P)
    fs = np.zeros(P)
    dfsa = np.zeros(P)
    dfsb = np.zeros(P)
    bs = np.zeros(P)
    wij = np.zeros(P)
    wji = np.zeros(P)
    da1 = np.zeros(P)
    db1 = np.zeros(P)
    da2 = np.zeros(P)
    m2 = np.zeros(P)
    bp = np.zeros(P)
    gr = np.zeros(P)
    gz = np.zeros(P)
    gbs = np.zeros(P)
    gbp = np.zeros(P)
    pwij = np.zeros(P)
    pwji = np.zeros(P)
    gzbi = np.zeros(P)
    gzbj = np.zeros(P)
    # α, заряды, исключение 1-3
    cci = np.zeros(P)
    ccj = np.zeros(P)
    dcci = np.zeros(P)
    dccj = np.zeros(P)
    gor = np.zeros(P)
    # «хвост» (зарождающаяся связь радикалов)
    wt = np.zeros(P)
    uti = np.zeros(P)
    utj = np.zeros(P)
    tau = np.zeros(P)
    tphi = np.ones(P)       # множитель «углового вытеснения» хвоста
    tda = np.zeros(P)
    tdb = np.zeros(P)
    tdw = np.zeros(P)
    tfi = np.zeros(P)
    tfj = np.zeros(P)
    dtfi = np.zeros(P)      # dc/d(V − Bt + b)
    dtfj = np.zeros(P)
    gtfi = np.zeros(P)
    gtfj = np.zeros(P)
    gtau = np.zeros(P)
    pui = np.zeros(P)
    puj = np.zeros(P)
    nn = np.zeros(P)
    dnri = np.zeros(P)
    dnrj = np.zeros(P)
    e13 = np.zeros(P)
    de13r = np.zeros(P)
    de13z = np.zeros(P)
    de13o = np.zeros(P)
    de13qi = np.zeros(P)
    de13qj = np.zeros(P)
    s13 = np.zeros(P)
    gs13 = np.zeros(P)
    ecp = np.zeros(P)       # кулоновская часть пары (для исключения 1-4)
    decr = np.zeros(P)
    decz = np.zeros(P)
    decqi = np.zeros(P)
    decqj = np.zeros(P)
    s14 = np.zeros(P)
    gs14 = np.zeros(P)
    # ---- по атомам
    Z = np.zeros(N)
    Bs = np.zeros(N)
    sp = np.zeros(N)
    dsp = np.zeros(N)
    W = np.zeros(N)
    Bp = np.zeros(N)
    Bt = np.zeros(N)
    Ut = np.zeros(N)
    gUt = np.zeros(N)
    c0 = np.zeros(N)
    dc0 = np.zeros(N)
    dlp = np.zeros(N)
    q = np.zeros(N)
    gq = np.zeros(N)
    gBs = np.zeros(N)
    gBt = np.zeros(N)
    gs = np.zeros(N)
    gW = np.zeros(N)
    gc0 = np.zeros(N)

    # ================= прямой проход =================
    # 1. расстояния и «весовые» функции связи z(r), zL(r)
    for p in range(P):
        i = pi[p]
        j = pj[p]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        if r < 1e-6:
            r = 1e-6
        dvec[p, 0] = dx
        dvec[p, 1] = dy
        dvec[p, 2] = dz
        rr[p] = r
        ti = typ[i]
        tj = typ[j]
        if pairpar[ti, tj, P_BOND] > 0.0:
            z, dzr = _taper(r, pairpar[ti, tj, P_RON], pairpar[ti, tj, P_ROFF])
            zz[p] = z
            dzz[p] = dzr
            Z[i] += z
            Z[j] += z
            zl[p], dzl[p] = _taper(r, pairpar[ti, tj, P_RONL], pairpar[ti, tj, P_ROFFL])

    # 2. списки соседей в окне связывания и координация, видимая из связи
    bdeg = np.zeros(N + 1, dtype=np.int64)
    for p in range(P):
        if zz[p] > 0.0:
            bdeg[pi[p] + 1] += 1
            bdeg[pj[p] + 1] += 1
    for a in range(N):
        bdeg[a + 1] += bdeg[a]
    badj = np.empty(bdeg[N], dtype=np.int64)
    bfill = bdeg[:N].copy()
    for p in range(P):
        if zz[p] > 0.0:
            badj[bfill[pi[p]]] = p
            bfill[pi[p]] += 1
            badj[bfill[pj[p]]] = p
            bfill[pj[p]] += 1
    # Z_ij = z_ij + Σ_k z_ik·w_jk·g(θ_jik):
    #   w_jk = min(1, D_ik/D_ij) — более слабая связь не может «отнять» валентность у
    #   более прочной (электроны идут туда, где энергия ниже; ср. метод BEBO);
    #   g = 1 + c·cos²θ при θ < 90° — соседи под острым углом конкурируют сильнее.
    for p in range(P):
        if zz[p] > 0.0:
            zbi[p] = zz[p]
            zbj[p] = zz[p]
    for a in range(N):
        ta = typ[a]
        cg = elempar[ta, E_GACUTE]
        for u_ in range(bdeg[a], bdeg[a + 1]):
            p1 = badj[u_]
            ux, uy, uz, o1 = _side_vec(p1, a, pi, pj, dvec)
            d1 = pairpar[ta, typ[o1], P_D1]
            for v_ in range(u_ + 1, bdeg[a + 1]):
                p2 = badj[v_]
                vx, vy, vz, o2 = _side_vec(p2, a, pi, pj, dvec)
                d2 = pairpar[ta, typ[o2], P_D1]
                w12 = d2 / d1 if d2 < d1 else 1.0
                w21 = d1 / d2 if d1 < d2 else 1.0
                g = 1.0
                if cg > 0.0:
                    cs = (ux * vx + uy * vy + uz * vz) / (rr[p1] * rr[p2])
                    if cs > 0.0:
                        g = 1.0 + cg * cs * cs
                if pi[p1] == a:
                    zbi[p1] += zz[p2] * w12 * g
                else:
                    zbj[p1] += zz[p2] * w12 * g
                if pi[p2] == a:
                    zbi[p2] += zz[p1] * w21 * g
                else:
                    zbj[p2] += zz[p1] * w21 * g

    # 3. порядок σ-связи
    for p in range(P):
        z = zz[p]
        if z > 0.0:
            i = pi[p]
            j = pj[p]
            Vi = elempar[typ[i], E_VAL]
            Vj = elempar[typ[j], E_VAL]
            fi_[p], dfi[p] = _satur(zbi[p], Vi, pairpar[typ[i], typ[j], P_DI], kover)
            fj_[p], dfj[p] = _satur(zbj[p], Vj, pairpar[typ[j], typ[i], P_DI], kover)
            v, a_, b_ = _smin(fi_[p], fj_[p], kmin)
            fs[p] = v
            dfsa[p] = a_
            dfsb[p] = b_
            b = z * v
            bs[p] = b
            Bs[i] += b
            Bs[j] += b

    # 4. свободная (неиспользованная σ-связями) валентность
    for a in range(N):
        V = elempar[typ[a], E_VAL]
        if V > 0.0:
            sp[a], dsp[a] = _ramp(V - Bs[a], kspare)

    # 5. веса распределения π-связей
    for p in range(P):
        z = zz[p]
        if z > 0.0:
            i = pi[p]
            j = pj[p]
            if pairpar[typ[i], typ[j], P_PIMAX] > 0.0:
                wij[p] = z * sp[j]
                wji[p] = z * sp[i]
                W[i] += wij[p]
                W[j] += wji[p]

    # 6. порядок π-связи
    for p in range(P):
        z = zz[p]
        if z > 0.0:
            i = pi[p]
            j = pj[p]
            pimax = pairpar[typ[i], typ[j], P_PIMAX]
            if pimax > 0.0:
                Di = W[i] + epsw
                Dj = W[j] + epsw
                oij = sp[i] * wij[p] / Di
                oji = sp[j] * wji[p] / Dj
                v1, a1_, b1_ = _smin(oij, oji, kmin)
                v2, a2_, _ = _smin(v1, pimax, kmin)
                da1[p] = a1_
                db1[p] = b1_
                da2[p] = a2_
                m2[p] = v2
                b = z * v2
                bp[p] = b
                Bp[i] += b
                Bp[j] += b
    for a in range(N):
        Bt[a] = Bs[a] + Bp[a]

    # 6b. дальнее притяжение радикалов («хвост» Морзе).
    #   τ_ij = (1−z)·zL · c_i·c_j · n_ij,
    # где c — свободная валентность атома (без учёта связи с этим же партнёром),
    # а n_ij ≤ 1 нормирует сумму «хвостов» атома, если радикальных партнёров
    # несколько: Σ_k (1−z)zL·c_k ≤ 1.  Для изолированной пары радикалов bσ + τ
    # воспроизводит кривую Морзе, рекомбинация идёт без барьера, а один радикал не
    # может притягивать сразу многих.
    for p in range(P):
        if zl[p] > 0.0 and zz[p] < 1.0:
            i = pi[p]
            j = pj[p]
            b = bs[p] + bp[p]
            tfi[p], dtfi[p] = _clamp01(elempar[typ[i], E_VAL] - Bt[i] + b, kmin)
            tfj[p], dtfj[p] = _clamp01(elempar[typ[j], E_VAL] - Bt[j] + b, kmin)
            w_ = (1.0 - zz[p]) * zl[p]
            wt[p] = w_
            Ut[i] += w_ * tfj[p]
            Ut[j] += w_ * tfi[p]
    for p in range(P):
        if wt[p] > 0.0 and tfi[p] > 0.0 and tfj[p] > 0.0:
            i = pi[p]
            j = pj[p]
            ri = 1.0 / (Ut[i] + 1e-9)
            rj = 1.0 / (Ut[j] + 1e-9)
            m1, a1_, b1_ = _smin(ri, rj, kmin)
            n_, a2_, _ = _smin(m1, 1.0, kmin)
            prod = tfi[p] * tfj[p]
            tau[p] = wt[p] * prod * n_
            tdw[p] = prod * n_                     # dτ/dw
            tda[p] = wt[p] * prod * a2_ * a1_      # dτ/dr_i
            tdb[p] = wt[p] * prod * a2_ * b1_      # dτ/dr_j
            uti[p] = wt[p] * tfj[p] * n_           # dτ/dc_i
            utj[p] = wt[p] * tfi[p] * n_           # dτ/dc_j

    # 6c. угловое вытеснение хвоста: если у радикала есть другие партнёры под
    # острым углом к направлению хвоста, они претендуют на ту же орбиталь
    # (как g(θ) в потенциале Терсоффа).  τ ← τ/(1 + C_i + C_j),
    # C_i = Σ_k ρ_ik·c_g·cos²θ_jik (θ < 90°),  ρ = z + τ (связь + хвост партнёра).
    tdeg = np.zeros(N + 1, dtype=np.int64)
    ntail = 0
    for p in range(P):
        if zz[p] > 0.0 or zl[p] > 0.0:
            tdeg[pi[p] + 1] += 1
            tdeg[pj[p] + 1] += 1
        if tau[p] > 0.0:
            ntail += 1
    tadj = np.empty(0, dtype=np.int64)
    if ntail > 0:
        for a in range(N):
            tdeg[a + 1] += tdeg[a]
        tadj = np.empty(tdeg[N], dtype=np.int64)
        tfill = tdeg[:N].copy()
        for p in range(P):
            if zz[p] > 0.0 or zl[p] > 0.0:
                tadj[tfill[pi[p]]] = p
                tfill[pi[p]] += 1
                tadj[tfill[pj[p]]] = p
                tfill[pj[p]] += 1
        for p in range(P):
            if tau[p] > 0.0:
                cc = 0.0
                for side in range(2):
                    a = pi[p] if side == 0 else pj[p]
                    cg = elempar[typ[a], E_GACUTE]
                    if cg <= 0.0:
                        continue
                    ux, uy, uz, o1 = _side_vec(p, a, pi, pj, dvec)
                    for q_ in range(tdeg[a], tdeg[a + 1]):
                        qp = tadj[q_]
                        if qp == p:
                            continue
                        vx, vy, vz, o2 = _side_vec(qp, a, pi, pj, dvec)
                        cs = (ux * vx + uy * vy + uz * vz) / (rr[p] * rr[qp])
                        if cs > 0.0:
                            rho = zz[qp] + tau[qp]
                            cc += rho * cg * cs * cs
                tphi[p] = 1.0 / (1.0 + cc)

    # 7. стерическое число и равновесный угол (VSEPR)
    for a in range(N):
        t = typ[a]
        if elempar[t, E_KANG] > 0.0:
            lp, dl = _ramp(elempar[t, E_EVAL] - Bt[a], klp)
            sn = Bs[a] + 0.5 * lp
            e2, de2 = _ramp(sn - 2.0, klp)
            snc = 2.0 + e2
            # идеальный угол правильного симплекса + сжатие неподелёнными парами
            # (VSEPR: NH3 107°, H2O 104.5°)
            c0[a] = -1.0 / (snc - 1.0) + LP_SQUEEZE * 0.5 * lp
            dc0[a] = de2 / ((snc - 1.0) * (snc - 1.0))   # dc0/dSN
            dlp[a] = -0.5 * dl                            # dSN/dBt = dLP/dBt при фикс. Bs

    # 7b. частичные заряды: перенос I по каждой единице валентности связи
    if ke > 0.0:
        for p in range(P):
            b = bs[p] + bp[p]
            if b > 0.0:
                i = pi[p]
                j = pj[p]
                qt = pairpar[typ[i], typ[j], P_QT]
                if qt != 0.0:
                    ri = elempar[typ[i], E_VAL] / Bt[i]
                    rj = elempar[typ[j], E_VAL] / Bt[j]
                    n1, a1_, b1_ = _smin(ri, rj, kmin)
                    n2, a2_, _ = _smin(n1, 1.0, kmin)
                    nn[p] = n2
                    dnri[p] = a2_ * a1_
                    dnrj[p] = a2_ * b1_
                    x = b * n2 * qt
                    q[i] += x
                    q[j] -= x

    # 8. парные энергии и их прямые производные
    etot = 0.0
    for p in range(P):
        i = pi[p]
        j = pj[p]
        ti = typ[i]
        tj = typ[j]
        r = rr[p]
        if r >= rcut:
            continue
        T, dT = _taper(r, rsw, rcut)
        # ван-дер-ваальсовы члены
        eps = pairpar[ti, tj, P_EPS]
        x6 = pairpar[ti, tj, P_X6]
        r2 = r * r
        r5 = r2 * r2 * r
        r6 = r5 * r
        rs6 = r6 + pairpar[ti, tj, P_GR6]
        qq = x6 / rs6
        rep0 = c_rep * eps * qq * qq
        drep0 = -12.0 * c_rep * eps * x6 * x6 * r5 / (rs6 * rs6 * rs6)
        den = r6 + pairpar[ti, tj, P_GD6]
        dsp0 = -c_disp * eps * x6 / den
        ddsp0 = 6.0 * c_disp * eps * x6 * r5 / (den * den)
        rep = rep0 * T
        drep = drep0 * T + rep0 * dT
        edsp = dsp0 * T
        ddsp = ddsp0 * T + dsp0 * dT
        ebreak[EB_DISP] += edsp
        g = ddsp
        e = edsp
        # кулоновский член (без множителя исключения 1-2): затухающая сила со
        # сдвигом (DSF, Fennell & Gezelter 2006) — энергия и сила плавно обращаются
        # в ноль на r_cut, а нейтральные молекулы не «видят» ложных зарядов на границе
        ecl = 0.0
        decl = 0.0
        gqi = 0.0
        gqj = 0.0
        if ke > 0.0 and q[i] != 0.0 and q[j] != 0.0:
            u_, du_ = _dsf_h(r, as3)
            u_ = u_ - hc - dhc * (r - rcut)
            du_ = du_ - dhc
            ecl = ke * q[i] * q[j] * u_
            decl = ke * q[i] * q[j] * du_
            gqi = ke * q[j] * u_
            gqj = ke * q[i] * u_
        if pairpar[ti, tj, P_BOND] > 0.0:
            z = zz[p]
            D1 = pairpar[ti, tj, P_D1]
            a1 = pairpar[ti, tj, P_A1]
            ex = math.exp(-a1 * (r - pairpar[ti, tj, P_R1]))
            R1 = D1 * ex * ex * T
            dR1 = -2.0 * a1 * D1 * ex * ex * T + D1 * ex * ex * dT
            A1 = 2.0 * D1 * ex
            dA1 = -a1 * A1
            # свободная валентность атомов для этого партнёра
            b = bs[p] + bp[p]
            ci, dci = _clamp01(elempar[ti, E_VAL] - Bt[i] + b, kmin)
            cj, dcj = _clamp01(elempar[tj, E_VAL] - Bt[j] + b, kmin)
            # у металла есть электроны проводимости — замкнутой оболочки нет
            if elempar[ti, E_METAL] > 0.0:
                ci = 1.0
                dci = 0.0
            if elempar[tj, E_METAL] > 0.0:
                cj = 1.0
                dcj = 0.0
            cci[p] = ci
            ccj[p] = cj
            dcci[p] = dci
            dccj[p] = dcj
            aor = ci + cj - ci * cj
            # σ: Морзе в форме Терсоффа + хвост притяжения радикалов
            att = bs[p] + tau[p] * tphi[p]
            esig = R1 - att * A1
            e += esig
            ebreak[EB_SIGMA] += esig
            g += dR1 - att * dA1
            gbs[p] += -A1
            gtau[p] += -A1 * tphi[p]
            gphi_ = -A1 * tau[p]
            if tphi[p] < 1.0 and gphi_ != 0.0:
                # φ = 1/(1+C): dE/dC = −gφ·φ²
                gC = -gphi_ * tphi[p] * tphi[p]
                for side in range(2):
                    a = i if side == 0 else j
                    cg = elempar[typ[a], E_GACUTE]
                    if cg <= 0.0:
                        continue
                    ux, uy, uz, o1 = _side_vec(p, a, pi, pj, dvec)
                    for q_ in range(tdeg[a], tdeg[a + 1]):
                        qp = tadj[q_]
                        if qp == p:
                            continue
                        vx, vy, vz, o2 = _side_vec(qp, a, pi, pj, dvec)
                        cs = (ux * vx + uy * vy + uz * vz) / (rr[p] * rr[qp])
                        if cs > 0.0:
                            rho = zz[qp] + tau[qp]
                            h = cg * cs * cs
                            grho = gC * h
                            gz[qp] += grho
                            gtau[qp] += grho
                            gcs = gC * rho * 2.0 * cg * cs
                            _add_cos_force(forces, a, o1, o2, ux, uy, uz, rr[p],
                                           vx, vy, vz, rr[qp], cs, gcs)
            # π
            if bp[p] > 0.0:
                Pv = pairpar[ti, tj, P_CP] * math.exp(-pairpar[ti, tj, P_AP] * (r - pairpar[ti, tj, P_R2]))
                dPv = -pairpar[ti, tj, P_AP] * Pv
                h2, dh2 = _ramp(bp[p] - 1.0, kpi)
                h1 = bp[p] - h2
                epi = -h1 * Pv
                g += -h1 * dPv
                gbp[p] += -Pv * (1.0 - dh2)
                if pairpar[ti, tj, P_PIMAX] > 1.0 and h2 > 0.0:
                    Qv = pairpar[ti, tj, P_CQ] * math.exp(-pairpar[ti, tj, P_AQ] * (r - pairpar[ti, tj, P_R3]))
                    dQv = -pairpar[ti, tj, P_AQ] * Qv
                    epi += -h2 * Qv
                    g += -h2 * dQv
                    gbp[p] += -Qv * dh2
                e += epi
                ebreak[EB_PI] += epi
            # замкнуто-оболочечное отталкивание (выключено, если есть радикал).
            # На расстояниях связывания (r < r1 + 1.5/a1) его заменяет R1, поэтому
            # оно плавно выключается «хвостовым» окном zL.
            ozl = 1.0 - zl[p]
            wrep = ozl * (1.0 - aor)
            e += wrep * rep
            ebreak[EB_REP] += wrep * rep
            g += wrep * drep - dzl[p] * (1.0 - aor) * rep
            gor[p] += -ozl * rep
            # кулон: связанные пары (z=1) исключены — ионный вклад уже в D
            om = 1.0 - z
            e += om * ecl
            ebreak[EB_COUL] += om * ecl
            g += om * decl
            gz[p] += -ecl
            gq[i] += om * gqi
            gq[j] += om * gqj
            # «отталкивательно-электростатическая» часть пары — для исключения 1-3
            e13[p] = R1 + wrep * rep + om * ecl
            de13r[p] = dR1 + wrep * drep - dzl[p] * (1.0 - aor) * rep + om * decl
            de13z[p] = -ecl
            de13o[p] = -ozl * rep
            de13qi[p] = om * gqi
            de13qj[p] = om * gqj
            ecp[p] = om * ecl
            decr[p] = om * decl
            decz[p] = -ecl
            decqi[p] = om * gqi
            decqj[p] = om * gqj
        else:
            e += rep + ecl
            ebreak[EB_REP] += rep
            ebreak[EB_COUL] += ecl
            g += drep + decl
            gq[i] += gqi
            gq[j] += gqj
        gr[p] += g
        etot += e

    # 9. тройки i–c–j по σ-связям: углы VSEPR и исключение 1-3
    # 9a. полные списки соседей (для поиска пары 1-3)
    nstart = np.zeros(N + 1, dtype=np.int64)
    for p in range(P):
        nstart[pi[p] + 1] += 1
        nstart[pj[p] + 1] += 1
    for a in range(N):
        nstart[a + 1] += nstart[a]
    nbr = np.empty(2 * P, dtype=np.int64)
    npr = np.empty(2 * P, dtype=np.int64)
    nfill = nstart[:N].copy()
    for p in range(P):
        a = pi[p]
        nbr[nfill[a]] = pj[p]
        npr[nfill[a]] = p
        nfill[a] += 1
        a = pj[p]
        nbr[nfill[a]] = pi[p]
        npr[nfill[a]] = p
        nfill[a] += 1
    # 9b. σ-смежность
    deg = np.zeros(N + 1, dtype=np.int64)
    for p in range(P):
        if bs[p] > 1e-8:
            deg[pi[p] + 1] += 1
            deg[pj[p] + 1] += 1
    for a in range(N):
        deg[a + 1] += deg[a]
    adj = np.empty(deg[N], dtype=np.int64)
    fill = deg[:N].copy()
    ntrip = 0
    for p in range(P):
        if bs[p] > 1e-8:
            adj[fill[pi[p]]] = p
            fill[pi[p]] += 1
            adj[fill[pj[p]]] = p
            fill[pj[p]] += 1
    for a in range(N):
        d_ = deg[a + 1] - deg[a]
        ntrip += d_ * (d_ - 1) // 2
    tc = np.empty(ntrip, dtype=np.int64)
    tp1 = np.empty(ntrip, dtype=np.int64)
    tp2 = np.empty(ntrip, dtype=np.int64)
    tp13 = np.empty(ntrip, dtype=np.int64)
    t = 0
    for c in range(N):
        for u_ in range(deg[c], deg[c + 1]):
            p1 = adj[u_]
            o1 = pj[p1] if pi[p1] == c else pi[p1]
            for v_ in range(u_ + 1, deg[c + 1]):
                p2 = adj[v_]
                o2 = pj[p2] if pi[p2] == c else pi[p2]
                p13 = -1
                for q_ in range(nstart[o1], nstart[o1 + 1]):
                    if nbr[q_] == o2:
                        p13 = npr[q_]
                        break
                tc[t] = c
                tp1[t] = p1
                tp2[t] = p2
                tp13[t] = p13
                t += 1
    # 9c. углы + накопление весов исключения
    eang = 0.0
    for t in range(ntrip):
        c = tc[t]
        p1 = tp1[t]
        p2 = tp2[t]
        w = bs[p1] * bs[p2]
        p13 = tp13[t]
        if p13 >= 0 and e13[p13] != 0.0:
            s13[p13] += w * (1.0 - zz[p13])
        K = elempar[typ[c], E_KANG]
        if K <= 0.0:
            continue
        ux, uy, uz, o1 = _side_vec(p1, c, pi, pj, dvec)
        vx, vy, vz, o2 = _side_vec(p2, c, pi, pj, dvec)
        ru = rr[p1]
        rv = rr[p2]
        cs = (ux * vx + uy * vy + uz * vz) / (ru * rv)
        dc = cs - c0[c]
        eang += K * w * dc * dc
        gcs = 2.0 * K * w * dc
        gw = K * dc * dc
        gbs[p1] += gw * bs[p2]
        gbs[p2] += gw * bs[p1]
        gc0[c] += -gcs
        _add_cos_force(forces, c, o1, o2, ux, uy, uz, ru, vx, vy, vz, rv, cs, gcs)
    ebreak[EB_ANGLE] = eang
    etot += eang
    # 9d. исключение 1-3 с насыщением веса (кольца: несколько общих соседей)
    eexcl = 0.0
    for p in range(P):
        if s13[p] > 0.0:
            x_, dx_, _ = _smin(s13[p], 1.0, kmin)
            eexcl -= x_ * e13[p]
            gs13[p] = -dx_ * e13[p]
            gr[p] -= x_ * de13r[p]
            gz[p] -= x_ * de13z[p]
            gor[p] -= x_ * de13o[p]
            gq[pi[p]] -= x_ * de13qi[p]
            gq[pj[p]] -= x_ * de13qj[p]
    for t in range(ntrip):
        p13 = tp13[t]
        if p13 >= 0 and gs13[p13] != 0.0:
            p1 = tp1[t]
            p2 = tp2[t]
            om = 1.0 - zz[p13]
            g_ = gs13[p13]
            gbs[p1] += g_ * om * bs[p2]
            gbs[p2] += g_ * om * bs[p1]
            gz[p13] -= g_ * bs[p1] * bs[p2]
    ebreak[EB_REP] += eexcl
    etot += eexcl

    # 9f. пары 1-4 (a–b–c–d): кулон ослабляется до SCALE14 (как в силовых полях)
    nquad = 0
    if ke > 0.0:
        for p in range(P):
            if bs[p] > 1e-8:
                b_ = pi[p]
                c_ = pj[p]
                nquad += (deg[b_ + 1] - deg[b_] - 1) * (deg[c_ + 1] - deg[c_] - 1)
    qa = np.empty(nquad, dtype=np.int64)
    qb = np.empty(nquad, dtype=np.int64)
    qc = np.empty(nquad, dtype=np.int64)
    qd = np.empty(nquad, dtype=np.int64)
    nq = 0
    if nquad > 0:
        for p in range(P):
            if bs[p] <= 1e-8:
                continue
            b_ = pi[p]
            c_ = pj[p]
            for u_ in range(deg[b_], deg[b_ + 1]):
                p1 = adj[u_]
                if p1 == p:
                    continue
                a_ = pj[p1] if pi[p1] == b_ else pi[p1]
                for v_ in range(deg[c_], deg[c_ + 1]):
                    p2 = adj[v_]
                    if p2 == p:
                        continue
                    d_ = pj[p2] if pi[p2] == c_ else pi[p2]
                    if d_ == a_ or d_ == b_ or a_ == c_:
                        continue
                    p14 = -1
                    for q_ in range(nstart[a_], nstart[a_ + 1]):
                        if nbr[q_] == d_:
                            p14 = npr[q_]
                            break
                    if p14 < 0 or ecp[p14] == 0.0:
                        continue
                    qa[nq] = p1
                    qb[nq] = p
                    qc[nq] = p2
                    qd[nq] = p14
                    nq += 1
                    s14[p14] += bs[p1] * bs[p] * bs[p2] * (1.0 - zz[p14])
        e14 = 0.0
        for p in range(P):
            if s14[p] > 0.0:
                x_, dx_, _ = _smin(s14[p], 1.0, kmin)
                f_ = (1.0 - SCALE14) * x_
                e14 -= f_ * ecp[p]
                gs14[p] = -(1.0 - SCALE14) * dx_ * ecp[p]
                gr[p] -= f_ * decr[p]
                gz[p] -= f_ * decz[p]
                gq[pi[p]] -= f_ * decqi[p]
                gq[pj[p]] -= f_ * decqj[p]
        for t in range(nq):
            p14 = qd[t]
            g_ = gs14[p14]
            if g_ != 0.0:
                p1 = qa[t]
                p = qb[t]
                p2 = qc[t]
                om = 1.0 - zz[p14]
                gbs[p1] += g_ * om * bs[p] * bs[p2]
                gbs[p] += g_ * om * bs[p1] * bs[p2]
                gbs[p2] += g_ * om * bs[p1] * bs[p]
                gz[p14] -= g_ * bs[p1] * bs[p] * bs[p2]
        ebreak[EB_COUL] += e14
        etot += e14

    # ================= обратный проход =================
    # 9e. хвост радикалов -> w, c, U -> (z, r, Bt, b)
    for p in range(P):
        if tau[p] > 0.0:
            i = pi[p]
            j = pj[p]
            gt = gtau[p]
            gtfi[p] += gt * uti[p]
            gtfj[p] += gt * utj[p]
            # r = 1/(U + ε)
            ri = 1.0 / (Ut[i] + 1e-9)
            rj = 1.0 / (Ut[j] + 1e-9)
            gUt[i] -= gt * tda[p] * ri * ri
            gUt[j] -= gt * tdb[p] * rj * rj
            gtau[p] = gt * tdw[p]          # прямой вклад в dE/dw
        else:
            gtau[p] = 0.0
    for p in range(P):
        if wt[p] > 0.0:
            i = pi[p]
            j = pj[p]
            # U_i = Σ w·c_j,  U_j = Σ w·c_i
            gwt = gtau[p] + gUt[i] * tfj[p] + gUt[j] * tfi[p]
            gtfj[p] += gUt[i] * wt[p]
            gtfi[p] += gUt[j] * wt[p]
            gz[p] -= gwt * zl[p]
            gr[p] += gwt * (1.0 - zz[p]) * dzl[p]
            x_ = gtfi[p] * dtfi[p]
            y_ = gtfj[p] * dtfj[p]
            gBt[i] -= x_
            gBt[j] -= y_
            gbs[p] += x_ + y_
            gbp[p] += x_ + y_

    # 10a. α∨ -> c_i, c_j -> (Bt_i, b_ij)
    for p in range(P):
        if gor[p] != 0.0:
            i = pi[p]
            j = pj[p]
            ci = cci[p]
            cj = ccj[p]
            gFi = gor[p] * (1.0 - cj) * dcci[p]
            gFj = gor[p] * (1.0 - ci) * dccj[p]
            gBt[i] -= gFi
            gBt[j] -= gFj
            gbs[p] += gFi + gFj
            gbp[p] += gFi + gFj
    # 10b. заряды -> порядки связей и Bt
    if ke > 0.0:
        for p in range(P):
            if nn[p] > 0.0:
                i = pi[p]
                j = pj[p]
                qt = pairpar[typ[i], typ[j], P_QT]
                gx = gq[i] - gq[j]
                if gx != 0.0:
                    b = bs[p] + bp[p]
                    gb = gx * nn[p] * qt
                    gbs[p] += gb
                    gbp[p] += gb
                    gn = gx * b * qt
                    Vi = elempar[typ[i], E_VAL]
                    Vj = elempar[typ[j], E_VAL]
                    gBt[i] -= gn * dnri[p] * Vi / (Bt[i] * Bt[i])
                    gBt[j] -= gn * dnrj[p] * Vj / (Bt[j] * Bt[j])
    # 10c. угол -> стерическое число -> Bσ, Bt
    for a in range(N):
        if gc0[a] != 0.0:
            gsn = gc0[a] * dc0[a]
            gBt[a] += (gsn + gc0[a] * LP_SQUEEZE) * dlp[a]
            gBs[a] += gsn
        gBs[a] += gBt[a]          # Bt = Bσ + Bπ

    # 11a. bπ -> предложения o_ij -> s, W
    for p in range(P):
        if bp[p] > 0.0 or wij[p] > 0.0 or wji[p] > 0.0:
            i = pi[p]
            j = pj[p]
            z = zz[p]
            gb = gbp[p] + gBt[i] + gBt[j]
            gz[p] += gb * m2[p]
            gm1 = gb * z * da2[p]
            goij = gm1 * da1[p]
            goji = gm1 * db1[p]
            Di = W[i] + epsw
            Dj = W[j] + epsw
            gs[i] += goij * wij[p] / Di
            gs[j] += goji * wji[p] / Dj
            gW[i] -= goij * sp[i] * wij[p] / (Di * Di)
            gW[j] -= goji * sp[j] * wji[p] / (Dj * Dj)
            pwij[p] = goij * sp[i] / Di
            pwji[p] = goji * sp[j] / Dj
    # 11b. веса w -> z, s
    for p in range(P):
        if wij[p] > 0.0 or wji[p] > 0.0 or bp[p] > 0.0:
            i = pi[p]
            j = pj[p]
            z = zz[p]
            gwij = pwij[p] + gW[i]
            gwji = pwji[p] + gW[j]
            gz[p] += gwij * sp[j] + gwji * sp[i]
            gs[j] += gwij * z
            gs[i] += gwji * z
    # 12. s = ramp(V − Bσ)
    for a in range(N):
        gBs[a] -= gs[a] * dsp[a]
    # 13. Bσ -> bσ -> z, f_ij -> Z_ij
    for p in range(P):
        z = zz[p]
        if z > 0.0:
            i = pi[p]
            j = pj[p]
            gb = gbs[p] + gBs[i] + gBs[j]
            gz[p] += gb * fs[p]
            gfs = gb * z
            gzbi[p] = gfs * dfsa[p] * dfi[p]
            gzbj[p] = gfs * dfsb[p] * dfj[p]
            gz[p] += gzbi[p] + gzbj[p]          # собственный вклад z_ij в Z_ij
    # 14. Z_ij = z_ij + Σ_k z_ik·w·g(θ)
    for a in range(N):
        ta = typ[a]
        cg = elempar[ta, E_GACUTE]
        for u_ in range(bdeg[a], bdeg[a + 1]):
            p1 = badj[u_]
            ux, uy, uz, o1 = _side_vec(p1, a, pi, pj, dvec)
            d1 = pairpar[ta, typ[o1], P_D1]
            g1s = gzbi[p1] if pi[p1] == a else gzbj[p1]
            for v_ in range(u_ + 1, bdeg[a + 1]):
                p2 = badj[v_]
                g2s = gzbi[p2] if pi[p2] == a else gzbj[p2]
                if g1s == 0.0 and g2s == 0.0:
                    continue
                vx, vy, vz, o2 = _side_vec(p2, a, pi, pj, dvec)
                d2 = pairpar[ta, typ[o2], P_D1]
                w12 = d2 / d1 if d2 < d1 else 1.0
                w21 = d1 / d2 if d1 < d2 else 1.0
                g = 1.0
                cs = 0.0
                if cg > 0.0:
                    cs = (ux * vx + uy * vy + uz * vz) / (rr[p1] * rr[p2])
                    if cs > 0.0:
                        g = 1.0 + cg * cs * cs
                gz[p2] += g1s * w12 * g
                gz[p1] += g2s * w21 * g
                if cg > 0.0 and cs > 0.0:
                    gcs = (g1s * zz[p2] * w12 + g2s * zz[p1] * w21) * 2.0 * cg * cs
                    if gcs != 0.0:
                        _add_cos_force(forces, a, o1, o2, ux, uy, uz, rr[p1], vx, vy, vz,
                                       rr[p2], cs, gcs)
    # 15. Z_i -> z -> r;  силы
    for p in range(P):
        i = pi[p]
        j = pj[p]
        if dzz[p] != 0.0:
            gr[p] += gz[p] * dzz[p]
        s = gr[p] / rr[p]
        fx = s * dvec[p, 0]
        fy = s * dvec[p, 1]
        fzz = s * dvec[p, 2]
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fzz
        forces[j, 0] -= fx
        forces[j, 1] -= fy
        forces[j, 2] -= fzz
        # эффективный порядок связи для анализа: σ + π + «хвост» (растянутая
        # колебательно-возбуждённая связь ещё не считается разорванной)
        bond_order[p] = bs[p] + bp[p] + tau[p] * tphi[p]

    tsum = np.zeros(N)
    for p in range(P):
        if tau[p] > 0.0:
            tsum[pi[p]] += tau[p] * tphi[p]
            tsum[pj[p]] += tau[p] * tphi[p]
    for a in range(N):
        atomout[a, A_Z] = Z[a]
        fv = elempar[typ[a], E_VAL] - Bt[a] - tsum[a]
        atomout[a, A_SPARE] = fv if fv > 0.0 else 0.0
        atomout[a, A_BTOT] = Bt[a]
        atomout[a, A_SN] = 1.0 - 1.0 / c0[a] if c0[a] != 0.0 else 0.0
        atomout[a, A_Q] = q[a]
    return etot


@njit
def all_pairs(n):
    """Все пары i<j (для маленьких систем и тестов)."""
    m = n * (n - 1) // 2
    pi = np.empty(m, dtype=np.int64)
    pj = np.empty(m, dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            pi[k] = i
            pj[k] = j
            k += 1
    return pi, pj
